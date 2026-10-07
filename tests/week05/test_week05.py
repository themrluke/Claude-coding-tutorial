import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, create_mask, flex_attention

from exercises.week05 import ex01_online_softmax as ex01
from exercises.week05 import ex02_varlen as ex02
from exercises.week05 import ex03_locality as ex03
from minihep.data.dataset import ToyTrackingDataset
from minihep.flex.masks import block_sparsity, mask_mod_to_dense, sliding_window_mask, sliding_window_mask_wrapped, strided_window_mask
from minihep.models.attention import Attention
from minihep.models.encoder import Encoder
from tests.conftest import TOY_INPUTS, TOY_TARGETS

# ----------------------------------------------------------------------------- ex01 online softmax


@pytest.mark.parametrize("block_size", [1, 7, 64, 1000])
def test_logsumexp_blocks(block_size):
    scores = torch.randn(3, 5, 100) * 10
    assert torch.allclose(ex01.logsumexp_blocks(scores, block_size), torch.logsumexp(scores, -1), atol=1e-4)


@pytest.mark.parametrize("block_size", [16, 50, 128])
def test_tiled_attention_matches_sdpa(block_size):
    q, k, v = torch.randn(2, 3, 37, 16), torch.randn(2, 3, 100, 16), torch.randn(2, 3, 100, 8)
    out, lse = ex01.tiled_attention(q, k, v, block_size)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    scores = q @ k.transpose(-2, -1) / 4
    assert torch.allclose(lse, torch.logsumexp(scores, -1), atol=1e-4)


def test_tiled_attention_with_huge_scores():
    q, k, v = torch.randn(1, 4, 8) * 100, torch.randn(1, 50, 8) * 100, torch.randn(1, 50, 8)
    out, _ = ex01.tiled_attention(q, k, v, 8)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-4)


def test_merge_two_disjoint_key_sets_is_exact():
    q, k, v = torch.randn(2, 10, 8), torch.randn(2, 30, 8), torch.randn(2, 30, 4)
    out_a, lse_a = ex01.tiled_attention(q, k[:, :12], v[:, :12])
    out_b, lse_b = ex01.tiled_attention(q, k[:, 12:], v[:, 12:])
    out, lse = ex01.merge_two(out_a, lse_a, out_b, lse_b)
    full_out, full_lse = ex01.tiled_attention(q, k, v)
    assert torch.allclose(out, full_out, atol=1e-5)
    assert torch.allclose(lse, full_lse, atol=1e-5)


def test_or_merge():
    q, k, v = torch.randn(10, 8), torch.randn(30, 8), torch.randn(30, 4)
    parts = [ex01.tiled_attention(q, k[s], v[s]) for s in (slice(0, 10), slice(10, 20), slice(20, 30))]
    outs, lses = torch.stack([p[0] for p in parts]), torch.stack([p[1] for p in parts])
    assert torch.allclose(ex01.or_merge(outs, lses), F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    # Overlapping key sets: no longer equal to attention over the union
    a = ex01.tiled_attention(q, k[:20], v[:20])
    b = ex01.tiled_attention(q, k[10:], v[10:])
    merged = ex01.or_merge(torch.stack([a[0], b[0]]), torch.stack([a[1], b[1]]))
    assert not torch.allclose(merged, F.scaled_dot_product_attention(q, k, v), atol=1e-3)


# ----------------------------------------------------------------------------- ex02 varlen


def test_unpad_pad_roundtrip():
    x = torch.randn(3, 6, 4)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    packed, indices, cu, max_len = ex02.unpad(x, valid)
    assert packed.shape == (10, 4)
    assert cu.tolist() == [0, 3, 8, 10]
    assert cu.dtype == torch.int32
    assert max_len == 5
    assert torch.equal(packed[3:8], x[1, :5])
    back = ex02.pad(packed, indices, 3, 6)
    assert torch.equal(back[valid], x[valid])
    assert torch.all(back[~valid] == 0)


def test_varlen_attention_matches_masked_padded_attention():
    b, s, h, d = 3, 9, 2, 8
    valid = torch.zeros(b, s, dtype=torch.bool)
    for i, n in enumerate([4, 9, 6]):
        valid[i, :n] = True
    q, k, v = (torch.randn(b, s, h, d) for _ in range(3))
    packed = [ex02.unpad(t, valid) for t in (q, k, v)]
    out_packed = ex02.varlen_self_attention(packed[0][0], packed[1][0], packed[2][0], packed[0][2])
    out = ex02.pad(out_packed, packed[0][1], b, s)
    mask = (valid[:, None, :, None] & valid[:, None, None, :]) | torch.eye(s, dtype=torch.bool)
    ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask).transpose(1, 2)
    assert torch.allclose(out[valid], ref[valid], atol=1e-5)


# ----------------------------------------------------------------------------- ex03 locality


def test_positions_from_sort_value():
    assert ex03.positions_from_sort_value(np.array([0.3, -1.0, 2.0])).tolist() == [1, 0, 2]


def test_pair_recall_basics():
    pid = np.array([1, 1, 2, 2, 0])
    assert ex03.pair_recall(pid, np.array([0, 1, 2, 3, 4]), window_size=2) == 1.0
    assert ex03.pair_recall(pid, np.array([0, 3, 1, 4, 2]), window_size=2) == 0.0
    assert ex03.pair_recall(pid, np.array([0, 3, 1, 4, 2]), window_size=6) == 1.0
    assert ex03.pair_recall(np.array([1, 1]), np.array([0, 9]), window_size=2, seq_len=10) == 1.0
    assert ex03.pair_recall(np.array([0, 0]), np.array([0, 1]), window_size=2) == 1.0


def test_phi_sorting_helps(toy_dir):
    ds = ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS)
    hits, _ = ds.load_event(0)
    pid = hits["particle_id"].to_numpy()
    n = len(hits)
    rng = np.random.default_rng(0)
    random_pos = rng.permutation(n)
    phi_pos = ex03.positions_from_sort_value(hits["phi"].to_numpy())
    cell_pos = ex03.positions_from_sort_value(ex03.cell_sort_value(hits["eta"].to_numpy(), hits["phi"].to_numpy(), 8))
    w = 32
    r_random = ex03.pair_recall(pid, random_pos, w, n)
    r_phi = ex03.pair_recall(pid, phi_pos, w, n)
    r_cell = ex03.pair_recall(pid, cell_pos, w, n)
    assert r_phi > 2 * r_random
    assert r_cell > r_random


# ----------------------------------------------------------------------------- masks


def test_sliding_window_mask_dense():
    dense = mask_mod_to_dense(sliding_window_mask(4), 8, 8)
    expected = (torch.arange(8)[:, None] - torch.arange(8)[None, :]).abs() <= 2
    assert torch.equal(dense, expected)


def test_wrapped_window_mask_dense():
    q_len = torch.tensor([8])
    dense = mask_mod_to_dense(sliding_window_mask_wrapped(4, q_len), 8, 8)
    d = (torch.arange(8)[:, None] - torch.arange(8)[None, :]).abs()
    assert torch.equal(dense, torch.minimum(d, 8 - d) <= 2)
    q_len[0] = 100
    assert not mask_mod_to_dense(sliding_window_mask_wrapped(4, q_len), 8, 8)[0, 7]


def test_strided_window_mask():
    dense = mask_mod_to_dense(strided_window_mask(4, q_len=4, kv_len=16), 4, 16)
    assert dense.shape == (4, 16)
    assert dense[0, :3].all()
    assert not dense[0, 3]
    assert dense[2, 6:11].all()
    assert dense.sum(-1).max() == 5


def test_block_mask_skips_blocks():
    bm = create_block_mask(sliding_window_mask(64), B=None, H=None, Q_LEN=1024, KV_LEN=1024, device="cpu")
    assert block_sparsity(bm) == pytest.approx(1 - 22 / 64)  # 8 diagonal + 14 neighbouring blocks of 64 are needed
    dense = create_block_mask(lambda b, h, q, kv: q >= 0, B=None, H=None, Q_LEN=1024, KV_LEN=1024, device="cpu")
    assert block_sparsity(dense) == 0.0


# ----------------------------------------------------------------------------- flex / flash backends


def test_flex_matches_sdpa_with_the_same_mask_cpu():
    q, k, v = torch.randn(1, 2, 256, 16), torch.randn(1, 2, 256, 16), torch.randn(1, 2, 256, 16)
    mod = sliding_window_mask_wrapped(32, torch.tensor([256]))
    bm = create_block_mask(mod, B=None, H=None, Q_LEN=256, KV_LEN=256, device="cpu")
    dense = create_mask(mod, 1, 1, 256, 256, device="cpu")
    assert torch.allclose(flex_attention(q, k, v, block_mask=bm), F.scaled_dot_product_attention(q, k, v, attn_mask=dense), atol=1e-5)


def copy_attention(src: Attention, dst: Attention) -> None:
    dst.load_state_dict(src.state_dict())


def test_attention_flex_backend_cpu():
    torch_attn = Attention(32, 4, attn_type="torch")
    flex_attn = Attention(32, 4, attn_type="flex")
    copy_attention(torch_attn, flex_attn)
    x = torch.randn(1, 200, 32)
    mod = sliding_window_mask(16)
    bm = create_block_mask(mod, B=None, H=None, Q_LEN=200, KV_LEN=200, device="cpu")
    dense = create_mask(mod, 1, 1, 200, 200, device="cpu")[0]
    assert torch.allclose(flex_attn(x, attn_mask=bm), torch_attn(x, attn_mask=dense), atol=1e-5)
    with pytest.raises(ValueError):
        flex_attn(x, attn_mask=dense)


@pytest.mark.parametrize("wrap", [False, True])
def test_encoder_window_torch_vs_flex_cpu(wrap):
    kwargs = {"num_layers": 2, "dim": 32, "window_size": 32, "window_wrap": wrap, "attn_kwargs": {"num_heads": 4}}
    enc_torch = Encoder(attn_type="torch", **kwargs).eval()
    enc_flex = Encoder(attn_type="flex", **kwargs).eval()
    enc_flex.load_state_dict(enc_torch.state_dict())
    x, phi = torch.randn(1, 150, 32), torch.rand(1, 150)
    assert torch.allclose(enc_torch(x, phi), enc_flex(x, phi), atol=1e-4)
    # a window genuinely changes the answer compared with full attention
    full = Encoder(attn_type="torch", **{**kwargs, "window_size": None, "window_wrap": False}).eval()
    full.load_state_dict(enc_torch.state_dict())
    assert not torch.allclose(full(x, phi), enc_torch(x, phi), atol=1e-3)


@pytest.mark.gpu
def test_compiled_flex_on_gpu():
    torch_attn = Attention(64, 4, attn_type="torch").cuda()
    flex_attn = Attention(64, 4, attn_type="flex").cuda()
    copy_attention(torch_attn, flex_attn)
    x = torch.randn(1, 1000, 64, device="cuda")
    mod = sliding_window_mask(128)
    bm = create_block_mask(mod, B=None, H=None, Q_LEN=1000, KV_LEN=1000, device="cuda")
    dense = create_mask(mod, 1, 1, 1000, 1000, device="cuda")[0]
    assert torch.allclose(flex_attn(x, attn_mask=bm), torch_attn(x, attn_mask=dense), atol=1e-4)


@pytest.mark.gpu
@pytest.mark.flash
def test_flash_window_matches_torch_window():
    kwargs = {"num_layers": 1, "dim": 64, "window_size": 64, "attn_kwargs": {"num_heads": 4}}
    enc_torch = Encoder(attn_type="torch", **kwargs).cuda().half().eval()
    enc_flash = Encoder(attn_type="flash", **kwargs).cuda().half().eval()
    enc_flash.load_state_dict(enc_torch.state_dict())
    x = torch.randn(1, 500, 64, device="cuda", dtype=torch.float16)
    assert torch.allclose(enc_torch(x), enc_flash(x), atol=2e-2)
