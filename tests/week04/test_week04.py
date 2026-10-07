import pytest
import torch
from torch import nn

from exercises.week04 import ex01_attention_from_scratch as ex01
from minihep.data.dataset import ToyTrackingDataset
from minihep.models.attention import Attention, merge_masks
from minihep.models.dense import Dense
from minihep.models.encoder import DropPath, Encoder, EncoderLayer, LayerScale, Residual
from minihep.models.hitfilter import HitFilter
from minihep.models.input import InputNet
from minihep.models.norm import DyT, RMSNorm
from minihep.models.posenc import PositionEncoder
from minihep.models.tasks import HitFilterTask
from tests.conftest import TOY_INPUTS, TOY_TARGETS

# ----------------------------------------------------------------------------- ex01


def test_stable_softmax():
    x = torch.randn(3, 7)
    assert torch.allclose(ex01.stable_softmax(x), torch.softmax(x, -1))
    big = torch.tensor([[1000.0, 1000.0, 999.0]])
    out = ex01.stable_softmax(big)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.softmax(big, -1))
    assert torch.allclose(ex01.stable_softmax(x, dim=0), torch.softmax(x, 0))


def test_naive_attention_matches_sdpa():
    q, k, v = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 9, 8), torch.randn(2, 4, 9, 6)
    out, w = ex01.naive_attention(q, k, v)
    assert out.shape == (2, 4, 5, 6)
    assert w.shape == (2, 4, 5, 9)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 5))
    assert torch.allclose(out, torch.nn.functional.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_naive_attention_mask():
    q, k, v = torch.randn(1, 1, 5, 8), torch.randn(1, 1, 6, 8), torch.randn(1, 1, 6, 8)
    mask = torch.rand(1, 1, 5, 6) > 0.4
    mask[..., 0] = True
    out, w = ex01.naive_attention(q, k, v, mask)
    assert torch.all(w[~mask.expand_as(w)] == 0)
    assert torch.allclose(out, torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5)


def test_fully_masked_rows_make_nan_and_the_fix():
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    mask = torch.ones(1, 4, 6, dtype=torch.bool)
    mask[0, 2] = False
    assert ex01.fully_masked_rows(mask).tolist() == [[False, False, True, False]]
    out, _ = ex01.naive_attention(q, k, v, mask)
    assert torch.isnan(out[0, 2]).all()
    fixed = ex01.unmask_all_false(mask)
    assert fixed[0, 2].all()
    assert torch.equal(fixed[0, [0, 1, 3]], mask[0, [0, 1, 3]])
    out, _ = ex01.naive_attention(q, k, v, fixed)
    assert torch.isfinite(out).all()


def test_permutation_equivariance():
    a, b = ex01.attention_is_permutation_equivariant(torch.randn(2, 7, 4), torch.randn(2, 9, 4), torch.randn(2, 9, 3))
    assert torch.allclose(a, b, atol=1e-6)


# ----------------------------------------------------------------------------- norms


def test_rmsnorm_matches_torch():
    x = torch.randn(3, 5, 16)
    ours, ref = RMSNorm(16), nn.RMSNorm(16, eps=1e-6)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)
    assert sum(p.numel() for p in ours.parameters()) == 16


def test_dyt():
    m = DyT(4, alpha_init_value=0.5)
    x = torch.randn(2, 4)
    assert torch.allclose(m(x), torch.tanh(0.5 * x))
    assert {n for n, _ in m.named_parameters()} == {"alpha", "weight", "bias"}


# ----------------------------------------------------------------------------- attention module


def test_merge_masks():
    assert merge_masks(None, None, None, 3, 4, 2, "cpu") is None
    kv = torch.tensor([[True, True, False, True]])
    m = merge_masks(None, kv, None, 3, 4, 1, "cpu")
    assert m.shape == (1, 3, 4)
    assert torch.equal(m[0, 1], kv[0])
    qm = torch.tensor([[True, False, True]])
    attn = torch.ones(1, 3, 4, dtype=torch.bool)
    attn[0, 0, 0] = False
    m = merge_masks(qm, kv, attn, 3, 4, 1, "cpu")
    assert not m[0, 1].any()
    assert not m[0, 0, 0]
    assert m[0, 0, 1]


def copy_weights(src: Attention, dst: nn.MultiheadAttention) -> None:
    dst.in_proj_weight.data.copy_(src.in_proj_weight.data)
    if src.in_proj_bias is not None:
        dst.in_proj_bias.data.copy_(src.in_proj_bias.data)
    dst.out_proj.weight.data.copy_(src.out_proj.weight.data)
    if src.out_proj.bias is not None:
        dst.out_proj.bias.data.copy_(src.out_proj.bias.data)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_attention_matches_multiheadattention(bias, cross, masked):
    dim, heads = 32, 4
    ours = Attention(dim, heads, bias=bias)
    ref = nn.MultiheadAttention(dim, heads, bias=bias, batch_first=True)
    copy_weights(ours, ref)
    q = torch.randn(2, 11, dim)
    kv = torch.randn(2, 13, dim) if cross else q
    kv_mask = None
    if masked:
        kv_mask = torch.ones(2, kv.shape[1], dtype=torch.bool)
        kv_mask[1, -4:] = False
    out = ours(q, kv, kv, kv_mask=kv_mask) if cross else ours(q, kv_mask=kv_mask)
    expected, _ = ref(q, kv, kv, key_padding_mask=None if kv_mask is None else ~kv_mask)
    assert out.shape == (2, 11, dim)
    assert torch.allclose(out, expected, atol=1e-5)


def test_attention_heads_roundtrip_and_layout():
    a = Attention(24, num_heads=3)
    x = torch.randn(2, 5, 24)
    h = a.separate_heads(x)
    assert h.shape == (2, 3, 5, 8)
    assert torch.equal(a.recombine_heads(h), x)


def test_attention_qkv_norm_and_validation():
    a = Attention(16, num_heads=2, qkv_norm=True, norm="RMSNorm")
    assert isinstance(a.q_norm, RMSNorm)
    assert a(torch.randn(1, 4, 16)).shape == (1, 4, 16)
    with pytest.raises(ValueError):
        Attention(16, num_heads=3)
    with pytest.raises(ValueError):
        Attention(16, num_heads=2, qkv_norm=True)
    with pytest.raises(ValueError):
        Attention(16, attn_type="flex", window_size=8)


def test_attention_is_differentiable():
    a = Attention(16, num_heads=2)
    a(torch.randn(1, 4, 16)).sum().backward()
    assert a.in_proj_weight.grad is not None
    assert a.out_proj.weight.grad is not None


# ----------------------------------------------------------------------------- encoder pieces


def test_layerscale():
    ls = LayerScale(8, init_value=1e-3)
    x = torch.randn(2, 8)
    assert torch.allclose(ls(x), 1e-3 * x)
    assert ls.gamma.requires_grad


def test_droppath():
    dp = DropPath(0.5)
    x = torch.ones(1000, 3, 4)
    dp.train()
    out = dp(x)
    per_sample = out.flatten(1)
    assert torch.all((per_sample == 0).all(1) | (per_sample == 2).all(1)), "drop whole samples, rescale survivors"
    assert 0.4 < (per_sample[:, 0] == 0).float().mean() < 0.6
    dp.eval()
    assert torch.equal(dp(x), x)
    assert torch.equal(DropPath(0.0).train()(x), x)


def test_residual_pre_and_post_norm():
    fn = nn.Linear(8, 8)
    x = torch.randn(2, 3, 8)
    pre = Residual(fn, 8, norm="LayerNorm")
    assert torch.allclose(pre(x), x + fn(nn.functional.layer_norm(x, (8,))), atol=1e-6)
    post = Residual(fn, 8, norm="LayerNorm", post_norm=True)
    xn = nn.functional.layer_norm(x, (8,))
    assert torch.allclose(post(x), xn + fn(xn), atol=1e-6)
    none = Residual(fn, 8, norm=None)
    assert torch.allclose(none(x), x + fn(x))
    with pytest.raises(ValueError):
        Residual(fn, 8, norm=None, post_norm=True)


def test_residual_passes_kwargs():
    class Echo(nn.Module):
        def forward(self, x, scale=1.0):
            return x * scale

    r = Residual(Echo(), 4, norm=None)
    x = torch.ones(1, 4)
    assert torch.allclose(r(x, scale=3.0), 4 * x)


def test_encoder_layer_hybrid_norm():
    first = EncoderLayer(16, depth=0, hybrid_norm=True, attn_kwargs={"num_heads": 2})
    later = EncoderLayer(16, depth=1, hybrid_norm=True, attn_kwargs={"num_heads": 2})
    assert isinstance(first.attn.norm, nn.LayerNorm)
    assert isinstance(later.attn.norm, nn.Identity)
    assert later.dense.post_norm
    assert later.attn.fn.qkv_norm


def test_encoder_shapes_and_config():
    enc = Encoder(num_layers=3, dim=16, attn_kwargs={"num_heads": 2}, layer_scale=1e-2)
    assert len(enc.layers) == 3
    assert all(layer.attn.fn.num_heads == 2 for layer in enc.layers)
    assert enc(torch.randn(2, 10, 16)).shape == (2, 10, 16)


def test_encoder_is_permutation_equivariant_and_sort_is_invisible():
    enc = Encoder(num_layers=2, dim=16, attn_kwargs={"num_heads": 2}).eval()
    x = torch.randn(1, 12, 16)
    perm = torch.randperm(12)
    assert torch.allclose(enc(x)[:, perm], enc(x[:, perm]), atol=1e-5)
    # With full attention, sorting first then unsorting must give exactly the unsorted answer.
    phi = torch.rand(1, 12)
    assert torch.allclose(enc(x, x_sort_value=phi), enc(x), atol=1e-5)


def test_encoder_padding_does_not_leak():
    enc = Encoder(num_layers=2, dim=16, attn_kwargs={"num_heads": 2}).eval()
    x = torch.randn(1, 10, 16)
    kv_mask = torch.ones(1, 10, dtype=torch.bool)
    kv_mask[0, 7:] = False
    out = enc(x, kv_mask=kv_mask)
    x2 = x.clone()
    x2[0, 7:] = 100.0  # change only the padding
    out2 = enc(x2, kv_mask=kv_mask)
    assert torch.allclose(out[0, :7], out2[0, :7], atol=1e-5)


# ----------------------------------------------------------------------------- transformer hit filter


def make_transformer_filter(dim: int = 32) -> HitFilter:
    fields = TOY_INPUTS["hit"]
    posenc = PositionEncoder("hit", ["r", "eta", "phi"], dim, sym_fields=["phi"], alpha=10)
    input_net = InputNet("hit", Dense(len(fields), dim, hidden_layers=[dim]), fields, posenc=posenc)
    encoder = Encoder(num_layers=2, dim=dim, attn_kwargs={"num_heads": 4})
    task = HitFilterTask("hit_filter", "hit", "on_valid_particle", dim)
    return HitFilter(nn.ModuleList([input_net]), nn.ModuleList([task]), encoder=encoder, input_sort_field="phi")


def test_transformer_hit_filter_runs(toy_dir):
    ds = ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS)
    model = make_transformer_filter()
    inputs, targets = ds[0]
    outputs = model(inputs)
    _, _, losses = model.loss(outputs, targets)
    losses["final"]["hit_filter"]["hit_bce"].backward()
    assert model.encoder.layers[0].attn.fn.in_proj_weight.grad is not None
