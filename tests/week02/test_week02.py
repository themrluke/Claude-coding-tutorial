import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from exercises.week02 import ex01_numpy as ex01
from exercises.week02 import ex02_tensor_shapes as ex02
from minihep.data import prep
from minihep.data.dataset import PARTICLE_PAD_ID, ToyTrackingDataset, pad_collate
from minihep.data.toy_detector import simulate_event
from tests.conftest import TOY_INPUTS, TOY_TARGETS

# ----------------------------------------------------------------------------- ex01 numpy


def test_wrap_phi():
    phi = np.array([0.0, np.pi, -np.pi, 3 * np.pi, 7.0, -7.0])
    out = ex01.wrap_phi(phi)
    assert np.all(out >= -np.pi)
    assert np.all(out < np.pi)
    np.testing.assert_allclose(np.cos(out), np.cos(phi), atol=1e-12)
    np.testing.assert_allclose(np.sin(out), np.sin(phi), atol=1e-12)


def test_delta_phi():
    np.testing.assert_allclose(ex01.delta_phi(np.array(3.1), np.array(-3.1)), 6.2 - 2 * np.pi, atol=1e-12)
    np.testing.assert_allclose(ex01.delta_phi(np.array(0.5), np.array(0.2)), 0.3, atol=1e-12)


def test_eta_from_xyz():
    assert ex01.eta_from_xyz(np.array(1.0), np.array(0.0), np.array(0.0)) == pytest.approx(0.0, abs=1e-12)
    # eta = asinh(z / r)
    x, y, z = np.array([1.0, 0.3]), np.array([2.0, -0.4]), np.array([5.0, -1.0])
    np.testing.assert_allclose(ex01.eta_from_xyz(x, y, z), np.arcsinh(z / np.hypot(x, y)), rtol=1e-10)


def test_pairwise_distances():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
    d = ex01.pairwise_distances(a, b)
    assert d.shape == (5, 7)
    assert d[2, 4] == pytest.approx(np.linalg.norm(a[2] - b[4]))


def test_inverse_permutation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20)
    perm = rng.permutation(20)
    inv = ex01.inverse_permutation(perm)
    np.testing.assert_array_equal(x[perm][inv], x)


def test_ranks():
    np.testing.assert_array_equal(ex01.ranks(np.array([30, 10, 20])), [2, 0, 1])
    np.testing.assert_array_equal(ex01.ranks(np.array([5, 5, 1])), [1, 2, 0])  # stable on ties
    rng = np.random.default_rng(2)
    v = rng.normal(size=50)
    np.testing.assert_array_equal(np.sort(v)[ex01.ranks(v)], v)


def test_innermost_hit_mask_matches_pandas():
    hits, _ = simulate_event(3)
    r = np.hypot(hits["x"], hits["y"]).to_numpy()
    pid = hits["particle_id"].to_numpy()
    mask = ex01.innermost_hit_mask(pid, r)
    tmp = pd.DataFrame({"pid": pid, "r": r})
    expected = np.zeros(len(tmp), dtype=bool)
    expected[tmp[tmp.pid != 0].groupby("pid")["r"].idxmin().to_numpy()] = True
    np.testing.assert_array_equal(mask, expected)


def test_hits_per_particle():
    pid = np.array([5, 0, 5, 7, 5, 7])
    np.testing.assert_array_equal(ex01.hits_per_particle(pid, np.array([5, 7, 9])), [3, 2, 0])


# ----------------------------------------------------------------------------- ex02 tensors


def test_build_target_masks():
    object_ids = torch.tensor([[11, 22, -999]])
    hit_ids = torch.tensor([[22, 0, 11, 22]])
    mask = ex02.build_target_masks(object_ids, hit_ids)
    assert mask.shape == (1, 3, 4)
    assert mask.dtype == torch.bool
    expected = torch.tensor([[[False, False, True, False], [True, False, False, True], [False, False, False, False]]])
    assert torch.equal(mask, expected)


def test_sort_and_unsort():
    x = torch.randn(2, 10, 4)
    phi = torch.rand(2, 10)
    xs, idx = ex02.sort_tokens(x, phi)
    sorted_phi = torch.gather(phi, -1, idx)
    assert torch.all(sorted_phi[:, 1:] >= sorted_phi[:, :-1])
    assert torch.equal(xs[1, 0], x[1, phi[1].argmin()])
    assert torch.equal(ex02.unsort_tokens(xs, idx), x)


def test_mask_logits():
    q, k = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    logits = ex02.mask_logits(q, k)
    assert logits.shape == (2, 3, 5)
    assert torch.allclose(logits[1, 2, 4], q[1, 2] @ k[1, 4])
    valid = torch.tensor([[True] * 5, [True, True, True, False, False]])
    masked = ex02.mask_logits(q, k, valid)
    assert torch.all(masked[1, :, 3:] == torch.finfo(torch.float32).min)
    assert torch.equal(masked[0], logits[0])
    assert torch.sigmoid(masked[1, :, 3:]).max() == 0


def test_heads_roundtrip():
    x = torch.randn(2, 7, 12)
    h = ex02.separate_heads(x, 3)
    assert h.shape == (2, 3, 7, 4)
    assert torch.equal(h[1, 2, 5], x[1, 5, 8:12])
    assert torch.equal(ex02.recombine_heads(h), x)


def test_permute_objects():
    x = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    idx = torch.tensor([[3, 2, 1, 0], [0, 2, 1, 3]])
    out = ex02.permute_objects(x, idx)
    assert torch.equal(out[0], x[0].flip(0))
    assert torch.equal(out[1, 1], x[1, 2])


def test_expand_vs_repeat():
    x = torch.randn(1, 5, 4)
    expanded, repeated = ex02.expand_is_a_view(x, 3)
    assert expanded.shape == repeated.shape == (3, 5, 4)
    assert expanded.data_ptr() == x.data_ptr()
    assert expanded.stride(0) == 0
    assert repeated.data_ptr() != x.data_ptr()


# ----------------------------------------------------------------------------- prep


def test_write_event_parquet(tmp_path):
    hits, parts = simulate_event(7)
    hits_path, parts_path = prep.write_event_parquet(hits, parts, tmp_path / "new" / "dir", 7)
    assert hits_path.name == "event000000007-hits.parquet"
    assert parts_path.name == "event000000007-parts.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(hits_path), hits)
    pd.testing.assert_frame_equal(pd.read_parquet(parts_path), parts)


def test_generate_split_resumes(tmp_path):
    assert prep.generate_split(tmp_path, 3, first_event_id=10) == 3
    assert prep.generate_split(tmp_path, 4, first_event_id=10) == 1
    assert len(list(tmp_path.glob("*-hits.parquet"))) == 4


def test_hdf5_roundtrip(toy_dir, tmp_path):
    out = tmp_path / "toy.h5"
    n = prep.parquet_dir_to_hdf5(toy_dir, out)
    assert n == 20
    with h5py.File(out, "r") as f:
        names = sorted(f.keys())
        assert len(names) == 20
        group = f[names[0]]
        assert group["hits/x"].compression == "lzf"
        hits, parts = prep.read_event_hdf5(f, names[0])
        assert group.attrs["num_hits"] == len(hits)
        assert group.attrs["num_particles"] == len(parts)
    ref_hits = pd.read_parquet(toy_dir / f"{names[0]}-hits.parquet")
    ref_parts = pd.read_parquet(toy_dir / f"{names[0]}-parts.parquet")
    pd.testing.assert_frame_equal(hits[ref_hits.columns], ref_hits)
    pd.testing.assert_frame_equal(parts[ref_parts.columns], ref_parts)


# ----------------------------------------------------------------------------- dataset


@pytest.fixture
def dataset(toy_dir):
    return ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS, particle_min_pt=0.5, event_max_num_particles=60)


def test_load_event_columns_and_cuts(dataset):
    hits, particles = dataset.load_event(0)
    for col in ("r", "s", "theta", "phi", "eta", "on_valid_particle", "is_first"):
        assert col in hits.columns
    for col in ("pt", "p", "eta", "phi", "qopt"):
        assert col in particles.columns
    assert hits["r"].max() < 1.0, "coordinates should be in metres"
    assert (particles["pt"] > 0.5).all()
    assert (particles["eta"].abs() < 2.5).all()
    counts = hits["particle_id"].value_counts()
    assert (counts[particles["particle_id"]] >= 3).all()
    # exactly one first hit per valid particle, and only on valid particles
    assert hits["is_first"].sum() == len(particles)
    assert not (hits["is_first"] & ~hits["on_valid_particle"]).any()
    assert not hits.loc[hits["particle_id"] == 0, "on_valid_particle"].any()


def test_volume_selection(toy_dir):
    ds = ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS, hit_volume_ids=[8], particle_min_num_hits=2)
    hits, _ = ds.load_event(0)
    assert set(hits["volume_id"].unique()) == {8}
    assert hits["r"].max() < 0.2


def test_getitem_shapes_and_dtypes(dataset):
    inputs, targets = dataset[0]
    n = inputs["hit_valid"].shape[-1]
    assert inputs["hit_valid"].shape == (1, n)
    assert inputs["hit_valid"].dtype == torch.bool
    for field in TOY_INPUTS["hit"]:
        assert inputs[f"hit_{field}"].shape == (1, n)
        assert inputs[f"hit_{field}"].dtype == torch.float32
    assert targets["particle_valid"].shape == (1, 60)
    assert targets["particle_hit_valid"].shape == (1, 60, n)
    assert targets["hit_on_valid_particle"].dtype == torch.bool
    assert targets["particle_pt"].shape == (1, 60)
    assert targets["sample_id"].shape == (1,)


def test_getitem_consistency(dataset):
    _, targets = dataset[1]
    valid = targets["particle_valid"][0]
    masks = targets["particle_hit_valid"][0]
    # padded particles own no hits; real particles own >= 3
    assert not masks[~valid].any()
    assert (masks[valid].sum(-1) >= 3).all()
    # a hit is on a valid particle iff some particle claims it; no hit is claimed twice
    assert torch.equal(masks.any(0), targets["hit_on_valid_particle"][0])
    assert masks.sum(0).max() <= 1
    # NaN tripwire in padded slots only
    assert torch.isnan(targets["particle_pt"][0, ~valid]).all()
    assert not torch.isnan(targets["particle_pt"][0, valid]).any()
    assert targets["hit_is_first"][0].sum() == valid.sum()


def test_truncation(toy_dir):
    ds = ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS, particle_min_pt=0.0, particle_min_num_hits=1, event_max_num_particles=3)
    _, targets = ds[0]
    assert targets["particle_valid"].shape == (1, 3)
    assert targets["particle_valid"].all()


def test_hdf5_source_matches_parquet(toy_dir, tmp_path, dataset):
    h5_path = tmp_path / "toy.h5"
    prep.parquet_dir_to_hdf5(toy_dir, h5_path)
    ds_h5 = ToyTrackingDataset(str(h5_path), TOY_INPUTS, TOY_TARGETS, particle_min_pt=0.5, event_max_num_particles=60)
    assert len(ds_h5) == len(dataset)
    for (a_in, a_t), (b_in, b_t) in [(dataset[i], ds_h5[i]) for i in (0, 5)]:
        for key in a_in:
            assert torch.equal(a_in[key], b_in[key]), key
        for key in a_t:
            assert torch.allclose(a_t[key].float(), b_t[key].float(), equal_nan=True), key


def test_dataloader_batch_size_none(dataset):
    loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=0)
    inputs, targets = next(iter(loader))
    assert inputs["hit_x"].shape[0] == 1
    assert targets["particle_valid"].shape == (1, 60)


def test_pad_collate(dataset):
    batch = [dataset[i] for i in range(3)]
    lengths = [b[0]["hit_valid"].shape[-1] for b in batch]
    inputs, targets = pad_collate(batch)
    n = max(lengths)
    assert inputs["hit_x"].shape == (3, n)
    assert inputs["hit_valid"].sum(-1).tolist() == lengths
    assert targets["particle_hit_valid"].shape == (3, 60, n)
    assert targets["particle_valid"].shape == (3, 60)
    for i, length in enumerate(lengths):
        assert torch.equal(inputs["hit_x"][i, :length], batch[i][0]["hit_x"][0])
        assert (inputs["hit_x"][i, length:] == 0).all()
        assert not targets["particle_hit_valid"][i, :, length:].any()


def test_pad_id_never_matches_noise():
    assert PARTICLE_PAD_ID not in (0, -1)
