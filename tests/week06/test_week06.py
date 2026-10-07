import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exercises.week03 import ex02_training_loop as training
from exercises.week06 import ex01_assignment as ex01
from exercises.week06.ex02_build_tracker import build_tracker
from minihep.data.dataset import ToyTrackingDataset
from minihep.metrics import double_majority, matched_efficiency
from minihep.models import loss as L
from minihep.models.decoder import MaskFormerDecoder, MaskFormerDecoderLayer
from minihep.models.matcher import Matcher, match_one, permute_outputs
from minihep.models.tasks import HitMaskTask, ObjectValidTask
from tests.conftest import TOY_INPUTS, TOY_TARGETS

# ----------------------------------------------------------------------------- ex01 assignment


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 6)])
def test_assignment_methods_agree_on_optimum(shape):
    rng = np.random.default_rng(sum(shape))
    cost = rng.random(shape)
    _, brute = ex01.brute_force_assignment(cost)
    cols, best = ex01.scipy_assignment(cost)
    assert best == pytest.approx(brute)
    assert len(set(cols.tolist())) == shape[0]
    _, greedy = ex01.greedy_assignment(cost)
    assert greedy >= best - 1e-12


def test_greedy_counterexample():
    cost = ex01.greedy_counterexample()
    assert cost.shape == (2, 2)
    assert ex01.greedy_assignment(cost)[1] > ex01.scipy_assignment(cost)[1]


# ----------------------------------------------------------------------------- costs


def _mask_problem(b=2, q=5, t=4, n=30):
    logits = torch.randn(b, q, n) * 3
    targets = (torch.rand(b, t, n) > 0.7).float()
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[1, -7:] = False
    targets = targets * valid.unsqueeze(1)  # padded hits never belong to a target (the dataset pads with False)
    return logits, targets, valid


def test_object_bce_cost():
    logits, targets = torch.randn(2, 5), torch.tensor([[1.0, 1, 0], [1, 0, 0]])
    cost = L.object_bce_cost(logits, targets)
    assert cost.shape == (2, 5, 3)
    p = logits.sigmoid()
    assert torch.allclose(cost[0, 2, 0], -p[0, 2])
    assert torch.allclose(cost[1, 4, 2], -(1 - p[1, 4]))


@pytest.mark.parametrize("name", ["mask_bce", "mask_focal", "mask_dice"])
def test_mask_costs_match_pairwise_loops(name):
    logits, targets, valid = _mask_problem()
    cost = L.COST_FNS[name](logits, targets, valid)
    assert cost.shape == (2, 5, 4)
    for b, qi, ti in [(0, 1, 2), (1, 4, 0), (1, 0, 3)]:
        lg, tg, v = logits[b, qi][valid[b]], targets[b, ti][valid[b]], None
        p = lg.sigmoid()
        if name == "mask_bce":
            expected = F.binary_cross_entropy_with_logits(lg, tg, reduction="sum")
        elif name == "mask_focal":
            p_t = p * tg + (1 - p) * (1 - tg)
            expected = (F.binary_cross_entropy_with_logits(lg, tg, reduction="none") * (1 - p_t) ** 2).sum()
        else:
            expected = 1 - (2 * (p * tg).sum() + 1) / (p.sum() + tg.sum() + 1)
        assert torch.allclose(cost[b, qi, ti], expected, atol=1e-4), (name, b, qi, ti, v)


def test_costs_recover_a_permutation():
    """A perfect but shuffled prediction must be matched back exactly (cf. hepattn tests/matching)."""
    true = (torch.rand(1, 6, 40) > 0.6).float()
    perm = torch.randperm(6)
    logits = (true[:, perm] * 2 - 1) * 10
    for name in ("mask_bce", "mask_focal", "mask_dice"):
        cost = L.COST_FNS[name](logits, true)
        idx = Matcher()(cost, torch.ones(1, 6, dtype=torch.bool))
        assert torch.equal(permute_outputs(logits, idx), (true * 2 - 1) * 10), name


@pytest.mark.parametrize("name", ["mask_bce", "mask_focal", "mask_dice"])
def test_costs_stay_float32_under_autocast(name):
    """Mixed precision must not reach the matching costs: autocast would run their einsums in bf16."""
    logits, targets, valid = _mask_problem(q=32, t=32, n=200)
    expected = L.COST_FNS[name](logits, targets, valid)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cost = L.COST_FNS[name](logits, targets, valid)
    assert cost.dtype == torch.float32
    assert torch.allclose(cost, expected)


# ----------------------------------------------------------------------------- losses


def test_object_bce_loss_weights():
    logits, targets = torch.randn(1, 6), torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    plain = F.binary_cross_entropy_with_logits(logits, targets.float())
    assert torch.allclose(L.object_bce_loss(logits, targets, null_weight=1.0), plain)
    w = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])
    assert torch.allclose(L.object_bce_loss(logits, targets, null_weight=0.5), F.binary_cross_entropy_with_logits(logits, targets.float(), weight=w))


def test_mask_losses_ignore_fake_objects_and_padding():
    logits, targets, valid = _mask_problem(t=5)
    obj_valid = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])
    for fn in (L.mask_dice_loss, L.mask_focal_loss):
        base = fn(logits, targets, obj_valid, valid)
        poisoned = logits.clone()
        poisoned[0, 3] = 50.0  # a fake object
        poisoned[1, 0, -7:] = 50.0  # padded hits of a real object
        assert torch.allclose(fn(poisoned, targets, obj_valid, valid), base), fn.__name__
    perfect = (targets * 2 - 1) * 20
    assert L.mask_dice_loss(perfect, targets, obj_valid, valid) < 0.05
    assert L.mask_focal_loss(perfect, targets, obj_valid, valid) < 1e-6


# ----------------------------------------------------------------------------- matcher


def test_match_one_convention():
    cost = np.array([[9.0, 1.0, 9.0], [1.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 1.0]])  # (Q=4, T=3)
    idx = match_one(cost, num_valid_targets=2)
    assert idx.tolist() == [1, 0, 2, 3]
    assert sorted(match_one(cost, 3).tolist()) == [0, 1, 2, 3]


def test_matcher_batched_and_rejects_nan():
    cost = torch.rand(3, 8, 5)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    idx = Matcher()(cost, valid)
    assert idx.shape == (3, 8)
    assert idx.dtype == torch.int64
    for b in range(3):
        assert sorted(idx[b].tolist()) == list(range(8))
    cost[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        Matcher()(cost, valid)


# ----------------------------------------------------------------------------- tasks


def test_object_valid_task():
    task = ObjectValidTask("track_valid", "query", "track", "particle", dim=8)
    assert task.outputs == ["track_logit"]
    out = task({"query_embed": torch.randn(2, 5, 8)})
    assert out["track_logit"].shape == (2, 5)
    targets = {"particle_valid": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)}
    assert task.cost(out, targets)["object_bce"].shape == (2, 5, 3)
    preds = task.predict(out)
    assert preds["track_valid"].dtype == torch.bool


def test_hit_mask_task():
    task = HitMaskTask(
        "track_hit_valid", "hit", "query", "track", "particle", dim=8, losses={"mask_dice": 1.0}, costs={"mask_dice": 1.0, "mask_focal": 2.0}
    )
    valid = torch.ones(1, 10, dtype=torch.bool)
    valid[0, 8:] = False
    x = {"query_embed": torch.randn(1, 4, 8), "hit_embed": torch.randn(1, 10, 8), "hit_valid": valid}
    out = task(x)
    logits = out["track_hit_logit"]
    assert logits.shape == (1, 4, 10)
    assert torch.all(logits[0, :, 8:] == torch.finfo(logits.dtype).min)
    expected = torch.einsum("bqd,bnd->bqn", task.object_net(x["query_embed"]), x["hit_embed"])
    assert torch.allclose(logits[0, :, :8], expected[0, :, :8])
    assert task.attn_mask(out)["hit"].shape == (1, 4, 10)
    targets = {"particle_hit_valid": torch.rand(1, 3, 10) > 0.5, "hit_valid": valid, "particle_valid": torch.tensor([[True, True, False]])}
    costs = task.cost(out, targets)
    assert set(costs) == {"mask_dice", "mask_focal"}
    assert torch.isfinite(costs["mask_focal"]).all()
    assert set(task.loss({"track_hit_logit": logits[:, :3]}, targets)) == {"mask_dice"}


# ----------------------------------------------------------------------------- decoder


def test_decoder_layer_respects_mask():
    layer = MaskFormerDecoderLayer(16, attn_kwargs={"num_heads": 2}).eval()
    q, kv = torch.randn(1, 3, 16), torch.randn(1, 10, 16)
    mask = torch.zeros(1, 3, 10, dtype=torch.bool)
    mask[0, :, :4] = True
    out, kv_out = layer(q, kv, attn_mask=mask)
    kv2 = kv.clone()
    kv2[0, 4:] += 5.0  # change only hits the queries cannot see
    out2, _ = layer(q, kv2, attn_mask=mask)
    assert torch.allclose(out, out2, atol=1e-5)
    assert torch.equal(kv_out, kv)  # not bidirectional: hits unchanged


def test_decoder_layer_bidirectional_updates_hits():
    layer = MaskFormerDecoderLayer(16, attn_kwargs={"num_heads": 2}, bidirectional_ca=True)
    q, kv = torch.randn(1, 3, 16), torch.randn(1, 10, 16)
    mask = torch.zeros(1, 3, 10, dtype=torch.bool)
    mask[0, 0, :2] = True
    _, kv_out = layer(q, kv, attn_mask=mask)
    assert kv_out.shape == kv.shape
    assert torch.isfinite(kv_out).all()
    assert not torch.allclose(kv_out, kv)


def test_decoder_builds_mask_from_tasks():
    task = HitMaskTask("track_hit_valid", "hit", "query", "track", "particle", dim=16, losses={}, costs={})
    dec = MaskFormerDecoder(num_queries=4, dim=16, num_layers=2, layer_kwargs={"attn_kwargs": {"num_heads": 2}})
    dec.tasks = torch.nn.ModuleList([task])
    x = {
        "key_embed": torch.randn(1, 12, 16),
        "key_is_hit": torch.ones(12, dtype=torch.bool),
        "hit_embed": None,
        "hit_valid": torch.ones(1, 12, dtype=torch.bool),
    }
    x["hit_embed"] = x["key_embed"]
    x, outputs = dec(x, ["hit"])
    assert set(outputs) == {"layer_0", "layer_1"}
    assert outputs["layer_0"]["track_hit_valid"]["track_hit_logit"].shape == (1, 4, 12)
    assert x["query_embed"].shape == (1, 4, 16)
    m = dec.build_attn_mask(x, outputs["layer_1"], ["hit"])
    assert m.shape == (1, 4, 12)
    assert m.any(-1).all(), "unmask_all_false: no query may be left with nothing to look at"
    assert not m.requires_grad


# ----------------------------------------------------------------------------- metrics


def test_matched_efficiency():
    true_valid = torch.tensor([[True, True, False]])
    pred_valid = torch.tensor([[True, True, True]])
    true = torch.tensor([[[1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
    pred = torch.tensor([[[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0]]], dtype=torch.bool)
    m = matched_efficiency(pred_valid, pred, true_valid, true, 0.5)
    assert m["eff"].item() == pytest.approx(1.0)  # 3/4 and 1/2 are both >= 0.5
    assert m["pur"].item() == pytest.approx(2 / 3)  # tracks 0 (3/3) and 1 (1/2) are pure; slot 2 has no particle
    m = matched_efficiency(pred_valid, pred, true_valid, true, 0.75)
    assert m["eff"].item() == pytest.approx(0.5)


def test_double_majority():
    true_valid = torch.tensor([True, True, True])
    true = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.bool)
    pred_valid = torch.tensor([True, True, True, False])
    pred = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0, 0],  # matched to particle 0 (3/3 pure, 3/4 of particle)
            [1, 1, 1, 0, 0, 0, 0, 0],  # duplicate of particle 0
            [0, 0, 0, 0, 1, 0, 0, 1],  # 1 hit each from particles 1 and 2: no majority -> fake
            [0, 0, 0, 0, 1, 1, 1, 0],  # would match particle 1, but the track is not valid
        ],
        dtype=torch.bool,
    )
    m = double_majority(pred_valid, pred, true_valid, true)
    assert m["eff"] == pytest.approx(1 / 3)
    assert m["fake_rate"] == pytest.approx(1 / 3)
    assert m["duplicate_rate"] == pytest.approx(1 / 3)


# ----------------------------------------------------------------------------- the whole model


@pytest.fixture
def two_events(toy_dir):
    return ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS, num_events=2, event_max_num_particles=32)


def test_tracker_forward_and_loss(two_events):
    model = build_tracker(TOY_INPUTS["hit"], num_queries=32)
    inputs, targets = two_events[0]
    outputs = model(inputs)
    assert set(outputs) == {"layer_0", "layer_1", "final"}
    assert outputs["final"]["track_hit_valid"]["track_hit_logit"].shape == (1, 32, inputs["hit_valid"].shape[-1])
    outputs, targets, losses = model.loss(outputs, targets)
    total = sum(v for layer in losses.values() for task in layer.values() for v in task.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.decoder.initial_queries.grad is not None


@pytest.mark.slow
def test_tracker_overfits_two_events(two_events):
    torch.manual_seed(0)
    model = build_tracker(TOY_INPUTS["hit"], num_queries=32)
    history = training.train(model, two_events, num_steps=400, cfg=training.LRConfig(initial=1e-3, max=3e-3, end=1e-4, pct_start=0.05))
    assert history.loss[-1] < 0.2 * history.loss[0]
    model.eval()
    with torch.no_grad():
        for i in range(2):
            inputs, targets = two_events[i]
            outputs, targets, _ = model.loss(model(inputs), targets)
            preds = model.predict(outputs)["final"]
            dm = double_majority(
                preds["track_valid"]["track_valid"][0],
                preds["track_hit_valid"]["track_hit_valid"][0],
                targets["particle_valid"][0],
                targets["particle_hit_valid"][0],
            )
            assert dm["eff"] > 0.9, dm
            assert dm["fake_rate"] < 0.1, dm
