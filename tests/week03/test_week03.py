import math

import pytest
import torch
from torch import nn

from exercises.week03 import ex01_autograd_and_modules as ex01
from exercises.week03 import ex02_training_loop as ex02
from minihep.data.dataset import ToyTrackingDataset
from minihep.models.dense import Dense, SwiGLU
from minihep.models.hitfilter import HitFilter
from minihep.models.input import InputNet
from minihep.models.loss import focal_loss, hit_bce_loss
from minihep.models.posenc import PositionEncoder, pos_enc, pos_enc_symmetric
from minihep.models.tasks import HitFilterTask
from tests.conftest import TOY_INPUTS, TOY_TARGETS

# ----------------------------------------------------------------------------- ex01


def test_gradient_of_polynomial():
    x = torch.tensor([0.0, 1.0, -2.0, 0.5])
    g = ex01.gradient_of_polynomial(x)
    assert torch.allclose(g, 3 * x**2 - 2)
    assert x.grad is None
    assert not x.requires_grad


def test_standardiser_parameters_vs_buffers():
    m = ex01.Standardiser(3)
    assert {n for n, _ in m.named_parameters()} == {"weight", "bias"}
    assert {n for n, _ in m.named_buffers()} == {"mean", "std"}
    assert set(m.state_dict()) == {"weight", "bias", "mean", "std"}


def test_standardiser_fit_and_forward():
    m = ex01.Standardiser(2)
    x = torch.randn(1000, 2) * torch.tensor([3.0, 0.5]) + torch.tensor([10.0, -1.0])
    m.fit(x)
    y = m(x)
    assert torch.allclose(y.mean(0), torch.zeros(2), atol=1e-4)
    assert torch.allclose(y.std(0), torch.ones(2), atol=1e-3)
    assert not m.mean.requires_grad
    y.sum().backward()
    assert m.weight.grad is not None
    assert m.mean.grad is None


def test_count_parameters_and_freeze():
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
    counts = ex01.count_parameters(model)
    assert counts == {"trainable": 4 * 8 + 8 + 8 + 8 + 8 + 1, "frozen": 0, "buffers": 8 + 8 + 1}
    ex01.freeze(model[0])
    counts = ex01.count_parameters(model)
    assert counts["frozen"] == 4 * 8 + 8
    assert counts["trainable"] == 8 + 8 + 8 + 1


def test_sgd_step():
    model = nn.Linear(3, 1)
    before = model.weight.detach().clone()
    x = torch.randn(5, 3)
    model(x).sum().backward()
    grad = model.weight.grad.clone()
    ex01.sgd_step(model, lr=0.1)
    assert torch.allclose(model.weight, before - 0.1 * grad)
    assert model.weight.grad is None


def test_dropout_modes():
    x = torch.ones(10_000)
    train_out, eval_out = ex01.dropout_outputs(0.5, x)
    assert torch.equal(eval_out, x)
    zeros = (train_out == 0).float().mean()
    assert 0.45 < zeros < 0.55
    assert torch.allclose(train_out[train_out != 0], torch.full_like(train_out[train_out != 0], 2.0))


# ----------------------------------------------------------------------------- building blocks


def test_swiglu():
    x = torch.randn(2, 5, 8)
    out = SwiGLU()(x)
    assert out.shape == (2, 5, 4)
    assert torch.allclose(out, x[..., :4] * torch.nn.functional.silu(x[..., 4:]))


def test_dense_structure():
    d = Dense(6, 3, hidden_layers=[16, 8])
    linears = [m for m in d.net if isinstance(m, nn.Linear)]
    assert [(lin.in_features, lin.out_features) for lin in linears] == [(6, 16), (16, 8), (8, 3)]
    assert d.output_size == 3
    assert d(torch.randn(2, 7, 6)).shape == (2, 7, 3)


def test_dense_defaults_and_swiglu_width():
    d = Dense(4)
    assert d.output_size == 4
    assert [m.out_features for m in d.net if isinstance(m, nn.Linear)] == [8, 4]
    g = Dense(4, 2, hidden_layers=[16], activation="SwiGLU")
    linears = [m for m in g.net if isinstance(m, nn.Linear)]
    assert [(lin.in_features, lin.out_features) for lin in linears] == [(4, 32), (16, 2)]
    assert g(torch.randn(3, 4)).shape == (3, 2)


def test_dense_dropout_and_final_activation():
    d = Dense(4, 1, hidden_layers=[8, 8], dropout=0.1, final_activation=nn.Sigmoid())
    assert sum(isinstance(m, nn.Dropout) for m in d.net) == 2
    assert isinstance(d.net[-1], nn.Sigmoid)


def test_pos_enc_shapes_and_values():
    x = torch.randn(2, 5)
    pe = pos_enc(x, 16)
    assert pe.shape == (2, 5, 16)
    assert torch.allclose(pe[..., 0], torch.sin(x * 1000))
    assert pos_enc(x, 15).shape == (2, 5, 15)


def test_pos_enc_symmetric_is_periodic():
    phi = torch.linspace(-3, 3, 50, dtype=torch.float64)
    a, b = pos_enc_symmetric(phi, 32), pos_enc_symmetric(phi + 2 * math.pi, 32)
    assert torch.allclose(a, b, atol=1e-6)
    assert not torch.allclose(pos_enc(phi, 32), pos_enc(phi + 2 * math.pi, 32), atol=1e-3)


def test_position_encoder():
    pe = PositionEncoder("hit", ["r", "eta", "phi"], dim=32, sym_fields=["phi"])
    inputs = {"hit_r": torch.rand(1, 9), "hit_eta": torch.randn(1, 9), "hit_phi": torch.rand(1, 9) * 6 - 3}
    out = pe(inputs)
    assert out.shape == (1, 9, 32)
    assert torch.allclose(out[..., :10], pos_enc(inputs["hit_r"], 10))
    assert torch.allclose(out[..., 20:30], pos_enc_symmetric(inputs["hit_phi"], 10))
    assert torch.all(out[..., 30:] == 0)


def test_input_net():
    fields = ["x", "y"]
    net = InputNet("hit", nn.Linear(2, 8), fields)
    inputs = {"hit_x": torch.randn(1, 5), "hit_y": torch.randn(1, 5), "hit_z": torch.randn(1, 5)}
    out = net(inputs)
    expected = net.net(torch.stack([inputs["hit_x"], inputs["hit_y"]], dim=-1))
    assert torch.allclose(out, expected)
    with_pe = InputNet("hit", nn.Linear(2, 8), fields, posenc=PositionEncoder("hit", ["z"], 8))
    assert torch.allclose(with_pe(inputs), with_pe.net(torch.stack([inputs["hit_x"], inputs["hit_y"]], -1)) + pos_enc(inputs["hit_z"], 8))


# ----------------------------------------------------------------------------- losses


def test_hit_bce_loss_matches_torch():
    logits = torch.randn(1, 50)
    targets = torch.rand(1, 50) > 0.8
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=1 / targets.float().mean())
    assert torch.allclose(hit_bce_loss(logits, targets), expected)
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float())
    assert torch.allclose(hit_bce_loss(logits, targets, balance=False), plain)


def test_hit_bce_loss_ignores_padding():
    logits = torch.randn(2, 10)
    targets = torch.rand(2, 10) > 0.5
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1, 6:] = False
    poisoned = logits.clone()
    poisoned[1, 6:] = 1e6
    assert torch.allclose(hit_bce_loss(poisoned, targets, valid), hit_bce_loss(logits, targets, valid))
    assert torch.isfinite(hit_bce_loss(logits, torch.zeros(2, 10, dtype=torch.bool)))


def test_focal_loss():
    logits = torch.randn(4, 30)
    targets = torch.rand(4, 30) > 0.5
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float())
    assert torch.allclose(focal_loss(logits, targets, gamma=0.0), bce)
    assert focal_loss(logits, targets, gamma=2.0) < bce
    easy = torch.full((10,), 8.0)
    assert focal_loss(easy, torch.ones(10, dtype=torch.bool)) < 1e-6


# ----------------------------------------------------------------------------- model + training


def make_mlp_filter(dim: int = 32) -> HitFilter:
    fields = TOY_INPUTS["hit"]
    input_net = InputNet("hit", Dense(len(fields), dim, hidden_layers=[dim]), fields)
    task = HitFilterTask("hit_filter", "hit", "on_valid_particle", dim, threshold=0.5)
    return HitFilter(nn.ModuleList([input_net]), nn.ModuleList([task]))


def test_hit_filter_task_contract():
    task = HitFilterTask("hit_filter", "hit", "on_valid_particle", dim=8)
    x = {"hit_embed": torch.randn(1, 20, 8)}
    out = task(x)
    assert out["hit_logit"].shape == (1, 20)
    preds = task.predict(out)
    assert preds["hit_on_valid_particle"].dtype == torch.bool
    assert torch.allclose(preds["hit_on_valid_particle_prob"], out["hit_logit"].sigmoid())
    targets = {"hit_on_valid_particle": torch.rand(1, 20) > 0.5, "hit_valid": torch.ones(1, 20, dtype=torch.bool)}
    losses = task.loss(out, targets)
    assert set(losses) == {"hit_bce"}
    losses["hit_bce"].backward()
    assert task.net.net[0].weight.grad is not None


def test_hit_filter_task_metrics():
    task = HitFilterTask("hit_filter", "hit", "on_valid_particle", dim=8)
    preds = {"hit_on_valid_particle": torch.tensor([[True, True, False, False, True]])}
    targets = {
        "hit_on_valid_particle": torch.tensor([[True, False, True, False, True]]),
        "hit_valid": torch.tensor([[True, True, True, True, False]]),
    }
    m = task.metrics(preds, targets)
    assert m["recall"].item() == pytest.approx(0.5)
    assert m["precision"].item() == pytest.approx(0.5)


def test_hit_filter_forward_shapes(toy_dir):
    ds = ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS)
    model = make_mlp_filter()
    inputs, targets = ds[0]
    outputs = model(inputs)
    n = inputs["hit_valid"].shape[-1]
    assert outputs["final"]["hit_filter"]["hit_logit"].shape == (1, n)
    _, _, losses = model.loss(outputs, targets)
    assert torch.isfinite(losses["final"]["hit_filter"]["hit_bce"])


def test_optimizer_and_scheduler_follow_lrs_config():
    model = nn.Linear(2, 1)
    cfg = ex02.LRConfig(initial=1e-4, max=1e-3, end=1e-6, pct_start=0.25)
    opt, sch = ex02.make_optimizer_and_scheduler(model, total_steps=100, cfg=cfg)
    assert isinstance(opt, torch.optim.AdamW)
    lrs = []
    for _ in range(100):
        lrs.append(sch.get_last_lr()[0])
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx(1e-4)
    assert max(lrs) == pytest.approx(1e-3, rel=1e-3)
    assert lrs.index(max(lrs)) in range(23, 27)
    assert lrs[-1] < 2e-6


@pytest.mark.slow
def test_training_reduces_loss(toy_dir):
    ds = ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS)
    model = make_mlp_filter()
    history = ex02.train(model, ds, num_steps=200, cfg=ex02.LRConfig(initial=1e-3, max=5e-3, end=1e-4))
    assert len(history.loss) == len(history.lr) == len(history.grad_norm) == 200
    first, last = sum(history.loss[:20]) / 20, sum(history.loss[-20:]) / 20
    assert last < 0.9 * first, f"loss did not go down enough: {first:.3f} -> {last:.3f}"
    metrics = ex02.evaluate(model, ds, "hit_filter")
    assert set(metrics) == {"recall", "precision"}
    assert metrics["recall"] > 0.8
    assert model.training, "evaluate must restore train mode"
