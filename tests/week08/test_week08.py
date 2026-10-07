import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from exercises.week04.ex01_attention_from_scratch import naive_attention
from exercises.week08 import ex01_precision as ex01
from exercises.week08 import ex02_debugging_and_speed as ex02

# ----------------------------------------------------------------------------- precision


def test_overflows():
    values = torch.tensor([1.0, 65504.0, 70000.0, 121_000.0, 717_000.0, 1e30], dtype=torch.float64)
    assert ex01.overflows(values, torch.float16).tolist() == [False, False, True, True, True, True]
    assert not ex01.overflows(values, torch.bfloat16).any()


def test_rounding_error_bf16_is_coarser():
    x = torch.linspace(1, 1000, 1000, dtype=torch.float64) * torch.pi
    fp16 = ex01.relative_rounding_error(x, torch.float16).max()
    bf16 = ex01.relative_rounding_error(x, torch.bfloat16).max()
    assert fp16 <= 2**-11 + 1e-12
    assert bf16 <= 2**-8 + 1e-12
    assert bf16 > fp16


def test_swamping():
    assert ex01.accumulate(10_000, 1e-4, torch.float32) == pytest.approx(1.0, rel=1e-3)
    assert ex01.accumulate(10_000, 1e-4, torch.float16) < 0.5


def test_autocast_dtypes_cpu():
    assert ex01.autocast_dtypes("cpu") == {"linear": torch.bfloat16, "bce": torch.float32, "cost": torch.float32}


def test_mini_gradscaler_rules():
    s = ex01.MiniGradScaler(scale=8.0, growth_interval=3)
    assert s.update(False)
    assert s.update(False)
    assert s.update(False)
    assert s.scale == 16.0
    assert not s.update(True)
    assert s.scale == 8.0
    assert s.clean_steps == 0
    assert s.history == [8.0, 8.0, 16.0, 8.0]


def test_mini_gradscaler_matches_torch():
    """Feed the same inf pattern to torch's real GradScaler and compare the scale after every step."""
    pattern = [False, False, True, False, False, False, False, True, True, False, False, False]
    ours = ex01.MiniGradScaler(scale=1024.0, growth_interval=3)
    w = nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD([w], lr=0.1)
    real = torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=3)
    for bad in pattern:
        opt.zero_grad()
        loss = (w * (float("inf") if bad else 1.0)).sum()
        real.scale(loss).backward()
        real.step(opt)
        real.update()
        ours.update(bad)
        assert real.get_scale() == ours.scale


def test_corrupt_events_collapse_the_scale():
    healthy = ex01.simulate_corrupt_events(num_steps=20_000, bad_every=5_000, growth_interval=2000)
    assert healthy.history[-1] >= 4096.0
    dying = ex01.simulate_corrupt_events(num_steps=20_000, bad_every=1_400, growth_interval=2000)
    assert dying.history[-1] < 1.0
    assert all(b <= a for a, b in zip(dying.history, dying.history[1:]))


# ----------------------------------------------------------------------------- debugging


def test_check_inputs_finite():
    batch = {
        "hit_x": torch.randn(1, 10),
        "hit_charge_frac": torch.tensor([[0.3, 717_000.0]]),
        "hit_valid": torch.ones(1, 10, dtype=torch.bool),
        "hit_y": torch.tensor([[1.0, float("nan")]]),
        "sample_id": torch.tensor([5]),
    }
    assert ex02.check_inputs_finite(batch) == ["hit_y"]
    assert ex02.check_inputs_finite(batch, max_abs=65504) == ["hit_charge_frac", "hit_y"]


class Exploder(nn.Module):
    def forward(self, x):
        return x * float("inf")


def test_first_nonfinite_module():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Sequential(nn.Linear(4, 4), Exploder(), nn.Linear(4, 4)))
    assert ex02.first_nonfinite_module(model, torch.randn(2, 4)) == "2.1"
    fine = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    assert ex02.first_nonfinite_module(fine, torch.randn(2, 4)) is None
    assert not any(m._forward_hooks for m in model.modules()), "hooks must be removed"


def test_time_fn():
    ms = ex02.time_fn(lambda: sum(range(10_000)), warmup=1, iters=5)
    assert 0 < ms < 1000


def test_graph_breaks():
    def clean(x):
        return torch.relu(x) * 2

    def breaky(x):
        if x.sum().item() > 0:  # .item() pulls a value back to Python: graph break
            return x * 2
        return x

    x = torch.randn(8)
    assert ex02.count_graph_breaks(clean, x) == 0
    assert ex02.count_graph_breaks(breaky, x) >= 1


@pytest.mark.gpu
def test_naive_attention_uses_more_memory_than_sdpa():
    q = torch.randn(1, 4, 4096, 32, device="cuda", dtype=torch.float16)
    naive = ex02.peak_memory_mb(lambda: naive_attention(q, q, q))
    fused = ex02.peak_memory_mb(lambda: F.scaled_dot_product_attention(q, q, q))
    assert naive > 4 * fused  # the naive version materialises a 4 x 4096 x 4096 score matrix


# ----------------------------------------------------------------------------- compile callback


@pytest.mark.slow
def test_compile_callback(toy_dir):
    from lightning import Trainer

    from minihep.data.datamodule import ToyDataModule
    from minihep.lightning.callbacks import Compile
    from minihep.lightning.wrapper import Filter
    from tests.conftest import TOY_INPUTS
    from tests.week04.test_week04 import make_transformer_filter

    dm = ToyDataModule(str(toy_dir), str(toy_dir), num_train=3, num_val=1, inputs=TOY_INPUTS, targets={"hit": ["on_valid_particle"]})
    wrapper = Filter("compile-test", make_transformer_filter(), {"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1, "weight_decay": 0.0})
    callback = Compile()
    trainer = Trainer(
        max_steps=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
        num_sanity_val_steps=0,
    )
    trainer.fit(wrapper, dm)
    assert callback.compiled
    assert wrapper.model.encoder._compiled_call_impl is not None


# ----------------------------------------------------------------------------- git kata


def test_git_kata_bisect(tmp_path):
    kata = "exercises/week08/git_kata.py"
    repo = tmp_path / "bisect"
    subprocess.run([sys.executable, kata, "bisect", str(repo)], check=True, capture_output=True)
    first = subprocess.run(["git", "-C", str(repo), "rev-list", "--max-parents=0", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "bisect", "start", "HEAD", first], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "bisect", "run", sys.executable, "test_masks.py"], capture_output=True, text=True, cwd=repo, check=True)
    # The wording of bisect's final message changes between git versions; the ref does not.
    bad = subprocess.run(["git", "-C", str(repo), "rev-parse", "refs/bisect/bad"], capture_output=True, text=True, check=True).stdout.strip()
    check = subprocess.run([sys.executable, kata, "check-bisect", str(repo), bad], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout


def test_git_kata_conflict_and_upstream_build(tmp_path):
    kata = "exercises/week08/git_kata.py"
    subprocess.run([sys.executable, kata, "conflict", str(tmp_path / "c")], check=True, capture_output=True)
    subprocess.run([sys.executable, kata, "upstream", str(tmp_path / "u")], check=True, capture_output=True)
    assert (tmp_path / "u" / "mine" / "model.py").exists()
