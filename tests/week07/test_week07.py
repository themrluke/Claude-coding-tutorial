import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from exercises.week07 import ex01_jsonargparse as ex01
from minihep.cli import get_best_epoch, timestamped_run_dir
from minihep.data.dataset import ToyTrackingDataset
from minihep.models.dense import Dense
from tests.conftest import TOY_INPUTS, TOY_TARGETS, write_toy_events

# ----------------------------------------------------------------------------- jsonargparse


@pytest.fixture
def config_files(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        """
name: base-run
lr: 0.01
tags: [a, b]
model:
  class_path: minihep.models.dense.Dense
  init_args:
    input_size: &dim 8
    output_size: *dim
    hidden_layers: [16]
"""
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("lr: 0.5\ntags: [c]\n")
    return base, overlay


def test_parse_types_and_defaults(config_files):
    base, _ = config_files
    cfg = ex01.parse(["--config", str(base)])
    assert cfg.name == "base-run"
    assert cfg.lr == pytest.approx(0.01)
    assert cfg.epochs == 10
    assert cfg.tags == ["a", "b"]
    # jsonargparse normalises class_path to the shortest import path that works
    assert cfg.model.class_path in {"minihep.models.dense.Dense", "minihep.models.Dense"}
    assert cfg.model.init_args.output_size == 8  # the alias resolved


def test_later_configs_and_flags_override(config_files):
    base, overlay = config_files
    cfg = ex01.parse(["--config", str(base), "--config", str(overlay), "--epochs", "3"])
    assert cfg.lr == pytest.approx(0.5)
    assert cfg.tags == ["c"]
    assert cfg.epochs == 3
    assert cfg.name == "base-run"


def test_bad_types_are_rejected(config_files, capsys):
    base, _ = config_files
    with pytest.raises(SystemExit):
        ex01.parse(["--config", str(base), "--epochs", "three"])
    with pytest.raises(SystemExit):
        ex01.parse(["--config", str(base), "--model.init_args.hidden_layerz", "[4]"])


def test_build_instantiates_the_model(config_files):
    base, _ = config_files
    model, cfg = ex01.build(["--config", str(base), "--model.init_args.hidden_layers", "[4, 4]"])
    assert isinstance(model, Dense)
    assert [m.out_features for m in model.net if isinstance(m, torch.nn.Linear)] == [4, 4, 8]
    assert cfg.lr == pytest.approx(0.01)


def test_dump(config_files):
    base, overlay = config_files
    text = ex01.dump(["--config", str(base), "--config", str(overlay)])
    data = yaml.safe_load(text)
    assert data["lr"] == pytest.approx(0.5)
    assert data["model"]["init_args"]["input_size"] == 8


def test_deep_merge():
    base = {"trainer": {"max_epochs": 10, "logger": {"class_path": "CSV", "init_args": {"save_dir": "logs"}}}, "tags": [1, 2]}
    override = {"trainer": {"logger": {"class_path": "Comet", "init_args": {"project": "x"}}}, "tags": [3]}
    merged = ex01.deep_merge(base, override)
    assert merged["trainer"]["max_epochs"] == 10
    assert merged["trainer"]["logger"]["class_path"] == "Comet"
    assert merged["trainer"]["logger"]["init_args"] == {"save_dir": "logs", "project": "x"}
    assert merged["tags"] == [3]
    assert base["trainer"]["logger"]["class_path"] == "CSV", "inputs must not be modified"


# ----------------------------------------------------------------------------- CLI helpers


def test_get_best_epoch(tmp_path):
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir()
    for name in ("epoch=000-val_loss=0.90000.ckpt", "epoch=001-val_loss=0.45000.ckpt", "epoch=002-val_loss=0.50000.ckpt", "notes.txt"):
        (ckpts / name).touch()
    assert get_best_epoch(tmp_path).name == "epoch=001-val_loss=0.45000.ckpt"
    with pytest.raises(FileNotFoundError):
        get_best_epoch(tmp_path / "nothing")


def test_timestamped_run_dir(tmp_path):
    now = datetime(2026, 10, 7, 9, 30, 0)
    first = timestamped_run_dir(str(tmp_path / "logs"), "toy", now)
    assert first == str((tmp_path / "logs" / "toy_20261007-T093000").resolve())
    again = timestamped_run_dir(first, "toy", datetime(2026, 10, 8, 9, 30, 0))
    assert again == str((tmp_path / "logs" / "toy_20261008-T093000").resolve())


# ----------------------------------------------------------------------------- dataset reads filter output


def test_dataset_applies_hit_filter(toy_dir, tmp_path):
    ds = ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS)
    n = ds[0][0]["hit_valid"].shape[-1]
    probs = np.linspace(0, 1, n, dtype=np.float32)[None]
    path = tmp_path / "filter.h5"
    with h5py.File(path, "w") as f:
        for name in ds.event_names:
            hits = pd.read_parquet(Path(toy_dir) / f"{name}-hits.parquet")
            p = probs if name == ds.event_names[0] else np.ones((1, len(hits)), dtype=np.float32)
            f.create_dataset(f"{name}/preds/final/hit_filter/hit_on_valid_particle_prob", data=p)
    filtered = ToyTrackingDataset(str(toy_dir), TOY_INPUTS, TOY_TARGETS, hit_eval_path=str(path), hit_filter_threshold=0.5)
    inputs, _ = filtered[0]
    assert inputs["hit_valid"].shape[-1] == int((probs >= 0.5).sum())
    assert torch.equal(inputs["hit_x"][0], ds[0][0]["hit_x"][0, torch.from_numpy(probs[0] >= 0.5)])


# ----------------------------------------------------------------------------- the whole CLI


@pytest.fixture(scope="module")
def cli_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    for split, first, n in (("train", 1, 6), ("val", 900_001, 3), ("test", 950_001, 3)):
        write_toy_events(root / "data" / split, n, first)
    return root


def _common(root: Path) -> list[str]:
    return [
        *("--data.train_dir", str(root / "data/train"), "--data.val_dir", str(root / "data/val"), "--data.test_dir", str(root / "data/test")),
        *("--data.num_val", "-1", "--data.num_workers", "0", "--trainer.default_root_dir", str(root / "logs")),
        *("--trainer.max_epochs", "1", "--trainer.accelerator", "cpu", "--trainer.enable_progress_bar", "false", "--trainer.log_every_n_steps", "1"),
    ]


@pytest.mark.slow
def test_filter_then_tracking_through_the_cli(cli_data, monkeypatch):
    from minihep import run_filter, run_tracking

    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    root = cli_data

    run_filter.main(["fit", "--config", "configs/toy_filter.yaml", *_common(root)])
    run = max((root / "logs").glob("toy-filter_*"))
    assert {"config.yaml", "metadata.yaml", "metrics.csv", "ckpts"} <= {p.name for p in run.iterdir()}
    for split in ("train", "val", "test"):
        run_filter.main(["test", "--config", str(run / "config.yaml"), "--data.test_dir", str(root / "data" / split)])
    files = {split: next((run / "ckpts").glob(f"*__{split}.h5")) for split in ("train", "val", "test")}
    with h5py.File(files["test"], "r") as f:
        assert len(f) == 3
        group = f[sorted(f)[0]]
        assert group["preds/final/hit_filter/hit_on_valid_particle_prob"].shape[0] == 1
        assert "sample_id" in group.attrs

    hit_eval = [arg for split in ("train", "val", "test") for arg in (f"--data.hit_eval_{split}", str(files[split]))]
    run_tracking.main(["fit", "--config", "configs/toy_tracking.yaml", *_common(root), *hit_eval, "--data.hit_filter_threshold", "0.05"])
    trun = max((root / "logs").glob("toy-tracking_*"))
    metrics = pd.read_csv(trun / "metrics.csv")
    for key in ("train/loss", "val/loss", "val/final_track_valid_object_bce", "val/p0.5_eff", "val/layer_0_loss"):
        assert key in metrics.columns, key
    run_tracking.main(["test", "--config", str(trun / "config.yaml")])
    out = next((trun / "ckpts").glob("*__test.h5"))
    with h5py.File(out, "r") as f:
        group = f[sorted(f)[0]]
        q = group["preds/final/track_valid/track_valid"].shape[-1]
        assert group["preds/final/track_hit_valid/track_hit_valid"].shape[:2] == (1, q)
        assert group["targets/particle_hit_valid"].shape[1] == q


def test_comet_logger_offline(tmp_path):
    pytest.importorskip("comet_ml")
    from minihep.lightning.loggers import CometLogger

    logger = CometLogger(name="offline-test", project="minihep", offline_directory=str(tmp_path), online=False)
    logger.log_metrics({"val/loss": 1.0}, step=0)
    assert type(logger.experiment).__name__ == "OfflineExperiment"
