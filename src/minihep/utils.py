"""Small helpers for notebooks and scripts (provided)."""

from pathlib import Path

from minihep.data.toy_detector import EventConfig, event_name, simulate_event

SPLITS = {"train": 1, "val": 900_001, "test": 950_001}


def repo_root(start: Path | None = None) -> Path:
    """The directory containing pyproject.toml, searching upwards (so notebooks work from notebooks/)."""
    here = (start or Path.cwd()).resolve()
    for path in (here, *here.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "minihep").exists():
            return path
    raise FileNotFoundError("Could not find the hepattn-course repository root")


def ensure_toy_data(num_train: int = 2000, num_val: int = 200, num_test: int = 200, root: Path | None = None, seed: int = 0) -> Path:
    """Make sure ``data/toy/{train,val,test}`` hold at least the requested number of events; return ``data/toy``.

    Independent of the week 2 exercises (it writes parquet with pandas directly), so every
    notebook can call it even if you skipped ahead.
    """
    out = (root or repo_root()) / "data" / "toy"
    for split, num in (("train", num_train), ("val", num_val), ("test", num_test)):
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        first = SPLITS[split]
        for event_id in range(first, first + num):
            name = event_name(event_id)
            if (split_dir / f"{name}-parts.parquet").exists():
                continue
            hits, parts = simulate_event(event_id, config=EventConfig(), seed=seed)
            hits.to_parquet(split_dir / f"{name}-hits.parquet", index=False)
            parts.to_parquet(split_dir / f"{name}-parts.parquet", index=False)
    return out
