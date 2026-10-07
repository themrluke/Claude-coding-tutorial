import os

import pytest
import torch

# Tiny models run faster on a few threads than on dozens (thread start-up dominates).
torch.set_num_threads(min(4, os.cpu_count() or 1))


def pytest_collection_modifyitems(config, items):
    has_gpu = torch.cuda.is_available()
    try:
        import flash_attn  # noqa: F401

        has_flash = has_gpu
    except ImportError:
        has_flash = False

    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    skip_flash = pytest.mark.skip(reason="flash-attn not installed (use `pixi run -e fa2 ...`)")
    for item in items:
        if item.get_closest_marker("gpu") and not has_gpu:
            item.add_marker(skip_gpu)
        if item.get_closest_marker("flash") and not has_flash:
            item.add_marker(skip_flash)


@pytest.fixture(autouse=True)
def _seed():
    # Every test starts from the same random state, so failures are reproducible.
    torch.manual_seed(0)


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_toy_events(out_dir, num_events: int, first_event_id: int = 1, **config_kwargs):
    """Write toy events as parquet without going through any exercise code."""
    from pathlib import Path

    from minihep.data.toy_detector import EventConfig, event_name, simulate_event

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = EventConfig(**config_kwargs)
    for event_id in range(first_event_id, first_event_id + num_events):
        hits, parts = simulate_event(event_id, config=config)
        hits.to_parquet(out_dir / f"{event_name(event_id)}-hits.parquet", index=False)
        parts.to_parquet(out_dir / f"{event_name(event_id)}-parts.parquet", index=False)
    return out_dir


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    """20 small toy events on disk, shared by every test in the session."""
    return write_toy_events(tmp_path_factory.mktemp("toy") / "train", 20)


TOY_INPUTS = {"hit": ["x", "y", "z", "r", "eta", "phi"]}
TOY_TARGETS = {"hit": ["on_valid_particle", "is_first"], "particle": ["pt", "eta", "phi"]}
