"""Week 2: write the toy events to disk, as parquet (like hepattn's TrackML prep) and as HDF5.

Compare with src/hepattn/experiments/trackml/prep.py, which turns TrackML's CSV files
into ``event...-hits.parquet`` / ``event...-parts.parquet`` pairs.

Why two formats?
  * **Parquet**: columnar, compressed, one small file per event. hepattn's TrackML and
    ColliderML loaders read it with pandas / pyarrow.
  * **HDF5**: one big file holding many arrays in a tree of *groups* (like folders)
    and *datasets* (like files). hepattn writes its predictions in HDF5
    (callbacks/prediction_writer.py), the hit filter's outputs are read back from HDF5
    by the tracking data loader (``hit_eval_path``), and the ATLAS/ITk/CLD/TIDE
    experiments read their inputs from HDF5.

Command line (provided):

    python -m minihep.data.prep --out data/toy --train 2000 --val 200 --test 200
    python -m minihep.data.prep --out data/toy --hdf5        # also write data/toy/{split}.h5
"""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import pandas as pd

from minihep.data.toy_detector import EventConfig, Geometry, event_name, simulate_event

# Event ids for each split, kept far apart so the splits never overlap.
SPLIT_FIRST_EVENT = {"train": 1, "val": 900_001, "test": 950_001}


def write_event_parquet(hits: pd.DataFrame, particles: pd.DataFrame, out_dir: Path, event_id: int) -> tuple[Path, Path]:
    """Write one event as ``{event_name}-hits.parquet`` and ``{event_name}-parts.parquet`` in ``out_dir``.

    Create ``out_dir`` if it does not exist. Do not write the pandas index
    (``index=False``). Return the two paths (hits first).
    """
    # TODO(week02): mkdir(parents=True, exist_ok=True), then DataFrame.to_parquet for each table
    raise NotImplementedError("week02 exercise (prep.py)")


def generate_split(out_dir: Path, num_events: int, first_event_id: int, overwrite: bool = False, seed: int = 0) -> int:
    """Simulate ``num_events`` events with ids ``first_event_id, first_event_id + 1, ...`` and write them.

    Skip events whose two files already exist unless ``overwrite`` (the same rule as
    hepattn's prep.py, so an interrupted run can be resumed). Return the number of
    events actually written.
    """
    out_dir = Path(out_dir)
    written = 0
    for event_id in range(first_event_id, first_event_id + num_events):
        name = event_name(event_id)
        if not overwrite and (out_dir / f"{name}-hits.parquet").exists() and (out_dir / f"{name}-parts.parquet").exists():
            continue
        hits, particles = simulate_event(event_id, Geometry(), EventConfig(), seed=seed)
        write_event_parquet(hits, particles, out_dir, event_id)
        written += 1
    return written


def parquet_dir_to_hdf5(in_dir: Path, out_path: Path, compression: str | None = "lzf") -> int:
    """Pack every event in ``in_dir`` into a single HDF5 file.

    Layout (one group per event, one dataset per column):

        /event000000001/hits/x          float32 (num_hits,)
        /event000000001/hits/particle_id int64  (num_hits,)
        ...
        /event000000001/parts/px        float32 (num_particles,)
        ...

    Also store two *attributes* on each event group: ``num_hits`` and ``num_particles``.
    Use ``compression`` for every dataset (lzf is fast; gzip is smaller).
    Process events in sorted name order. Return the number of events written.

    Hints: ``h5py.File(path, "w")``, ``group.create_group(name)``,
    ``group.create_dataset(name, data=array, compression=...)``, ``group.attrs[key] = value``,
    ``df[column].to_numpy()``.
    """
    # TODO(week02): open the file with a `with` block, loop over sorted *-hits.parquet files, write one group per event
    raise NotImplementedError("week02 exercise (prep.py)")


def read_event_hdf5(file: h5py.File, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one event back from an open HDF5 file as ``(hits, particles)`` DataFrames.

    ``dataset[()]`` (or ``dataset[:]``) reads the whole array into memory as numpy.
    The values must match the parquet version exactly. (Column *order* may differ:
    h5py lists a group's keys alphabetically. Nothing downstream cares, since columns
    are always looked up by name.)
    """
    # TODO(week02): build a dict {column: dataset[()]} for "hits" and for "parts", wrap each in a DataFrame
    raise NotImplementedError("week02 exercise (prep.py)")


def main() -> None:
    parser = ArgumentParser(description="Generate the toy tracking dataset")
    parser.add_argument("--out", type=Path, default=Path("data/toy"))
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hdf5", action="store_true", help="also pack each split into {out}/{split}.h5")
    args = parser.parse_args()

    for split, num in (("train", args.train), ("val", args.val), ("test", args.test)):
        n = generate_split(args.out / split, num, SPLIT_FIRST_EVENT[split], overwrite=args.overwrite, seed=args.seed)
        print(f"{split}: wrote {n} new events to {args.out / split}")
        if args.hdf5:
            n_h5 = parquet_dir_to_hdf5(args.out / split, args.out / f"{split}.h5")
            print(f"{split}: packed {n_h5} events into {args.out / f'{split}.h5'}")


if __name__ == "__main__":
    main()
