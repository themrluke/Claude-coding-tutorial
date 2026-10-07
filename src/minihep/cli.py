"""Week 7: the command line. Mirror of src/hepattn/utils/cli.py.

``LightningCLI`` builds an argument parser from the *type-hinted signatures* of your
LightningModule, LightningDataModule and Trainer (via jsonargparse), reads YAML configs and
command-line overrides, instantiates everything, and runs a subcommand:

    python -m minihep.run_tracking fit  --config configs/toy_tracking.yaml
    python -m minihep.run_tracking fit  --config configs/toy_tracking.yaml --config configs/comet.yaml
    python -m minihep.run_tracking fit  --config configs/toy_tracking.yaml --trainer.max_epochs 2 --data.num_train 50
    python -m minihep.run_tracking fit  --config configs/toy_tracking.yaml --print_config      # resolved config, then exit
    python -m minihep.run_tracking test --config logs/toy-tracking_20261007-T101500/config.yaml

Later ``--config`` files and ``--a.b.c value`` flags override earlier ones, key by key.

This subclass adds what hepattn adds:
  * ``--name`` (linked into the model's ``name``),
  * a timestamped run directory per training run: ``{default_root_dir}/{name}_{YYYYmmdd-THHMMSS}``,
  * for ``test``: no logger, and the best checkpoint picked automatically from ``ckpts/``,
  * ``--matmul_precision`` (TF32 on Ampere+ GPUs: week 8).
"""

import re
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch.cli import LightningCLI

RUN_TIMESTAMP_FORMAT = "%Y%m%d-T%H%M%S"


def get_best_epoch(run_dir: Path) -> Path:
    """Return the checkpoint in ``run_dir / "ckpts"`` with the lowest ``val_loss=`` in its file name.

    File names look like ``epoch=007-val_loss=0.12345.ckpt`` (see callbacks.Checkpoint). Raise
    FileNotFoundError if there are none. (hepattn: get_best_epoch, with a regex lookbehind.)
    """
    # TODO(week07): glob *.ckpt, pull the number after "loss=" with a regex, return the argmin
    raise NotImplementedError("week07 exercise (cli.py)")


def timestamped_run_dir(default_root_dir: str, name: str, now: datetime | None = None) -> str:
    """``{default_root_dir}/{name}_{timestamp}``. If ``default_root_dir`` is *already* a run directory
    (its last ``_``-separated part parses as a timestamp, which happens when you re-train from a saved
    config.yaml), use its parent instead so runs do not nest. Returns an absolute path string.
    """
    # TODO(week07): detect an existing timestamp with datetime.strptime inside try/except ValueError
    raise NotImplementedError("week07 exercise (cli.py)")


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        """Add ``--name`` (str, default "minihep") and ``--matmul_precision`` (choices highest/high/medium,
        default "highest"), and link ``name`` to ``model.name`` with ``parser.link_arguments``.

        A linked argument is computed, so it must NOT appear in the YAML under ``model:``.
        """
        # TODO(week07): two add_argument calls and one link_arguments
        raise NotImplementedError("week07 exercise (cli.py)")

    def before_instantiate_classes(self) -> None:
        """Edit the parsed config before anything is built. ``sc = self.config[self.subcommand]``
        behaves like a dict with dotted keys, e.g. ``sc["trainer.default_root_dir"]``.

        fit:
          - ``sc["trainer.default_root_dir"] = timestamped_run_dir(sc["trainer.default_root_dir"], sc["name"])``
          - if a logger is configured (``sc["trainer.logger"]`` has ``init_args``), point it at the run dir:
            set ``init_args.save_dir`` if that key exists (CSVLogger) and ``init_args.offline_directory`` if
            that one does (Comet). For CSVLogger also set name and version to "" so its files land in the run
            dir itself (LightningCLI saves config.yaml into ``trainer.log_dir``). For Comet set
            ``init_args.name = sc["name"]`` so the experiment is named after the run.
        test:
          - ``self.save_config_callback = None`` (the run dir already has its config.yaml)
          - ``sc["trainer.logger"] = False``
          - if ``sc["ckpt_path"]`` is None: the run dir is the folder of the first ``--config`` file
            (``Path(str(sc["config"][0])).parent``); use ``get_best_epoch`` on it.
        always:
          - ``torch.set_float32_matmul_precision(sc["matmul_precision"])``
        """
        sc = self.config[self.subcommand]
        # TODO(week07): the fit, test and always branches
        raise NotImplementedError("week07 exercise (cli.py)")
