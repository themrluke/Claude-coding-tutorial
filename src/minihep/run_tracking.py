"""Entry point for tracking: ``python -m minihep.run_tracking fit --config configs/toy_tracking.yaml``.

Mirror of src/hepattn/experiments/trackml/run_tracking.py. Provided: everything interesting
is in cli.py, lightning/wrapper.py and the YAML.
"""

from lightning.pytorch.cli import ArgsType

from minihep.cli import CLI
from minihep.data.datamodule import ToyDataModule
from minihep.lightning.wrapper import Tracker


def main(args: ArgsType = None) -> None:
    # default_env=True lets every option also be set by an environment variable,
    # e.g. PL_FIT__TRAINER__MAX_EPOCHS=3 (print them with --help).
    CLI(model_class=Tracker, datamodule_class=ToyDataModule, args=args, parser_kwargs={"default_env": True})


if __name__ == "__main__":
    main()
