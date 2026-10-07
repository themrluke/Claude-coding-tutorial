"""Entry point for the hit filter: ``python -m minihep.run_filter fit --config configs/toy_filter.yaml``.

Mirror of src/hepattn/experiments/trackml/run_filtering.py. Provided.
"""

from lightning.pytorch.cli import ArgsType

from minihep.cli import CLI
from minihep.data.datamodule import ToyDataModule
from minihep.lightning.wrapper import Filter


def main(args: ArgsType = None) -> None:
    CLI(model_class=Filter, datamodule_class=ToyDataModule, args=args, parser_kwargs={"default_env": True})


if __name__ == "__main__":
    main()
