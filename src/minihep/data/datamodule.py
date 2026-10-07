"""Week 7: the LightningDataModule. Mirror of TrackMLDataModule in experiments/trackml/data.py.

A DataModule bundles "how to build the datasets" and "how to load them" so the Trainer can
ask for ``train_dataloader()`` etc. at the right moment. ``setup(stage)`` is called by
Lightning with ``stage`` in {"fit", "validate", "test", "predict"}: only build what that
stage needs (the test set does not exist while training).

Every extra keyword argument (inputs, targets, the cuts...) is forwarded to the datasets,
so in YAML they sit directly under ``data:`` alongside train_dir etc.
"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from minihep.data.dataset import ToyTrackingDataset


class ToyDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str | None = None,
        num_train: int = -1,
        num_val: int = -1,
        num_test: int = -1,
        num_workers: int = 0,
        pin_memory: bool = False,
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        """fit: build ``self.train_dataset`` and ``self.val_dataset``; test: build ``self.test_dataset``
        (raise ValueError if ``test_dir`` is None). Each with its own num_* and hit_eval_* and ``**self.kwargs``."""
        # TODO(week07): build the datasets this stage needs
        raise NotImplementedError("week07 exercise (datamodule.py)")

    def _loader(self, dataset: ToyTrackingDataset, shuffle: bool) -> DataLoader:
        """``batch_size=None``: every item is already a whole event with a batch dim of 1 (week 2)."""
        # TODO(week07): DataLoader with batch_size=None, the shuffle flag, num_workers and pin_memory
        raise NotImplementedError("week07 exercise (datamodule.py)")

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
