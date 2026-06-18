from collections import OrderedDict

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from base.base_data_loader import BaseDataLoader


def _collate_with_index(batch):
    inputs, targets, indices = zip(*batch)
    return torch.stack(inputs), torch.stack(targets), torch.tensor(indices, dtype=torch.long)


class NeSPReSODataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx


def _split_lengths(n: int, train_frac: float, val_frac: float, test_frac: float):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1")
    train_len = int(n * train_frac)
    val_len = int(n * val_frac)
    test_len = n - train_len - val_len
    return train_len, val_len, test_len


class NeSPReSODataLoader(DataLoader):
    """Loads train-ready pickle; 70/15/15 split via ``torch.random_split`` + seed."""

    def __init__(
        self,
        cache_path,
        batch_size,
        shuffle=True,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        split_seed=42,
        split="train",
        num_workers=0,
        pin_memory=False,
        target_key="targets",
        weight_key="weights",
        **kwargs,
    ):
        kwargs.pop("validation_split", None)
        kwargs.pop("training", None)
        kwargs.pop("batch_size_safety", None)
        target_key = kwargs.pop("target_key", target_key)
        weight_key = kwargs.pop("weight_key", weight_key)

        with open(cache_path, "rb") as f:
            self.cache = pickle.load(f)

        if target_key not in self.cache:
            raise KeyError(f"cache missing {target_key!r}; run scripts/export_ae_latents.py for decoder training")
        if weight_key not in self.cache:
            weight_key = "weights"

        inputs = torch.tensor(self.cache["inputs"], dtype=torch.float32)
        targets = torch.tensor(self.cache[target_key], dtype=torch.float32)
        full_ds = NeSPReSODataset(inputs, targets)

        n = len(full_ds)
        train_len, val_len, test_len = _split_lengths(n, train_frac, val_frac, test_frac)
        g = torch.Generator().manual_seed(int(split_seed))
        train_sub, val_sub, test_sub = random_split(
            full_ds, [train_len, val_len, test_len], generator=g
        )
        self.train_subset = train_sub
        self.val_subset = val_sub
        self.test_subset = test_sub
        self.split_indices = {
            "train": list(train_sub.indices),
            "val": list(val_sub.indices),
            "test": list(test_sub.indices),
        }

        subsets = {"train": train_sub, "val": val_sub, "test": test_sub}
        if split not in subsets:
            raise ValueError(f"split must be train|val|test, got {split}")
        active = subsets[split]

        self.cache_path = cache_path
        self.pca_models = self.cache["pca_models"]
        self.outputs = OrderedDict(self.cache["outputs"])
        self.weights = self.cache[weight_key]
        self.LAT = self.cache["LAT"]
        self.LON = self.cache["LON"]
        self.PRES = self.cache.get("PRES")
        self.profiles = self.cache.get("profiles")
        self.input_params = self.cache.get("input_params", {})
        self.dataset_tag = self.cache.get("dataset_tag", "unknown")
        self.min_depth = self.cache.get("min_depth", 0)
        self.max_depth = self.cache.get("max_depth", targets.shape[1] - 1)
        self.batch_size = batch_size
        self._split = split

        super().__init__(
            active,
            batch_size=batch_size,
            shuffle=shuffle and split == "train",
            num_workers=num_workers,
            collate_fn=_collate_with_index,
            pin_memory=pin_memory,
        )

    def split_validation(self):
        dl = DataLoader(
            self.val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate_with_index,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        dl.pca_models = self.pca_models
        dl.outputs = self.outputs
        return dl

    def split_test(self):
        dl = DataLoader(
            self.test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate_with_index,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        dl.pca_models = self.pca_models
        dl.outputs = self.outputs
        return dl


class MnistDataLoader(BaseDataLoader):
    """Legacy MNIST loader (template default)."""

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=True):
        from torchvision import datasets, transforms

        trs = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = datasets.MNIST(data_dir, train=training, download=True, transform=trs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
