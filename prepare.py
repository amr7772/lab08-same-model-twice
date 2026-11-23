"""Data preparation: the split, the scaler and encoder, the Dataset and loaders.

The order matters. The table is split first. Only then are the scaler and the
one-hot encoder fitted, and they are fitted on the training rows alone, so
nothing about the validation or test rows leaks into the features.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import STATION_TYPES, TARGET

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]
SPLIT_SEED = 2026


def split_frame(
    frame: pd.DataFrame, seed: int = SPLIT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into 60% train, 20% validation and 20% test rows."""
    train, rest = train_test_split(frame, test_size=0.4, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, val, test


def build_transformer() -> ColumnTransformer:
    """Scale the numeric columns and one-hot encode the station type."""
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "station",
                OneHotEncoder(categories=[STATION_TYPES], sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


@dataclass
class Prepared:
    """Features and targets for the three splits, plus the fitted transformer."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    transformer: ColumnTransformer

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


def prepare(frame: pd.DataFrame, seed: int = SPLIT_SEED) -> Prepared:
    """Split first, then fit the transformer on the training rows only."""
    train, val, test = split_frame(frame, seed)
    transformer = build_transformer()
    x_train = transformer.fit_transform(train[NUMERIC + CATEGORICAL])
    x_val = transformer.transform(val[NUMERIC + CATEGORICAL])
    x_test = transformer.transform(test[NUMERIC + CATEGORICAL])
    return Prepared(
        x_train=x_train.astype(np.float32),
        y_train=train[TARGET].to_numpy(dtype=np.float32),
        x_val=x_val.astype(np.float32),
        y_val=val[TARGET].to_numpy(dtype=np.float32),
        x_test=x_test.astype(np.float32),
        y_test=test[TARGET].to_numpy(dtype=np.float32),
        transformer=transformer,
    )


class RentalsDataset(Dataset):
    """One station-hour per item: a feature vector and the rentals as a scalar.

    The target is a zero-dimensional tensor, so a batch of targets has shape
    (N,). The network also returns shape (N,), which keeps the loss honest.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def make_loaders(
    prepared: Prepared, batch_size: int = 64, seed: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the three loaders. Only the training batches are shuffled."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    train = DataLoader(
        RentalsDataset(prepared.x_train, prepared.y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val = DataLoader(
        RentalsDataset(prepared.x_val, prepared.y_val), batch_size=512, shuffle=False
    )
    test = DataLoader(
        RentalsDataset(prepared.x_test, prepared.y_test), batch_size=512, shuffle=False
    )
    return train, val, test
