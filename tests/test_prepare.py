import numpy as np
import torch

from prepare import NUMERIC, RentalsDataset, make_loaders, split_frame


def test_dataset_length_and_float32_shapes(prepared):
    dataset = RentalsDataset(prepared.x_train, prepared.y_train)
    assert len(dataset) == len(prepared.x_train)
    x, y = dataset[0]
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == (prepared.n_features,)
    assert y.shape == ()


def test_split_sizes_add_up_and_do_not_overlap(frame):
    train, val, test = split_frame(frame)
    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert len(train) + len(val) + len(test) == len(frame)
    ids = [set(part.index) for part in (train, val, test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert set().union(*ids) == set(frame.index)


def test_scaler_is_fitted_on_training_rows_only(prepared):
    n = len(NUMERIC)
    assert np.allclose(prepared.x_train[:, :n].mean(axis=0), 0.0, atol=1e-5)
    # If the scaler had seen the validation rows, these would be exactly zero
    # too. They are close to zero, but not equal to it.
    assert not np.allclose(prepared.x_val[:, :n].mean(axis=0), 0.0, atol=1e-5)


def test_training_loader_shuffles_but_is_repeatable(prepared):
    def first_targets(seed):
        train, _, _ = make_loaders(prepared, batch_size=64, seed=seed)
        return next(iter(train))[1]

    assert torch.equal(first_targets(7), first_targets(7))
    assert not torch.equal(first_targets(7), first_targets(8))
    train, val, _ = make_loaders(prepared, batch_size=64, seed=7)
    # Validation batches are never shuffled: two passes give the same order.
    assert torch.equal(next(iter(val))[1], next(iter(val))[1])
    assert torch.equal(next(iter(val))[1], torch.as_tensor(prepared.y_val[:512]))
