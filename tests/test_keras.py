"""Tests for the Keras version, the weight copy and the comparison."""

from dataclasses import replace

import numpy as np
import pytest
import torch

import keras_model as km
from bikes_data import make_rentals
from compare import COLUMNS, compare
from model import RentalNet
from prepare import make_loaders, prepare
from train import MAIN
from train_keras import keras_arrays, train_keras_run

TINY = replace(MAIN, name="tiny", hidden_size=16, batch_size=64, epochs=4)


@pytest.fixture(scope="module")
def small_prepared():
    """A smaller table, so the Keras training tests stay quick."""
    return prepare(make_rentals(n=600))


@pytest.fixture
def torch_net(prepared):
    torch.manual_seed(0)
    return RentalNet(prepared.n_features, hidden_size=32, n_hidden=2)


def test_keras_network_returns_one_value_per_row(prepared):
    model = km.build_keras_model(prepared.n_features, hidden_size=16)
    x = prepared.x_val[:10]
    assert model(x).shape == (10, 1)
    assert km.predict_keras(model, x).shape == (10,)


@pytest.mark.parametrize("hidden_size", [4, 16, 64, 128])
@pytest.mark.parametrize("n_hidden", [2, 3])
def test_parameter_count_matches_pytorch(prepared, hidden_size, n_hidden):
    torch_model = RentalNet(prepared.n_features, hidden_size, n_hidden)
    keras_model = km.build_keras_model(prepared.n_features, hidden_size, n_hidden)
    assert keras_model.count_params() == km.count_torch_parameters(torch_model)


def test_weight_copy_gives_the_same_predictions(prepared, torch_net):
    # Real validation rows, not zeros: with a zero input every kernel is
    # multiplied by nothing and a wrong copy would still look right.
    keras_net = km.build_keras_model(prepared.n_features, hidden_size=32)
    km.copy_torch_weights(torch_net, keras_net)
    torch_pred = km.predict_torch(torch_net, prepared.x_val)
    keras_pred = km.predict_keras(keras_net, prepared.x_val)
    np.testing.assert_allclose(keras_pred, torch_pred, atol=km.EQUIVALENCE_TOLERANCE)
    assert km.equivalence_check(torch_net, prepared.x_val) < km.EQUIVALENCE_TOLERANCE


def test_copy_without_the_transpose_is_caught(prepared):
    # The hidden-to-hidden kernel is square, so leaving out the transpose
    # raises no shape error there. The predictions must give it away.
    torch.manual_seed(1)
    torch_model = RentalNet(prepared.n_features, hidden_size=32)
    keras_model = km.build_keras_model(prepared.n_features, hidden_size=32)
    km.copy_torch_weights(torch_model, keras_model)
    square = keras_model.get_layer("hidden_2")
    kernel, bias = square.get_weights()
    square.set_weights([kernel.T, bias])
    gap = km.max_prediction_gap(torch_model, keras_model, prepared.x_val)
    assert gap > 100 * km.EQUIVALENCE_TOLERANCE


def test_both_versions_use_the_same_rows(prepared):
    train_loader, val_loader, test_loader = make_loaders(prepared, batch_size=64)
    arrays = keras_arrays(prepared)
    for split, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        dataset = loader.dataset
        np.testing.assert_array_equal(arrays[f"x_{split}"], dataset.features.numpy())
        np.testing.assert_array_equal(arrays[f"y_{split}"], dataset.targets.numpy())
        assert arrays[f"x_{split}"].dtype == np.float32


def test_keras_network_can_memorize_one_small_batch(prepared):
    x, y = prepared.x_train[:32], prepared.y_train[:32]
    km.keras.utils.set_random_seed(0)
    model = km.build_keras_model(prepared.n_features, hidden_size=32)
    model.compile(optimizer=km.keras.optimizers.Adam(0.01), loss="mse")
    first = model.evaluate(x, y, verbose=0)
    model.fit(x, y, batch_size=32, epochs=300, verbose=0)
    last = model.evaluate(x, y, verbose=0)
    assert last < 0.05 * first


def test_saved_and_reloaded_model_gives_the_same_predictions(prepared, tmp_path):
    km.keras.utils.set_random_seed(0)
    model = km.build_keras_model(prepared.n_features, hidden_size=16)
    model.compile(optimizer="adam", loss="mse")
    model.fit(prepared.x_train[:64], prepared.y_train[:64], epochs=1, verbose=0)
    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = km.keras.models.load_model(path)
    np.testing.assert_allclose(
        km.predict_keras(reloaded, prepared.x_val),
        km.predict_keras(model, prepared.x_val),
        rtol=0,
        atol=1e-6,
    )


def test_best_weights_are_restored(small_prepared):
    # A large learning rate makes the validation loss jump about, so the best
    # epoch is not the last one and restoring the weights actually matters.
    config = replace(TINY, lr=0.3, epochs=3)
    model, history, best_epoch = train_keras_run(config, small_prepared, seed=0)
    assert len(history) == 3
    assert best_epoch < len(history), "pick settings where the last epoch is worse"
    restored = model.evaluate(
        small_prepared.x_val, small_prepared.y_val, batch_size=512, verbose=0
    )[0]
    assert restored == pytest.approx(history[best_epoch - 1]["val_loss"], rel=1e-4)


def test_same_seed_repeats_a_short_run(small_prepared):
    first_model, first, _ = train_keras_run(TINY, small_prepared, seed=7)
    second_model, second, _ = train_keras_run(TINY, small_prepared, seed=7)
    assert all(
        set(row) == {"epoch", "train_loss", "val_loss", "val_mae"} for row in first
    )
    for row_a, row_b in zip(first, second, strict=True):
        assert row_a == pytest.approx(row_b, rel=1e-6)
    np.testing.assert_allclose(
        km.predict_keras(first_model, small_prepared.x_val),
        km.predict_keras(second_model, small_prepared.x_val),
        rtol=1e-6,
    )


def test_comparison_has_one_row_per_framework_and_seed(small_prepared):
    table = compare(
        small_prepared, config=replace(TINY, epochs=2), seeds=(0, 1), verbose=False
    )
    assert list(table.columns) == COLUMNS
    assert len(table) == 4
    assert sorted(table["framework"].unique()) == ["keras", "pytorch"]
    assert table.groupby("framework")["seed"].apply(sorted).tolist() == [[0, 1], [0, 1]]
    assert (table["seconds"] > 0).all() and (table["test_mae"] > 0).all()
    assert table["best_epoch"].between(1, 2).all()
