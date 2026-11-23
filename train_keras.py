"""The Keras version of the main run: compile, fit and two callbacks.

Run `python train_keras.py` to recreate the Keras files in reports/.

The settings are not copied by hand. They come from train.MAIN, the main
PyTorch run, so both versions use the same rows, loss, optimizer, learning
rate, batch size, number of epochs and seed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import keras_model as km
from bikes_data import make_rentals
from keras_model import keras  # through keras_model: backend and import order
from model import RentalNet
from prepare import Prepared, prepare
from train import MAIN, REPORTS, SEED, RunConfig, plot_curves, scores

CHECKPOINT = Path("checkpoints") / "keras_main.keras"


def keras_arrays(prepared: Prepared) -> dict[str, np.ndarray]:
    """The arrays handed to Keras: the same arrays the PyTorch loaders wrap.

    Nothing is prepared a second time. Keras receives prepare.py's float32
    arrays as they are.
    """
    return {
        "x_train": prepared.x_train,
        "y_train": prepared.y_train,
        "x_val": prepared.x_val,
        "y_val": prepared.y_val,
        "x_test": prepared.x_test,
        "y_test": prepared.y_test,
    }


def make_optimizer(lr: float) -> keras.optimizers.Optimizer:
    """Adam with PyTorch's defaults. Keras uses epsilon=1e-7, PyTorch 1e-8."""
    return keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    )


def train_keras_run(
    config: RunConfig,
    prepared: Prepared,
    seed: int = SEED,
    checkpoint: Path | None = None,
    verbose: int = 0,
):
    """Build, compile and fit one network. Uses training and validation rows only.

    EarlyStopping with patience equal to the number of epochs never stops the
    run early, so it trains as long as the PyTorch run. restore_best_weights
    puts back the best epoch, as the PyTorch fit does by hand.

    Returns the model, the history (one dict per epoch, with the same keys as
    the PyTorch history) and the best epoch, counted from 1.
    """
    keras.utils.set_random_seed(seed)
    model = km.build_keras_model(
        prepared.n_features, config.hidden_size, config.n_hidden
    )
    model.compile(optimizer=make_optimizer(config.lr), loss="mse", metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.epochs, restore_best_weights=True
        )
    ]
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_loss", save_best_only=True
            )
        )

    arrays = keras_arrays(prepared)
    fitted = model.fit(
        arrays["x_train"],
        arrays["y_train"],
        validation_data=(arrays["x_val"], arrays["y_val"]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    logged = fitted.history
    history = [
        {"epoch": epoch, "train_loss": train, "val_loss": val, "val_mae": mae}
        for epoch, (train, val, mae) in enumerate(
            zip(logged["loss"], logged["val_loss"], logged["val_mae"], strict=True),
            start=1,
        )
    ]
    best_epoch = int(np.argmin(logged["val_loss"])) + 1
    return model, history, best_epoch


def main():
    REPORTS.mkdir(exist_ok=True)
    prepared = prepare(make_rentals())

    # Before training anything: the two networks must compute the same function.
    torch.manual_seed(SEED)
    torch_net = RentalNet(prepared.n_features, MAIN.hidden_size, MAIN.n_hidden)
    gap = km.equivalence_check(torch_net, prepared.x_val)
    print(f"Equivalence check on the validation rows: largest gap {gap:.2e} rentals")

    model, history, best_epoch = train_keras_run(
        MAIN, prepared, SEED, checkpoint=CHECKPOINT, verbose=2
    )
    pd.DataFrame(history).to_csv(REPORTS / "history_keras.csv", index=False)
    plot_curves(
        history,
        best_epoch,
        "Keras main run: training and validation loss",
        REPORTS / "learning_curves_keras.png",
    )

    # The test rows are used once, for the main network chosen on validation.
    metrics = {
        "framework": f"keras {keras.__version__} ({keras.backend.backend()})",
        "parameters": model.count_params(),
        "equivalence": {
            "rows": "validation",
            "largest_gap": gap,
            "tolerance": km.EQUIVALENCE_TOLERANCE,
        },
        "best_epoch": best_epoch,
        "val": scores(prepared.y_val, km.predict_keras(model, prepared.x_val)),
        "test": scores(prepared.y_test, km.predict_keras(model, prepared.x_test)),
    }
    (REPORTS / "metrics_keras.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
