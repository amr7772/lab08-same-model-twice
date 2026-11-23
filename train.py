"""The training loop, the baselines, the experiments and the reports.

Run `python train.py` to recreate everything in reports/.
"""

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch import nn

from bikes_data import make_rentals
from model import RentalNet
from prepare import Prepared, make_loaders, prepare

SEED = 42
REPORTS = Path("reports")


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Run one pass over the loader.

    With an optimizer the model is trained. Without one it is only measured:
    no gradients are tracked and the weights stay as they were.
    Returns the mean loss and the mean absolute error, both per row.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_abs_error = 0.0
    n_rows = 0
    with torch.set_grad_enabled(training):
        for features, targets in loader:
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch = len(targets)
            total_loss += loss.item() * batch
            total_abs_error += (predictions - targets).abs().sum().item()
            n_rows += batch
    return total_loss / n_rows, total_abs_error / n_rows


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    """Train for many epochs and put back the weights of the best epoch.

    Returns the history (one dict per epoch) and the best epoch, counted from 1.
    """
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_epoch


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    name: str
    change: str
    lr: float = 1e-3
    hidden_size: int = 64
    n_hidden: int = 2
    batch_size: int = 64
    epochs: int = 60


MAIN = RunConfig(
    "main",
    "reference run: 2 layers of 128 units, Adam at 3e-3, batches of 128",
    lr=3e-3,
    hidden_size=128,
    n_hidden=2,
    batch_size=128,
    epochs=120,
)

EXPERIMENTS = [
    replace(MAIN, name="lr_tiny", change="learning rate 1e-5", lr=1e-5),
    replace(MAIN, name="lr_huge", change="learning rate 0.3", lr=0.3),
    replace(MAIN, name="small_net", change="8 units per layer", hidden_size=8),
    replace(MAIN, name="wide_net", change="1024 units per layer", hidden_size=1024),
    replace(MAIN, name="long_run", change="400 epochs", epochs=400),
]


def run_experiment(config: RunConfig, prepared: Prepared, seed: int = SEED):
    """Train one network from scratch. Uses training and validation rows only."""
    torch.manual_seed(seed)
    train_loader, val_loader, _ = make_loaders(prepared, config.batch_size, seed)
    model = RentalNet(prepared.n_features, config.hidden_size, config.n_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history, best_epoch = fit(
        model, train_loader, val_loader, nn.MSELoss(), optimizer, config.epochs
    )
    return model, history, best_epoch


def summarize(config: RunConfig, history, best_epoch) -> dict:
    best = history[best_epoch - 1]
    return {
        **asdict(config),
        "best_epoch": best_epoch,
        "best_val_mae": best["val_mae"],
        "best_val_loss": best["val_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
    }


# --------------------------------------------------------------------------
# Baselines and metrics
# --------------------------------------------------------------------------


def scores(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def baseline_scores(prepared: Prepared) -> dict:
    """Training-mean baseline and a linear regression, on the same rows."""
    mean_model = DummyRegressor(strategy="mean").fit(prepared.x_train, prepared.y_train)
    linear = LinearRegression().fit(prepared.x_train, prepared.y_train)
    out = {}
    for name, fitted in [("train_mean", mean_model), ("linear_regression", linear)]:
        out[name] = {
            "val": scores(prepared.y_val, fitted.predict(prepared.x_val)),
            "test": scores(prepared.y_test, fitted.predict(prepared.x_test)),
        }
    return out


def predict(model, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(features, dtype=torch.float32)).numpy()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_curves(history, best_epoch, title, path: Path):
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, [r["train_loss"] for r in history], label="training loss")
    ax.plot(epochs, [r["val_loss"] for r in history], label="validation loss")
    ax.axvline(
        best_epoch, color="grey", linestyle="--", label=f"best epoch {best_epoch}"
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error (rentals squared)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rentals_by_hour(frame: pd.DataFrame, path: Path):
    """The 'before you start' plot: why a straight line cannot fit this."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, weekend, label in [(axes[0], 0, "weekday"), (axes[1], 1, "weekend")]:
        subset = frame[frame["is_weekend"] == weekend]
        for station, group in subset.groupby("station_type"):
            ax.plot(group.groupby("hour")["rentals"].mean(), marker="o", label=station)
        ax.set_title(f"Mean rentals by hour, {label}")
        ax.set_xlabel("hour of day")
    axes[0].set_ylabel("mean rentals")
    axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Everything
# --------------------------------------------------------------------------


def main():
    REPORTS.mkdir(exist_ok=True)
    frame = make_rentals()
    prepared = prepare(frame)
    plot_rentals_by_hour(frame, REPORTS / "rentals_by_hour.png")

    baselines = baseline_scores(prepared)

    rows = []
    main_model = None
    for config in [MAIN, *EXPERIMENTS]:
        model, history, best_epoch = run_experiment(config, prepared)
        rows.append(summarize(config, history, best_epoch))
        (REPORTS / "curves").mkdir(exist_ok=True)
        pd.DataFrame(history).to_csv(
            REPORTS / "curves" / f"history_{config.name}.csv", index=False
        )
        plot_curves(
            history,
            best_epoch,
            f"{config.name}: {config.change}",
            REPORTS / "curves" / f"{config.name}.png",
        )
        if config.name == MAIN.name:
            main_model, main_history, main_best = model, history, best_epoch

    pd.DataFrame(main_history).to_csv(REPORTS / "history.csv", index=False)
    plot_curves(
        main_history,
        main_best,
        "Main run: training and validation loss",
        REPORTS / "learning_curves.png",
    )
    pd.DataFrame(rows).to_csv(REPORTS / "experiments.csv", index=False)

    # The test rows are used once, for the main network chosen on validation.
    metrics = {
        "baselines": baselines,
        "network": {
            "best_epoch": main_best,
            "val": scores(prepared.y_val, predict(main_model, prepared.x_val)),
            "test": scores(prepared.y_test, predict(main_model, prepared.x_test)),
        },
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
