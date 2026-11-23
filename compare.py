"""Train the main configuration in both frameworks over several seeds.

Run `python compare.py` to recreate reports/comparison.csv. It trains six
networks (PyTorch and Keras, three seeds each), so it takes a few minutes.

Every run uses the same prepared rows and the settings of train.MAIN. Each
final network is scored on the test rows once, after training has finished.
"""

import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

import keras_model as km
from bikes_data import make_rentals
from prepare import Prepared, prepare
from train import MAIN, REPORTS, RunConfig, predict, run_experiment, scores
from train_keras import train_keras_run

SEEDS = (42, 43, 44)
COLUMNS = [
    "framework",
    "seed",
    "best_epoch",
    "val_mae",
    "test_mae",
    "seconds",
    "model_file_kb",
]


def saved_size_kb(save, suffix: str) -> float:
    """Save a model the way its framework does by default and measure the file."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"model{suffix}"
        save(path)
        return path.stat().st_size / 1024


def run_pytorch(config: RunConfig, prepared: Prepared, seed: int) -> dict:
    start = time.perf_counter()
    model, history, best_epoch = run_experiment(config, prepared, seed)
    seconds = time.perf_counter() - start
    return {
        "framework": "pytorch",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mae": history[best_epoch - 1]["val_mae"],
        "test_mae": scores(prepared.y_test, predict(model, prepared.x_test))["mae"],
        "seconds": seconds,
        "model_file_kb": saved_size_kb(
            lambda path: torch.save(model.state_dict(), path), ".pt"
        ),
    }


def run_keras(config: RunConfig, prepared: Prepared, seed: int) -> dict:
    start = time.perf_counter()
    model, history, best_epoch = train_keras_run(config, prepared, seed)
    seconds = time.perf_counter() - start
    return {
        "framework": "keras",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mae": history[best_epoch - 1]["val_mae"],
        "test_mae": scores(prepared.y_test, km.predict_keras(model, prepared.x_test))[
            "mae"
        ],
        "seconds": seconds,
        "model_file_kb": saved_size_kb(model.save, ".keras"),
    }


def compare(
    prepared: Prepared, config: RunConfig = MAIN, seeds=SEEDS, verbose: bool = True
) -> pd.DataFrame:
    """One row per run. The frameworks take turns, seed by seed."""
    rows = []
    for seed in seeds:
        for run in (run_pytorch, run_keras):
            row = run(config, prepared, seed)
            rows.append(row)
            if verbose:
                print(
                    f"{row['framework']:>8}  seed {seed}  best epoch "
                    f"{row['best_epoch']:>3}  val MAE {row['val_mae']:.3f}  "
                    f"test MAE {row['test_mae']:.3f}  {row['seconds']:.1f} s"
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over the seeds, per framework."""
    numbers = ["best_epoch", "val_mae", "test_mae", "seconds", "model_file_kb"]
    return table.groupby("framework", sort=False)[numbers].agg(["mean", "std"])


def main():
    REPORTS.mkdir(exist_ok=True)
    prepared = prepare(make_rentals())
    table = compare(prepared)
    table.to_csv(REPORTS / "comparison.csv", index=False)
    print()
    print(table.round(3).to_string(index=False))
    print()
    print(summarize(table).round(3).to_string())


if __name__ == "__main__":
    main()
