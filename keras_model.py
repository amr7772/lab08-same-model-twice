"""The Keras network, built to match RentalNet, and the weight copy between them.

Keras 3 is told to use the TensorFlow backend before it is imported. The
network has the same layers as model.RentalNet: hidden Dense layers with ReLU
and one output with no activation, because the target is a count.
"""

import os

# PyTorch's compiler (torch._dynamo, loaded the first time an optimizer is
# built) and TensorFlow both bundle LLVM. On Linux, if TensorFlow is loaded
# first, building a PyTorch optimizer later aborts with "Option ... registered
# more than once". Loading torch._dynamo before Keras avoids it, so every
# script imports Keras through this module.
import torch
import torch._dynamo  # noqa: F401

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from torch import nn  # noqa: E402

from model import RentalNet  # noqa: E402

if keras.backend.backend() != "tensorflow":
    raise RuntimeError(
        f"Keras is using the {keras.backend.backend()} backend. This project uses "
        "TensorFlow: set KERAS_BACKEND=tensorflow."
    )

# Largest gap allowed between the two networks' predictions, in rentals.
EQUIVALENCE_TOLERANCE = 1e-4


def build_keras_model(
    n_features: int, hidden_size: int = 64, n_hidden: int = 2
) -> keras.Sequential:
    """Same architecture as RentalNet, with the same configurable sizes."""
    if n_hidden < 2:
        raise ValueError("the network needs at least two hidden layers")
    layers: list = [keras.Input(shape=(n_features,))]
    for index in range(n_hidden):
        layers.append(
            keras.layers.Dense(
                hidden_size, activation="relu", name=f"hidden_{index + 1}"
            )
        )
    layers.append(keras.layers.Dense(1, name="output"))
    return keras.Sequential(layers, name="rental_net")


def count_torch_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def torch_linear_layers(model: RentalNet) -> list[nn.Linear]:
    """The Linear layers of a RentalNet, input side first, output last."""
    return [module for module in model.modules() if isinstance(module, nn.Linear)]


def keras_dense_layers(model: keras.Model) -> list:
    return [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]


def copy_torch_weights(torch_model: RentalNet, keras_model: keras.Model) -> None:
    """Copy every PyTorch Linear layer into the matching Keras Dense layer.

    A PyTorch weight is stored as (outputs, inputs) and a Keras kernel as
    (inputs, outputs), so each weight is transposed on the way across. The
    bias vectors have the same shape in both frameworks.
    """
    linears = torch_linear_layers(torch_model)
    dense = keras_dense_layers(keras_model)
    if len(linears) != len(dense):
        raise ValueError(
            f"PyTorch has {len(linears)} Linear layers, Keras has {len(dense)}"
        )
    for linear, layer in zip(linears, dense, strict=True):
        kernel = linear.weight.detach().cpu().numpy().T
        bias = linear.bias.detach().cpu().numpy()
        layer.set_weights([kernel, bias])


def predict_torch(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(features, dtype=torch.float32)).numpy()


def predict_keras(model: keras.Model, features: np.ndarray) -> np.ndarray:
    """One value per row. model.predict returns (N, 1), so flatten it to (N,)."""
    return model.predict(features, verbose=0).reshape(-1)


def max_prediction_gap(
    torch_model: nn.Module, keras_model: keras.Model, features: np.ndarray
) -> float:
    """Largest absolute difference between the two networks on the same rows."""
    gap = predict_torch(torch_model, features) - predict_keras(keras_model, features)
    return float(np.max(np.abs(gap)))


def equivalence_check(
    torch_model: RentalNet,
    features: np.ndarray,
    tolerance: float = EQUIVALENCE_TOLERANCE,
) -> float:
    """Copy a PyTorch network into a fresh Keras network and compare them.

    Returns the largest gap on the given rows. Raises AssertionError if the
    two networks disagree by more than the tolerance.
    """
    first = torch_linear_layers(torch_model)[0]
    keras_model = build_keras_model(
        first.in_features,
        hidden_size=first.out_features,
        n_hidden=len(torch_linear_layers(torch_model)) - 1,
    )
    copy_torch_weights(torch_model, keras_model)
    gap = max_prediction_gap(torch_model, keras_model, features)
    if gap > tolerance:
        raise AssertionError(f"the networks differ by {gap:.2e} > {tolerance:.0e}")
    return gap
