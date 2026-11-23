"""The network: a small multilayer perceptron for a count target."""

import torch
from torch import nn


class RentalNet(nn.Module):
    """Fully connected network with ReLU hidden layers and one linear output.

    The target is a count, not a probability, so the last layer has no
    activation. forward returns shape (N,), one value per row, to match the
    shape of the targets.
    """

    def __init__(self, n_features: int, hidden_size: int = 64, n_hidden: int = 2):
        super().__init__()
        if n_hidden < 2:
            raise ValueError("the network needs at least two hidden layers")
        layers: list[nn.Module] = []
        width_in = n_features
        for _ in range(n_hidden):
            layers.append(nn.Linear(width_in, hidden_size))
            layers.append(nn.ReLU())
            width_in = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x)).squeeze(-1)
