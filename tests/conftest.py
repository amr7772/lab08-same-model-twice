import pytest
import torch

from bikes_data import make_rentals
from model import RentalNet
from prepare import make_loaders, prepare


@pytest.fixture(scope="session")
def frame():
    return make_rentals()


@pytest.fixture(scope="session")
def prepared(frame):
    return prepare(frame)


@pytest.fixture
def loaders(prepared):
    return make_loaders(prepared, batch_size=64, seed=0)


@pytest.fixture
def small_batch(prepared):
    """The first 32 training rows as one fixed batch."""
    x = torch.as_tensor(prepared.x_train[:32])
    y = torch.as_tensor(prepared.y_train[:32])
    return x, y


@pytest.fixture
def net(prepared):
    torch.manual_seed(0)
    return RentalNet(prepared.n_features, hidden_size=32, n_hidden=2)
