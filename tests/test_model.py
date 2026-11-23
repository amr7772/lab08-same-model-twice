import pytest
import torch
from torch import nn

from model import RentalNet


@pytest.mark.parametrize("hidden_size", [4, 16, 64, 256])
def test_network_returns_one_value_per_row(prepared, hidden_size):
    net = RentalNet(prepared.n_features, hidden_size=hidden_size)
    x = torch.as_tensor(prepared.x_train[:10])
    assert net(x).shape == (10,)


def test_network_has_two_hidden_layers_and_no_output_activation(prepared):
    net = RentalNet(prepared.n_features, hidden_size=8)
    assert sum(isinstance(m, nn.ReLU) for m in net.modules()) == 2
    assert isinstance(net.output, nn.Linear)


def test_one_optimizer_step_changes_every_layer(net, small_batch):
    x, y = small_batch
    before = {k: v.clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss = nn.functional.mse_loss(net(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for name, tensor in net.state_dict().items():
        assert not torch.equal(before[name], tensor), f"{name} did not change"
