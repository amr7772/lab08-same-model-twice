import copy

import pytest
import torch
from torch import nn

from model import RentalNet
from train import fit, run_epoch

LOGGED = {"epoch", "train_loss", "val_loss", "val_mae"}


def memorize(net, batch, steps=300, clear_gradients=True, step=True):
    """A tiny training loop, with two switches to break it on purpose."""
    x, y = batch
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    first = loss_fn(net(x), y).item()
    for _ in range(steps):
        loss = loss_fn(net(x), y)
        if clear_gradients:
            optimizer.zero_grad()
        loss.backward()
        if step:
            optimizer.step()
    return first, loss_fn(net(x), y).item()


def test_network_can_memorize_one_small_batch(net, small_batch):
    first, last = memorize(net, small_batch)
    assert last < 0.05 * first


def test_memorize_check_fails_without_an_optimizer_step(net, small_batch):
    # The weights never move, so the loss cannot fall. The test above would
    # catch a loop that forgot optimizer.step().
    first, last = memorize(net, small_batch, step=False)
    assert last == pytest.approx(first)


def test_memorize_check_notices_gradients_that_are_never_cleared(net, small_batch):
    # Gradients pile up from step to step. The loop still "runs", but it
    # does not do as well as the correct one.
    good_net = copy.deepcopy(net)
    _, good = memorize(good_net, small_batch)
    _, bad = memorize(net, small_batch, clear_gradients=False)
    assert bad > good


def test_measuring_an_epoch_leaves_the_weights_unchanged(net, loaders):
    _, val_loader, _ = loaders
    before = copy.deepcopy(net.state_dict())
    run_epoch(net, val_loader, nn.MSELoss())
    for name, tensor in net.state_dict().items():
        assert torch.equal(before[name], tensor)
    assert all(p.grad is None for p in net.parameters())


def test_training_epoch_changes_the_weights_and_returns_loss_and_mae(net, loaders):
    train_loader, _, _ = loaders
    before = copy.deepcopy(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss, mae = run_epoch(net, train_loader, nn.MSELoss(), optimizer)
    assert loss > 0 and mae > 0
    assert any(not torch.equal(before[k], v) for k, v in net.state_dict().items())


def test_fit_logs_every_epoch_and_restores_the_best_weights(prepared, loaders):
    train_loader, val_loader, _ = loaders
    torch.manual_seed(0)
    net = RentalNet(prepared.n_features, hidden_size=16)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    history, best_epoch = fit(net, train_loader, val_loader, loss_fn, optimizer, 6)
    assert len(history) == 6
    assert all(set(row) == LOGGED for row in history)
    assert [row["epoch"] for row in history] == list(range(1, 7))
    val_losses = [row["val_loss"] for row in history]
    assert best_epoch == val_losses.index(min(val_losses)) + 1
    restored_loss, _ = run_epoch(net, val_loader, loss_fn)
    assert restored_loss == pytest.approx(min(val_losses), rel=1e-4)
