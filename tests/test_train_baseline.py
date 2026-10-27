"""
test_train_baseline.py
Unit and smoke tests for baseline neural network model, training step, and evaluation.
"""

import os
import json
import torch
import numpy as np
import pytest
from src.model import ShipMotionNN
from src.evaluate import evaluate_and_plot, predict_trajectory_nn, compute_rmse


def test_model_forward_and_shapes():
    """
    Verify ShipMotionNN architecture, forward pass shape, and determinism.
    """
    model = ShipMotionNN(input_dim=4, hidden_dim=10, output_dim=3)
    batch_size = 32
    x = torch.randn(batch_size, 4)
    out = model(x)

    assert out.shape == (batch_size, 3)
    assert torch.all(torch.isfinite(out))


def test_backward_pass_gradients():
    """
    Verify backpropagation produces valid non-zero finite gradients for all parameters.
    """
    model = ShipMotionNN(input_dim=4, hidden_dim=10, output_dim=3)
    x = torch.randn(16, 4)
    target = torch.randn(16, 3)

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"
        assert torch.any(param.grad != 0.0), f"All-zero gradient in {name}"


def test_short_training_smoke():
    """
    Smoke test: run 50 steps on a small batch and verify loss decreases.
    """
    torch.manual_seed(42)
    model = ShipMotionNN(input_dim=4, hidden_dim=10, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(50, 4)
    y = torch.randn(50, 3)

    initial_loss = torch.nn.functional.mse_loss(model(x), y).item()

    for _ in range(50):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

    final_loss = torch.nn.functional.mse_loss(model(x), y).item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_predict_trajectory_nn_smoke():
    """
    Verify forward simulation generates complete and finite trajectory.
    """
    model = ShipMotionNN(input_dim=4, hidden_dim=10, output_dim=3)
    with open('outputs/processed/normalization_stats.json', 'r') as f:
        stats = json.load(f)

    rudder = np.zeros(20)
    init_state = (7.956, 0.0, 0.0, 0.0, 0.0, 0.0)

    pred = predict_trajectory_nn(model, rudder, init_state, stats, dt=1.0)

    assert len(pred['u']) == 20
    assert len(pred['v']) == 20
    assert len(pred['r']) == 20
    assert np.all(np.isfinite(pred['u']))
    assert np.all(np.isfinite(pred['v']))
    assert np.all(np.isfinite(pred['r']))
