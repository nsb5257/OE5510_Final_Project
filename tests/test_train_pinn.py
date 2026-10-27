"""
test_train_pinn.py
Unit tests and acceptance checks for Task T5:
- PINN architecture and loss formulation
- Separation of data loss and physical residuals
- Backpropagation and gradient validity across network weights and 36 hydrodynamic derivatives
- Dynamic and constant loss weighting schedules
"""

import os
import torch
import numpy as np
import pytest
from src.model import ShipPINN
from src.train_pinn import get_beta, get_lr, train_pinn
from src.ship_dynamics import ORIGINAL_HYDRODYNAMIC_DERIVATIVES


def test_hydro_parameters_count_and_initialization():
    """
    Page 6, Section 4.2 & Page 7, Table 2:
    Verify exactly 36 hydrodynamic derivatives, all initialized to 0.0.
    """
    model = ShipPINN()
    assert len(model.hydro_params) == 36, f"Expected 36 parameters, got {len(model.hydro_params)}"
    assert set(model.hydro_params.keys()) == set(ORIGINAL_HYDRODYNAMIC_DERIVATIVES.keys())

    for name, param in model.hydro_params.items():
        assert param.requires_grad, f"Parameter {name} must require grad"
        assert param.item() == 0.0, f"Parameter {name} should initialize to 0.0, got {param.item()}"


def test_loss_components_separately_visible_on_tiny_batch():
    """
    Acceptance Criterion:
    Loss components are separately visible and finite on a tiny batch.
    """
    model = ShipPINN()
    data = np.load('outputs/processed/train_data.npz')

    # Tiny batch of 4 samples
    inputs_norm = torch.tensor(data['inputs_norm'][:4], dtype=torch.float32)
    targets_norm = torch.tensor(data['targets_norm'][:4], dtype=torch.float32)
    inputs_raw = torch.tensor(data['inputs_raw'][:4], dtype=torch.float32)

    tgt_mean = torch.tensor(data['tgt_mean'], dtype=torch.float32)
    tgt_std = torch.tensor(data['tgt_std'], dtype=torch.float32)

    loss_dict = model.compute_loss(
        inputs_norm=inputs_norm,
        targets_norm=targets_norm,
        inputs_raw=inputs_raw,
        tgt_mean=tgt_mean,
        tgt_std=tgt_std,
        beta=0.01
    )

    required_keys = ['total_loss', 'loss_data', 'loss_phy', 'loss_phy_u', 'loss_phy_v', 'loss_phy_r']
    for k in required_keys:
        assert k in loss_dict, f"Missing loss component: {k}"
        val = loss_dict[k].item()
        assert np.isfinite(val), f"Loss {k} is non-finite: {val}"
        assert val >= 0.0, f"Loss {k} must be non-negative: {val}"

    # Verify additive decomposition
    expected_phy = loss_dict['loss_phy_u'] + loss_dict['loss_phy_v'] + loss_dict['loss_phy_r']
    assert torch.isclose(loss_dict['loss_phy'], expected_phy)

    expected_total = loss_dict['loss_data'] + 0.01 * loss_dict['loss_phy']
    assert torch.isclose(loss_dict['total_loss'], expected_total)


def test_backward_pass_produces_valid_gradients():
    """
    Acceptance Criterion:
    A backward pass produces valid gradients across both NN weights and hydrodynamic derivatives.
    """
    model = ShipPINN()
    data = np.load('outputs/processed/train_data.npz')

    inputs_norm = torch.tensor(data['inputs_norm'][:8], dtype=torch.float32)
    targets_norm = torch.tensor(data['targets_norm'][:8], dtype=torch.float32)
    inputs_raw = torch.tensor(data['inputs_raw'][:8], dtype=torch.float32)

    tgt_mean = torch.tensor(data['tgt_mean'], dtype=torch.float32)
    tgt_std = torch.tensor(data['tgt_std'], dtype=torch.float32)

    loss_dict = model.compute_loss(
        inputs_norm=inputs_norm,
        targets_norm=targets_norm,
        inputs_raw=inputs_raw,
        tgt_mean=tgt_mean,
        tgt_std=tgt_std,
        beta=0.1
    )

    loss_dict['total_loss'].backward()

    # 1. Check neural network parameters
    for name, param in model.nn.named_parameters():
        assert param.grad is not None, f"Missing gradient in NN {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in NN {name}"
        assert torch.any(param.grad != 0.0), f"Zero gradient in NN {name}"

    # 2. Check hydrodynamic derivatives
    grad_count = 0
    for name, param in model.hydro_params.items():
        assert param.grad is not None, f"Missing gradient in hydro param {name}"
        assert torch.isfinite(param.grad), f"Non-finite gradient in hydro param {name}"
        if param.grad != 0.0:
            grad_count += 1

    assert grad_count > 0, "At least some hydrodynamic parameters must receive non-zero gradients"


def test_beta_and_lr_schedules():
    """
    Page 4, Equation (8) and Page 5, Equation (18):
    Verify dynamic/constant beta and learning rate schedules.
    """
    # Dynamic beta
    assert get_beta(0, 'dynamic') == 0.01
    assert get_beta(10000, 'dynamic') == 0.01
    assert get_beta(10001, 'dynamic') == 0.1
    assert get_beta(50000, 'dynamic') == 0.1
    assert get_beta(50001, 'dynamic') == 1.0
    assert get_beta(120000, 'dynamic') == 1.0

    # Constant beta ablation
    assert get_beta(0, 'constant') == 1.0
    assert get_beta(50000, 'constant') == 1.0
    assert get_beta(120000, 'constant') == 1.0

    # Learning rate schedule (Eq. 8)
    assert np.isclose(get_lr(0.001, 0), 1e-3)
    assert np.isclose(get_lr(0.001, 39999), 1e-3)
    assert np.isclose(get_lr(0.001, 40000), 1e-4)
    assert np.isclose(get_lr(0.001, 79999), 1e-4)
    assert np.isclose(get_lr(0.001, 80000), 1e-5)


def test_pinn_smoke_training_loop():
    """
    Smoke test: run 20 iterations through train_pinn and verify history logging.
    """
    model, history = train_pinn(iterations=20, weighting='dynamic', initial_lr=0.001, smoke=True)
    assert len(history['iteration']) > 0
    assert len(history['total_loss']) > 0
    assert len(history['loss_phy']) > 0
    assert np.all(np.isfinite(history['total_loss']))
