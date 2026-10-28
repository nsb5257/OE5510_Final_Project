"""
train_pinn.py
Physics-Informed Neural Network (PINN) training routine for ship maneuvering motion.
Implements:
  - Dynamic vs Constant loss weighting schedule (Page 5, Equation 18)
  - Joint optimization of neural network weights and 36 hydrodynamic derivatives
  - Multi-component loss tracking (data loss, total physics loss, individual u/v/r physics residuals)
Reference:
  "Identification modeling of ship maneuvering motion based on physics-informed neural network"
  (Hu et al., 2026, Section 3.3, Section 4.2)
"""

import os
import sys
import time
import json
import argparse

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import ShipPINN
from src.evaluate import evaluate_and_plot


def parse_args():
    parser = argparse.ArgumentParser(description='Train Ship PINN model.')
    parser.add_argument('--iterations', type=int, default=40000,
                        help='Number of training iterations (default: 40000, paper full: 120000)')
    parser.add_argument('--weighting', type=str, default='dynamic', choices=['dynamic', 'constant'],
                        help='Physics loss weighting strategy (default: dynamic)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001, Eq. 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_interval', type=int, default=2000,
                        help='Iterations between console logs')
    parser.add_argument('--smoke', action='store_true',
                        help='Run 50 iterations as a smoke test')
    parser.add_argument('--device', type=str, default=None,
                        help="Device to run on ('cuda', 'cpu', or None for auto-detection)")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_beta(iteration, strategy='dynamic'):
    """
    Computes physics loss weight beta according to the schedule on Page 5:
    - Dynamic:
        iter <= 10000: beta = 0.01
        10000 < iter <= 50000: beta = 0.1
        iter > 50000: beta = 1.0
    - Constant:
        beta = 1.0 throughout (Page 7 ablation)
    """
    if strategy == 'constant':
        return 1.0
    if iteration <= 10000:
        return 0.01
    elif iteration <= 50000:
        return 0.1
    else:
        return 1.0


def get_lr(initial_lr, iteration):
    """
    Equation (8): lr = initial_lr * (0.1)**(iteration // 40000)
    """
    decay_step = iteration // 40000
    return initial_lr * (0.1 ** decay_step)


def train_pinn(iterations=40000, weighting='dynamic', initial_lr=0.001, seed=42, log_interval=2000, smoke=False, device=None):
    if smoke:
        iterations = 50
        log_interval = 10

    set_seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device}')
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)

    # 1. Load data
    data = np.load('outputs/processed/train_data.npz')
    with open('outputs/processed/normalization_stats.json', 'r') as f:
        stats = json.load(f)

    X_train_norm = torch.tensor(data['inputs_norm'], dtype=torch.float32, device=device)
    Y_train_norm = torch.tensor(data['targets_norm'], dtype=torch.float32, device=device)
    X_train_raw = torch.tensor(data['inputs_raw'], dtype=torch.float32, device=device)

    tgt_mean = torch.tensor(data['tgt_mean'], dtype=torch.float32, device=device)
    tgt_std = torch.tensor(data['tgt_std'], dtype=torch.float32, device=device)

    # 2. Instantiate PINN model
    model = ShipPINN().to(device)
    # Optimizer optimizes both neural network parameters AND 36 hydrodynamic derivatives jointly
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Precompute invariant basis matrices for training set (exact O(1) evaluation)
    basis = model.precompute_basis(X_train_raw)

    history = {
        'iteration': [],
        'total_loss': [],
        'loss_data': [],
        'loss_phy': [],
        'loss_phy_u': [],
        'loss_phy_v': [],
        'loss_phy_r': [],
        'beta': [],
        'lr': [],
        # Track representative hydrodynamic derivatives
        'param_Xu_prime': [],
        'param_Yv_prime': [],
        'param_Nr_prime': []
    }

    print(f'Starting PINN Training: {iterations} iterations, weighting={weighting}, seed={seed}')
    start_time = time.time()

    for step in range(1, iterations + 1):
        # Update learning rate (Eq. 8)
        current_lr = get_lr(initial_lr, step - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Update physics weight beta
        beta = get_beta(step - 1, strategy=weighting)

        optimizer.zero_grad()
        loss_dict = model.compute_loss(
            inputs_norm=X_train_norm,
            targets_norm=Y_train_norm,
            inputs_raw=X_train_raw,
            tgt_mean=tgt_mean,
            tgt_std=tgt_std,
            beta=beta,
            basis=basis
        )

        loss_dict['total_loss'].backward()
        optimizer.step()

        if step % log_interval == 0 or step == 1 or step == iterations:
            elapsed = time.time() - start_time
            print(
                f"Iter {step:6d}/{iterations} | "
                f"Total: {loss_dict['total_loss'].item():.5e} | "
                f"Data: {loss_dict['loss_data'].item():.5e} | "
                f"Phy: {loss_dict['loss_phy'].item():.5e} | "
                f"beta: {beta:.2f} | "
                f"Xu': {model.hydro_params['Xu_prime'].item():.5f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

            history['iteration'].append(step)
            history['total_loss'].append(loss_dict['total_loss'].item())
            history['loss_data'].append(loss_dict['loss_data'].item())
            history['loss_phy'].append(loss_dict['loss_phy'].item())
            history['loss_phy_u'].append(loss_dict['loss_phy_u'].item())
            history['loss_phy_v'].append(loss_dict['loss_phy_v'].item())
            history['loss_phy_r'].append(loss_dict['loss_phy_r'].item())
            history['beta'].append(beta)
            history['lr'].append(current_lr)
            history['param_Xu_prime'].append(model.hydro_params['Xu_prime'].item())
            history['param_Yv_prime'].append(model.hydro_params['Yv_prime'].item())
            history['param_Nr_prime'].append(model.hydro_params['Nr_prime'].item())

    total_time = time.time() - start_time
    print(f'PINN training completed in {total_time:.2f}s ({total_time/60:.2f} min).')

    # Save checkpoint
    ckpt_prefix = f'pinn_{weighting}' if not smoke else f'pinn_{weighting}_smoke'
    ckpt_path = f'outputs/checkpoints/{ckpt_prefix}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'hydro_params': {k: v.item() for k, v in model.hydro_params.items()},
        'iterations': iterations,
        'weighting': weighting,
        'history': history,
        'training_time_s': total_time
    }, ckpt_path)
    print(f'Saved checkpoint: {ckpt_path}')

    # Save loss history curves
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(f'PINN Training Loss Curves ({weighting.capitalize()} Weighting)', fontsize=13, fontweight='bold')

    axes[0].semilogy(history['iteration'], history['total_loss'], 'k-', label='Total Loss', lw=1.5)
    axes[0].semilogy(history['iteration'], history['loss_data'], 'r--', label='Data Loss (MSE)', lw=1.5)
    axes[0].semilogy(history['iteration'], history['loss_phy'], 'b-.', label='Physics Loss (Residuals)', lw=1.5)
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].semilogy(history['iteration'], history['loss_phy_u'], label='Physics Residual e1 (u)', lw=1.2)
    axes[1].semilogy(history['iteration'], history['loss_phy_v'], label='Physics Residual e2 (v)', lw=1.2)
    axes[1].semilogy(history['iteration'], history['loss_phy_r'], label='Physics Residual e3 (r)', lw=1.2)
    axes[1].set_ylabel('Component Physics Loss')
    axes[1].set_xlabel('Iteration')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    curve_path = f'outputs/plots/{ckpt_prefix}_loss_curves.png'
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f'Saved loss curves: {curve_path}')

    if not smoke:
        # Evaluate on test maneuvers
        print(f'\nEvaluating {ckpt_prefix} on held-out test maneuvers...')
        evaluate_and_plot(model.nn, stats, output_prefix=ckpt_prefix)

    return model, history


def main():
    args = parse_args()
    train_pinn(
        iterations=args.iterations,
        weighting=args.weighting,
        initial_lr=args.lr,
        seed=args.seed,
        log_interval=args.log_interval,
        smoke=args.smoke,
        device=args.device
    )


if __name__ == '__main__':
    main()
