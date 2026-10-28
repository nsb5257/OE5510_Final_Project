"""
train_baseline.py
Train and evaluate a standard data-only feed-forward neural network baseline.
Reference:
  "Identification modeling of ship maneuvering motion based on physics-informed neural network"
  (Hu et al., 2026, Section 3.1, Section 5.3)
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
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import ShipMotionNN
from src.evaluate import evaluate_and_plot


def parse_args():
    parser = argparse.ArgumentParser(description='Train data-only baseline neural network.')
    parser.add_argument('--iterations', type=int, default=40000,
                        help='Number of training iterations (default: 40000, paper full: 120000)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001, Eq. 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_interval', type=int, default=2000,
                        help='Iterations between console logs')
    parser.add_argument('--device', type=str, default=None,
                        help="Device to run on ('cuda', 'cpu', or None for auto-detection)")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(initial_lr, iteration):
    """
    Equation (8): lr = initial_lr * (0.1)**(iteration // 40000)
    """
    decay_step = iteration // 40000
    return initial_lr * (0.1 ** decay_step)


def train_baseline(iterations=40000, initial_lr=0.001, seed=42, log_interval=2000, device=None):
    set_seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device}')
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    # 1. Load prepared data and normalization stats
    data = np.load('outputs/processed/train_data.npz')
    with open('outputs/processed/normalization_stats.json', 'r') as f:
        stats = json.load(f)

    X_train = torch.tensor(data['inputs_norm'], dtype=torch.float32, device=device)
    Y_train = torch.tensor(data['targets_norm'], dtype=torch.float32, device=device)

    n_samples = X_train.shape[0]
    print(f'Training Baseline NN: {n_samples} samples, {iterations} iterations, seed={seed}')

    # 2. Instantiate model and optimizer
    model = ShipMotionNN(input_dim=4, hidden_dim=10, output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    loss_fn = nn.MSELoss()

    history = {'iteration': [], 'loss': [], 'lr': []}
    start_time = time.time()

    # 3. Training loop
    for step in range(1, iterations + 1):
        # Update learning rate per schedule (Eq. 8)
        current_lr = get_lr(initial_lr, step - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_fn(predictions, Y_train)
        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or step == 1 or step == iterations:
            elapsed = time.time() - start_time
            print(f'Iter {step:6d}/{iterations} | Loss: {loss.item():.6e} | lr: {current_lr:.1e} | Elapsed: {elapsed:.1f}s')
            history['iteration'].append(step)
            history['loss'].append(loss.item())
            history['lr'].append(current_lr)

    total_time = time.time() - start_time
    print(f'Training complete in {total_time:.2f}s ({total_time/60:.2f} min). Final Loss: {loss.item():.6e}')

    # 4. Save model checkpoint
    checkpoint_path = 'outputs/checkpoints/baseline_nn.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'iterations': iterations,
        'final_loss': loss.item(),
        'seed': seed,
        'training_time_s': total_time
    }, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

    # 5. Save training history and loss plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history['iteration'], history['loss'], 'b-', lw=1.5, label='Baseline Data Loss (MSE)')
    ax.set_title('Baseline Neural Network Training Convergence', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (Normalized MSE)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    loss_plot_path = 'outputs/plots/baseline_loss_curve.png'
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f'Saved loss curve: {loss_plot_path}')

    # 6. Evaluate on test maneuvers
    print('\nEvaluating baseline on held-out test maneuvers...')
    metrics = evaluate_and_plot(model, stats, output_prefix='baseline')
    return model, metrics, history


def main():
    args = parse_args()
    train_baseline(
        iterations=args.iterations,
        initial_lr=args.lr,
        seed=args.seed,
        log_interval=args.log_interval,
        device=args.device
    )


if __name__ == '__main__':
    main()
