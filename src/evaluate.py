"""
evaluate.py
Evaluation routines for ship maneuvering prediction models.
Implements recursive forward Euler time-simulation and RMSE metrics.
Reference:
  "Identification modeling of ship maneuvering motion based on physics-informed neural network"
  (Hu et al., 2026, Section 4.4, Equations 6, 28, Table 3)
"""

import os
import sys
import json
import torch

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.ship_dynamics import euler_step, SHIP_PARTICULARS


def compute_rmse(pred, true):
    """
    Computes Root Mean Square Error (RMSE).
    Reference: Page 9, Equation (28).
    """
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def predict_trajectory_nn(model, rudder_history, initial_state, stats, dt=1.0, L=SHIP_PARTICULARS['L']):
    """
    Simulates recursive forward trajectory using neural network predictions.
    Reference: Page 4, Equation (6) & Page 9, Section 4.4.

    Parameters:
        model: Trained PyTorch ShipMotionNN
        rudder_history: Array of rudder angles [rad]
        initial_state: (u0, v0, r0, psi0, x0, y0)
        stats: Normalization statistics dictionary
        dt: Time step [s]
        L: Ship length [m]

    Returns:
        Dictionary with predicted trajectories ('time', 'u', 'v', 'r', 'psi', 'x', 'y')
    """
    model.eval()
    n_steps = len(rudder_history)

    in_mean = np.array([stats['inputs'][c]['mean'] for c in ['u', 'v', 'r', 'rudder_angle']], dtype=np.float32)
    in_std = np.array([stats['inputs'][c]['std'] for c in ['u', 'v', 'r', 'rudder_angle']], dtype=np.float32)
    tgt_mean = np.array([stats['targets'][c]['mean'] for c in ['du_dt', 'dv_dt', 'dr_dt']], dtype=np.float32)
    tgt_std = np.array([stats['targets'][c]['std'] for c in ['du_dt', 'dv_dt', 'dr_dt']], dtype=np.float32)

    u, v, r, psi, x, y = initial_state

    hist = {
        'time': np.arange(n_steps) * dt,
        'u': np.zeros(n_steps),
        'v': np.zeros(n_steps),
        'r': np.zeros(n_steps),
        'psi': np.zeros(n_steps),
        'x': np.zeros(n_steps),
        'y': np.zeros(n_steps),
        'du_dt': np.zeros(n_steps),
        'dv_dt': np.zeros(n_steps),
        'dr_dt': np.zeros(n_steps),
        'delta': np.array(rudder_history)
    }

    with torch.no_grad():
        for t in range(n_steps):
            hist['u'][t] = u
            hist['v'][t] = v
            hist['r'][t] = r
            hist['psi'][t] = psi
            hist['x'][t] = x
            hist['y'][t] = y

            delta = rudder_history[t]
            # Standardize inputs
            raw_in = np.array([u, v, r, delta], dtype=np.float32)
            norm_in = (raw_in - in_mean) / in_std
            tensor_in = torch.tensor(norm_in).unsqueeze(0)

            # Neural network prediction
            norm_out = model(tensor_in).squeeze(0).numpy()

            # Destandardize to physical accelerations
            u_dot, v_dot, r_dot = norm_out * tgt_std + tgt_mean
            hist['du_dt'][t] = u_dot
            hist['dv_dt'][t] = v_dot
            hist['dr_dt'][t] = r_dot

            # Forward Euler step
            u, v, r, psi, x, y = euler_step(u, v, r, psi, x, y, u_dot, v_dot, r_dot, dt=dt, L=L)

    return hist


def evaluate_and_plot(model, stats, test_npz_path='outputs/processed/test_data.npz', output_prefix='baseline'):
    """
    Evaluates model across all 3 held-out test maneuvers, plots comparison
    against ground truth simulation, and outputs RMSE metrics.
    """
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)

    test_data = np.load(test_npz_path)
    maneuvers = ['turning_circle_30', 'zigzag_20_10', 'zigzag_25_5']
    metrics = {}

    for m_name in maneuvers:
        u_true = test_data[f'{m_name}_u']
        v_true = test_data[f'{m_name}_v']
        r_true = test_data[f'{m_name}_r']
        delta_true = test_data[f'{m_name}_rudder_angle']

        init_state = (float(u_true[0]), float(v_true[0]), float(r_true[0]), 0.0, 0.0, 0.0)
        pred = predict_trajectory_nn(model, delta_true, init_state, stats)

        # RMSE calculations (in physical units)
        rmse_u = compute_rmse(pred['u'], u_true)
        rmse_v = compute_rmse(pred['v'], v_true)
        rmse_r_rad = compute_rmse(pred['r'], r_true)
        rmse_r_deg = compute_rmse(pred['r'] * (180.0 / np.pi), r_true * (180.0 / np.pi))

        # Sum of RMSEs (in rad/s and deg/s for transparency with paper Table 3)
        metrics[m_name] = {
            'rmse_u_m_s': rmse_u,
            'rmse_v_m_s': rmse_v,
            'rmse_r_rad_s': rmse_r_rad,
            'rmse_r_deg_s': rmse_r_deg,
            'sum_rmse_deg': rmse_u + rmse_v + rmse_r_deg,
            'sum_rmse_rad': rmse_u + rmse_v + rmse_r_rad
        }

        # Generate comparison plot
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        fig.suptitle(f'{output_prefix.capitalize()} Prediction vs Reference: {m_name}', fontsize=13, fontweight='bold')

        t = pred['time']
        # u
        axes[0].plot(t, u_true, 'k--', label='Simulation data (True)', lw=1.5)
        axes[0].plot(t, pred['u'], 'r-', label=f'{output_prefix.capitalize()} prediction', lw=1.5)
        axes[0].set_ylabel('u [m/s]')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # v
        axes[1].plot(t, v_true, 'k--', label='True', lw=1.5)
        axes[1].plot(t, pred['v'], 'r-', label='Pred', lw=1.5)
        axes[1].set_ylabel('v [m/s]')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # r
        axes[2].plot(t, r_true * (180.0 / np.pi), 'k--', label='True', lw=1.5)
        axes[2].plot(t, pred['r'] * (180.0 / np.pi), 'r-', label='Pred', lw=1.5)
        axes[2].set_ylabel('r [deg/s]')
        axes[2].set_xlabel('time [s]')
        axes[2].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_path = f'outputs/plots/{output_prefix}_eval_{m_name}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()

    # Save metrics JSON
    json_path = f'outputs/metrics/{output_prefix}_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Save metrics Markdown table
    md_path = f'outputs/metrics/{output_prefix}_metrics.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# {output_prefix.capitalize()} Evaluation Metrics (RMSE)\n\n')
        f.write('Evaluated via recursive forward Euler time simulation (Equation 6, $\\Delta t = 1.0$ s).\n\n')
        f.write('| Maneuver | Model | RMSE u [m/s] | RMSE v [m/s] | RMSE r [deg/s] | RMSE r [rad/s] | Sum (with deg/s) |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for m_name, m_data in metrics.items():
            f.write(
                f"| `{m_name}` | **{output_prefix.capitalize()}** | "
                f"{m_data['rmse_u_m_s']:.4f} | {m_data['rmse_v_m_s']:.4f} | "
                f"{m_data['rmse_r_deg_s']:.4f} | {m_data['rmse_r_rad_s']:.6f} | "
                f"{m_data['sum_rmse_deg']:.4f} |\n"
            )

    print(f'Saved metrics: {json_path} and {md_path}')
    return metrics
