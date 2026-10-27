"""
test_ship_dynamics.py
Unit tests and numerical verification for 3-DOF ship dynamics (Abkowitz model).
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.ship_dynamics import (
    SHIP_PARTICULARS,
    ORIGINAL_HYDRODYNAMIC_DERIVATIVES,
    compute_hydrodynamic_forces,
    compute_accelerations,
    euler_step,
    simulate_trajectory
)


def test_straight_line_equilibrium():
    """
    Page 4, Equation (4):
    In steady straight-line motion (u = U0, v = 0, r = 0, delta = 0),
    all hydrodynamic forces and accelerations should be exactly zero.
    """
    U0 = SHIP_PARTICULARS['U0']
    f1, f2, f3 = compute_hydrodynamic_forces(u=U0, v=0.0, r=0.0, delta=0.0)
    assert np.isclose(f1, 0.0, atol=1e-12)
    assert np.isclose(f2, 0.0, atol=1e-12)
    assert np.isclose(f3, 0.0, atol=1e-12)

    u_dot, v_dot, r_dot = compute_accelerations(u=U0, v=0.0, r=0.0, delta=0.0)
    assert np.isclose(u_dot, 0.0, atol=1e-12)
    assert np.isclose(v_dot, 0.0, atol=1e-12)
    assert np.isclose(r_dot, 0.0, atol=1e-12)


def test_hand_worked_numerical_example():
    """
    Hand-calculated analytical check against row 1 of turning_circle_35.txt:
    u = 7.956 m/s, v = 0 m/s, r = 0 rad/s, delta = -0.04084070449666731 rad.
    """
    u = 7.956
    v = 0.0
    r = 0.0
    delta = -0.04084070449666731

    # Hand calculation:
    # f1_prime = Xdd * delta^2 = -0.00272 * (-0.0408407)^2 = -4.53686e-6
    # u_dot_prime = f1_prime / (m_prime - X_udot_prime) = -4.53686e-6 / (0.019080625 - (-0.00114)) = -2.24368e-4
    # u_dot = u_dot_prime * (7.956^2) / 320 = -4.43813e-5 m/s^2
    u_dot, v_dot, r_dot = compute_accelerations(u, v, r, delta)

    # Reference values from data/train_data/turning_circle_35.txt row 1
    expected_u_dot = -4.438133375028741e-05
    expected_v_dot = -1.5094011667418155e-03
    expected_r_dot = 2.9611124293063036e-05

    # Compare with reference values from data/train_data/turning_circle_35.txt row 1
    # Agreement within 1e-5 atol reflects Table 2 printed precision (4-5 decimals)
    assert np.isclose(u_dot, expected_u_dot, rtol=1e-3, atol=1e-5)
    assert np.isclose(v_dot, expected_v_dot, rtol=1e-3, atol=1e-5)
    assert np.isclose(r_dot, expected_r_dot, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize('train_file', [
    'turning_circle_35.txt',
    'zigzag_10_10.txt',
    'zigzag_20_20.txt'
])
def test_dataset_acceleration_accuracy(train_file):
    """
    Verify that compute_accelerations reproduces raw simulation dataset
    accelerations across all rows with numerical error < 1e-5.
    """
    fpath = os.path.join('data', 'train_data', train_file)
    df = pd.read_csv(fpath, sep=r'\s+')

    u = df['u'].to_numpy()
    v = df['v'].to_numpy()
    r = df['r'].to_numpy()
    delta = df['rudder_angle'].to_numpy()

    du_calc, dv_calc, dr_calc = compute_accelerations(u, v, r, delta)

    du_err = np.max(np.abs(du_calc - df['du_dt'].to_numpy()))
    dv_err = np.max(np.abs(dv_calc - df['dv_dt'].to_numpy()))
    dr_err = np.max(np.abs(dr_calc - df['dr_dt'].to_numpy()))

    assert du_err < 1e-5, f"Surge acceleration error {du_err:.2e} exceeded threshold in {train_file}"
    assert dv_err < 1e-5, f"Sway acceleration error {dv_err:.2e} exceeded threshold in {train_file}"
    assert dr_err < 1e-5, f"Yaw acceleration error {dr_err:.2e} exceeded threshold in {train_file}"


def test_vectorization_and_finite_outputs():
    """
    Verify that compute_accelerations handles arrays and produces all finite values.
    """
    N = 50
    u = np.linspace(4.0, 8.0, N)
    v = np.linspace(-1.5, 1.5, N)
    r = np.linspace(-0.01, 0.01, N)
    delta = np.linspace(-0.6, 0.6, N)

    u_dot, v_dot, r_dot = compute_accelerations(u, v, r, delta)

    assert u_dot.shape == (N,)
    assert v_dot.shape == (N,)
    assert r_dot.shape == (N,)

    assert np.all(np.isfinite(u_dot))
    assert np.all(np.isfinite(v_dot))
    assert np.all(np.isfinite(r_dot))


def test_euler_step_integration():
    """
    Page 4, Equation (6):
    Verify forward Euler state updates.
    """
    u, v, r, psi, x, y = 7.0, -0.5, 0.005, 0.1, 1.0, 2.0
    u_dot, v_dot, r_dot = -0.01, 0.02, 0.0001
    dt = 1.0
    L = 320.0

    u_n, v_n, r_n, psi_n, x_n, y_n = euler_step(u, v, r, psi, x, y, u_dot, v_dot, r_dot, dt=dt, L=L)

    assert np.isclose(u_n, 7.0 - 0.01)
    assert np.isclose(v_n, -0.5 + 0.02)
    assert np.isclose(r_n, 0.005 + 0.0001)
    assert np.isclose(psi_n, 0.1 + 0.005)

    expected_dx = (7.0 * np.cos(0.1) - (-0.5) * np.sin(0.1)) * 1.0 / 320.0
    expected_dy = (7.0 * np.sin(0.1) + (-0.5) * np.cos(0.1)) * 1.0 / 320.0

    assert np.isclose(x_n, 1.0 + expected_dx)
    assert np.isclose(y_n, 2.0 + expected_dy)


def test_simulate_trajectory_execution():
    """
    Verify full trajectory simulation generates valid and finite data series.
    """
    rudder = np.full(100, -0.1)
    traj = simulate_trajectory(rudder)

    assert len(traj['time']) == 100
    assert len(traj['u']) == 100
    assert len(traj['v']) == 100
    assert len(traj['r']) == 100
    assert len(traj['psi']) == 100
    assert len(traj['x']) == 100
    assert len(traj['y']) == 100

    assert np.all(np.isfinite(traj['u']))
    assert np.all(np.isfinite(traj['v']))
    assert np.all(np.isfinite(traj['r']))
    assert np.all(np.isfinite(traj['x']))
    assert np.all(np.isfinite(traj['y']))
