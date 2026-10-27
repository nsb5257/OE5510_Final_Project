"""
ship_dynamics.py
Implementation of 3-DOF ship maneuvering motion equations (Abkowitz model).
Referenced to:
  "Identification modeling of ship maneuvering motion based on physics-informed neural network"
  (Hu et al., Ships and Offshore Structures, 2026, DOI: 10.1080/17445302.2026.2648148)
"""

import numpy as np

# ==============================================================================
# 1. Benchmark Ship Particulars: KVLCC2 (Second Variant)
# Reference: Page 6, Table 1 & Page 7 text
# ==============================================================================
SHIP_PARTICULARS = {
    'L': 320.00,        # Length between perpendiculars [m] (Table 1)
    'B': 58.00,         # Breadth [m] (Table 1)
    'd': 20.80,         # Design draft [m] (Table 1)
    'Cb': 0.81,         # Block coefficient [-] (Table 1)
    'U0': 7.956,        # Approach speed [m/s] (Table 1)
    # Prime-I non-dimensional mass properties (Page 7 & notes/paper_spec.md)
    'm_prime': 0.019080625,
    'xG_prime': 0.034636727,  # xG = 11.0838 m (~11.136 m in Table 1)
    'Iz_prime': 0.001198626,  # Iz = 2.0e12 kg*m^2 with rho = 1000 kg/m^3
}

# 5 Fixed Acceleration Derivatives (Page 7, left column)
# These derivatives are considered known and not identified by PINN.
FIXED_ACCELERATION_DERIVATIVES = {
    'X_udot_prime': -0.00114,
    'Y_vdot_prime': -0.0145,
    'Y_rdot_prime': -0.00121,
    'N_vdot_prime': -0.000588,
    'N_rdot_prime': -0.000564,
}

# 36 Ground-Truth Original Hydrodynamic Derivatives (Page 7, Table 2 "Original")
# From Yao et al. (2021), used to generate simulation data.
ORIGINAL_HYDRODYNAMIC_DERIVATIVES = {
    # 8 Surge coefficients (X-Coef)
    'Xu_prime': -0.00319,
    'Xuu_prime': 0.00199,
    'Xvv_prime': 0.00159,
    'Xrr_prime': 0.000338,
    'Xvr_prime': 0.01391,
    'Xdd_prime': -0.00272,
    'Xvd_prime': 0.001609,
    'Xrd_prime': -0.001034,

    # 14 Sway coefficients (Y-Coef)
    'Yv_prime': -0.01902,
    'Yvvv_prime': -0.1287,
    'Yr_prime': 0.005719,
    'Yrrr_prime': -0.000048,
    'Yvrr_prime': -0.02429,
    'Yvvr_prime': 0.0211,
    'Yd_prime': 0.00408,
    'Ydd_prime': -0.000114,
    'Yddd_prime': -0.003059,
    'Yud_prime': -0.00456,
    'Yvdd_prime': 0.00326,
    'Yvvd_prime': 0.003018,
    'Yrdd_prime': -0.002597,
    'Yrrd_prime': 0.000895,

    # 14 Yaw moment coefficients (N-Coef)
    'Nv_prime': -0.007886,
    'Nvvv_prime': 0.00175,
    'Nr_prime': -0.003701,
    'Nrrr_prime': -0.000707,
    'Nvrr_prime': 0.003726,
    'Nvvr_prime': -0.019,
    'Nd_prime': -0.001834,
    'Ndd_prime': -0.000056,
    'Nddd_prime': 0.001426,
    'Nud_prime': 0.00232,
    'Nvdd_prime': -0.001504,
    'Nvvd_prime': -0.001406,
    'Nrdd_prime': 0.001191,
    'Nrrd_prime': -0.000398,
}


def compute_hydrodynamic_forces(u, v, r, delta, params=None, L=SHIP_PARTICULARS['L'], U0=SHIP_PARTICULARS['U0']):
    """
    Computes non-dimensional hydrodynamic forces (f1', f2', f3').
    Reference: Page 4, Equation (4).

    Parameters:
        u: Surge velocity [m/s] (scalar or array)
        v: Sway velocity [m/s] (scalar or array)
        r: Yaw rate [rad/s] (scalar or array)
        delta: Rudder angle [rad] (scalar or array)
        params: dict of 36 non-dimensional hydrodynamic derivatives.
                Defaults to ORIGINAL_HYDRODYNAMIC_DERIVATIVES.
        L: Ship length [m]
        U0: Approach speed [m/s]

    Returns:
        f1_prime, f2_prime, f3_prime: Non-dimensional forces and moment.
    """
    if params is None:
        params = ORIGINAL_HYDRODYNAMIC_DERIVATIVES

    # Instantaneous surge speed U = u is used for Prime-I normalization
    U = np.maximum(u, 1e-4)  # prevent division by zero

    # Non-dimensional motion variables (Prime-I system)
    u_p = u / U               # u' = 1.0
    v_p = v / U               # v' = v / u
    r_p = r * L / U           # r' = r * L / u
    delta_u_p = (u - U0) / U  # delta_u' = (u - U0) / u
    delta_p = delta           # delta' = delta [rad]

    m_p = SHIP_PARTICULARS['m_prime']
    xG_p = SHIP_PARTICULARS['xG_prime']

    # Page 4, Equation (4) - f1 (Surge force)
    f1_p = (
        params['Xu_prime'] * delta_u_p
        + params['Xuu_prime'] * (delta_u_p ** 2)
        + params['Xvv_prime'] * (v_p ** 2)
        + (m_p * xG_p + params['Xrr_prime']) * (r_p ** 2)
        + (m_p + params['Xvr_prime']) * v_p * r_p
        + params['Xdd_prime'] * (delta_p ** 2)
        + params['Xvd_prime'] * v_p * delta_p
        + params['Xrd_prime'] * r_p * delta_p
    )

    # Page 4, Equation (4) - f2 (Sway force)
    f2_p = (
        -m_p * u_p * r_p
        + params['Yv_prime'] * v_p
        + params['Yvvv_prime'] * (v_p ** 3)
        + params['Yr_prime'] * r_p
        + params['Yrrr_prime'] * (r_p ** 3)
        + params['Yd_prime'] * delta_p
        + params['Ydd_prime'] * (delta_p ** 2)
        + params['Yddd_prime'] * (delta_p ** 3)
        + params['Yvrr_prime'] * v_p * (r_p ** 2)
        + params['Yvvr_prime'] * (v_p ** 2) * r_p
        + params['Yud_prime'] * delta_u_p * delta_p
        + params['Yvdd_prime'] * v_p * (delta_p ** 2)
        + params['Yvvd_prime'] * (v_p ** 2) * delta_p
        + params['Yrdd_prime'] * r_p * (delta_p ** 2)
        + params['Yrrd_prime'] * (r_p ** 2) * delta_p
    )

    # Page 4, Equation (4) - f3 (Yaw moment)
    f3_p = (
        -m_p * xG_p * u_p * r_p
        + params['Nv_prime'] * v_p
        + params['Nvvv_prime'] * (v_p ** 3)
        + params['Nr_prime'] * r_p
        + params['Nrrr_prime'] * (r_p ** 3)
        + params['Nd_prime'] * delta_p
        + params['Ndd_prime'] * (delta_p ** 2)
        + params['Nddd_prime'] * (delta_p ** 3)
        + params['Nvrr_prime'] * v_p * (r_p ** 2)
        + params['Nvvr_prime'] * (v_p ** 2) * r_p
        + params['Nud_prime'] * delta_u_p * delta_p
        + params['Nvdd_prime'] * v_p * (delta_p ** 2)
        + params['Nvvd_prime'] * (v_p ** 2) * delta_p
        + params['Nrdd_prime'] * r_p * (delta_p ** 2)
        + params['Nrrd_prime'] * (r_p ** 2) * delta_p
    )

    return f1_p, f2_p, f3_p


def compute_accelerations(u, v, r, delta, params=None, L=SHIP_PARTICULARS['L'], U0=SHIP_PARTICULARS['U0']):
    """
    Computes dimensional physical accelerations (du/dt, dv/dt, dr/dt).
    Reference: Page 4, Equation (5).

    Parameters:
        u: Surge velocity [m/s]
        v: Sway velocity [m/s]
        r: Yaw rate [rad/s]
        delta: Rudder angle [rad]
        params: dict of 36 hydrodynamic derivatives.
        L: Ship length [m]
        U0: Approach speed [m/s]

    Returns:
        u_dot: Surge acceleration [m/s^2]
        v_dot: Sway acceleration [m/s^2]
        r_dot: Yaw acceleration [rad/s^2]
    """
    f1_p, f2_p, f3_p = compute_hydrodynamic_forces(u, v, r, delta, params=params, L=L, U0=U0)

    m_p = SHIP_PARTICULARS['m_prime']
    xG_p = SHIP_PARTICULARS['xG_prime']
    Iz_p = SHIP_PARTICULARS['Iz_prime']

    Xudot_p = FIXED_ACCELERATION_DERIVATIVES['X_udot_prime']
    Yvdot_p = FIXED_ACCELERATION_DERIVATIVES['Y_vdot_prime']
    Yrdot_p = FIXED_ACCELERATION_DERIVATIVES['Y_rdot_prime']
    Nvdot_p = FIXED_ACCELERATION_DERIVATIVES['N_vdot_prime']
    Nrdot_p = FIXED_ACCELERATION_DERIVATIVES['N_rdot_prime']

    # Page 4, Equation (5) - Mass matrix determinant
    # detM = (m - Y_vdot)*(Iz - N_rdot) - (m*xG - Y_rdot)*(m*xG - N_vdot)
    detM = (m_p - Yvdot_p) * (Iz_p - Nrdot_p) - (m_p * xG_p - Yrdot_p) * (m_p * xG_p - Nvdot_p)

    # Page 4, Equation (5) - Non-dimensional accelerations
    u_dot_p = f1_p / (m_p - Xudot_p)
    v_dot_p = ((Iz_p - Nrdot_p) * f2_p - (m_p * xG_p - Yrdot_p) * f3_p) / detM
    r_dot_p = ((m_p - Yvdot_p) * f3_p - (m_p * xG_p - Nvdot_p) * f2_p) / detM

    # Convert non-dimensional accelerations to dimensional physical units
    # Reference: notes/paper_spec.md, Section 3
    U = np.maximum(u, 1e-4)
    u_dot = u_dot_p * (U ** 2) / L
    v_dot = v_dot_p * (U ** 2) / L
    r_dot = r_dot_p * (U ** 2) / (L ** 2)

    return u_dot, v_dot, r_dot


def euler_step(u, v, r, psi=0.0, x=0.0, y=0.0, u_dot=0.0, v_dot=0.0, r_dot=0.0, dt=1.0, L=SHIP_PARTICULARS['L']):
    """
    Updates ship motion state using the explicit Euler forward integration scheme.
    Reference: Page 4, Equation (6).

    Parameters:
        u, v, r: Current velocities [m/s, m/s, rad/s]
        psi: Current heading angle [rad]
        x, y: Current Earth-fixed non-dimensional positions (x/L, y/L)
        u_dot, v_dot, r_dot: Accelerations [m/s^2, m/s^2, rad/s^2]
        dt: Time step [s] (default: 1.0 s)
        L: Ship length [m]

    Returns:
        u_next, v_next, r_next, psi_next, x_next, y_next
    """
    # Velocities update (Page 4, Equation 6)
    u_next = u + u_dot * dt
    v_next = v + v_dot * dt
    r_next = r + r_dot * dt

    # Heading update
    psi_next = psi + r * dt

    # Non-dimensional Earth-fixed position update
    # dx/dt = (u*cos(psi) - v*sin(psi)) / L
    # dy/dt = (u*sin(psi) + v*cos(psi)) / L
    x_next = x + (u * np.cos(psi) - v * np.sin(psi)) * dt / L
    y_next = y + (u * np.sin(psi) + v * np.cos(psi)) * dt / L

    return u_next, v_next, r_next, psi_next, x_next, y_next


def simulate_trajectory(rudder_history, initial_state=None, params=None, dt=1.0, L=SHIP_PARTICULARS['L'], U0=SHIP_PARTICULARS['U0']):
    """
    Recursively simulates a full maneuver trajectory from rudder commands.
    Reference: Page 9, Section 4.4 (Identified model approach).

    Parameters:
        rudder_history: 1D array of rudder angles [rad] at each second.
        initial_state: tuple (u0, v0, r0, psi0, x0, y0).
                       Defaults to (U0, 0.0, 0.0, 0.0, 0.0, 0.0).
        params: Hydrodynamic parameters dict.
        dt: Integration time step [s].

    Returns:
        trajectory: dict containing numpy arrays for 'time', 'u', 'v', 'r', 'psi', 'x', 'y', 'delta'.
    """
    if initial_state is None:
        u, v, r, psi, x, y = U0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        u, v, r, psi, x, y = initial_state

    n_steps = len(rudder_history)
    history = {
        'time': np.arange(n_steps) * dt,
        'u': np.zeros(n_steps),
        'v': np.zeros(n_steps),
        'r': np.zeros(n_steps),
        'psi': np.zeros(n_steps),
        'x': np.zeros(n_steps),
        'y': np.zeros(n_steps),
        'delta': np.array(rudder_history)
    }

    for t in range(n_steps):
        history['u'][t] = u
        history['v'][t] = v
        history['r'][t] = r
        history['psi'][t] = psi
        history['x'][t] = x
        history['y'][t] = y

        delta = rudder_history[t]
        u_dot, v_dot, r_dot = compute_accelerations(u, v, r, delta, params=params, L=L, U0=U0)
        u, v, r, psi, x, y = euler_step(u, v, r, psi, x, y, u_dot, v_dot, r_dot, dt=dt, L=L)

    return history
