"""
model.py
Neural network architectures for 3-DOF ship maneuvering motion.
Implements:
  1. ShipMotionNN: Standard MLP baseline (Section 3.1)
  2. ShipPINN: Physics-Informed Neural Network with 36 trainable
     hydrodynamic parameters and physical residuals (Sections 3.2, 3.3, 4.2)
Reference:
  "Identification modeling of ship maneuvering motion based on physics-informed neural network"
  (Hu et al., 2026, DOI: 10.1080/17445302.2026.2648148)
"""

import torch
import torch.nn as nn
from src.ship_dynamics import SHIP_PARTICULARS, FIXED_ACCELERATION_DERIVATIVES, ORIGINAL_HYDRODYNAMIC_DERIVATIVES

# Parameter names in fixed order for vectorized computations
X_PARAM_NAMES = [
    'Xu_prime', 'Xuu_prime', 'Xvv_prime', 'Xrr_prime', 'Xvr_prime', 'Xdd_prime', 'Xvd_prime', 'Xrd_prime'
]
Y_PARAM_NAMES = [
    'Yv_prime', 'Yvvv_prime', 'Yr_prime', 'Yrrr_prime', 'Yd_prime', 'Ydd_prime', 'Yddd_prime',
    'Yvrr_prime', 'Yvvr_prime', 'Yud_prime', 'Yvdd_prime', 'Yvvd_prime', 'Yrdd_prime', 'Yrrd_prime'
]
N_PARAM_NAMES = [
    'Nv_prime', 'Nvvv_prime', 'Nr_prime', 'Nrrr_prime', 'Nd_prime', 'Ndd_prime', 'Nddd_prime',
    'Nvrr_prime', 'Nvvr_prime', 'Nud_prime', 'Nvdd_prime', 'Nvvd_prime', 'Nrdd_prime', 'Nrrd_prime'
]


class ShipMotionNN(nn.Module):
    """
    Standard feed-forward neural network for ship maneuvering prediction.
    Reference: Page 4, Section 3.1 & Page 6, Section 4.2.
    Architecture:
      - 4 inputs: u, v, r, rudder_angle
      - 3 hidden layers, each with 10 neurons
      - Activation: Tanh on all hidden layers
      - 3 outputs: du_dt, dv_dt, dr_dt (standardized accelerations)
      - Weight initialization: Xavier uniform (Glorot and Bengio, 2010)
      - Bias initialization: zeros
    """
    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ShipPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for 3-DOF ship maneuvering motion.
    Reference: Page 5, Equations (15)-(18) & Page 6, Figure 2, Section 4.2.

    Components:
      - Neural network (ShipMotionNN) predicting standardized accelerations.
      - 36 independent trainable hydrodynamic derivatives initialized to 0.0 (Page 6, Section 4.2).
      - Fixed mass properties and 5 acceleration derivatives (Page 7).
      - Autograd-differentiable physics residuals for surge, sway, and yaw.
    """
    def __init__(self, L=SHIP_PARTICULARS['L'], U0=SHIP_PARTICULARS['U0']):
        super().__init__()
        self.L = L
        self.U0 = U0

        # Base neural network
        self.nn = ShipMotionNN(input_dim=4, hidden_dim=10, output_dim=3)

        # 36 Trainable hydrodynamic parameters initialized to 0.0 (Page 6, Section 4.2)
        self.hydro_params = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            for k in ORIGINAL_HYDRODYNAMIC_DERIVATIVES.keys()
        })

        # Fixed parameters (registered as buffers)
        self.register_buffer('m_prime', torch.tensor(SHIP_PARTICULARS['m_prime'], dtype=torch.float32))
        self.register_buffer('xG_prime', torch.tensor(SHIP_PARTICULARS['xG_prime'], dtype=torch.float32))
        self.register_buffer('Iz_prime', torch.tensor(SHIP_PARTICULARS['Iz_prime'], dtype=torch.float32))

        self.register_buffer('X_udot_p', torch.tensor(FIXED_ACCELERATION_DERIVATIVES['X_udot_prime'], dtype=torch.float32))
        self.register_buffer('Y_vdot_p', torch.tensor(FIXED_ACCELERATION_DERIVATIVES['Y_vdot_prime'], dtype=torch.float32))
        self.register_buffer('Y_rdot_p', torch.tensor(FIXED_ACCELERATION_DERIVATIVES['Y_rdot_prime'], dtype=torch.float32))
        self.register_buffer('N_vdot_p', torch.tensor(FIXED_ACCELERATION_DERIVATIVES['N_vdot_prime'], dtype=torch.float32))
        self.register_buffer('N_rdot_p', torch.tensor(FIXED_ACCELERATION_DERIVATIVES['N_rdot_prime'], dtype=torch.float32))

    def forward(self, x_norm):
        """Forward pass through neural network."""
        return self.nn(x_norm)

    def precompute_basis(self, inputs_raw):
        """
        Precomputes the invariant kinematic basis matrices for training samples.
        Inputs: inputs_raw tensor [N, 4] with columns (u, v, r, delta).
        Returns a dict of tensors enabling fast O(1) matrix multiplication per step.
        """
        u = inputs_raw[:, 0]
        v = inputs_raw[:, 1]
        r = inputs_raw[:, 2]
        delta = inputs_raw[:, 3]

        U = torch.clamp(u, min=1e-4)
        u_p = u / U
        v_p = v / U
        r_p = r * self.L / U
        delta_u_p = (u - self.U0) / U
        delta_p = delta

        # Basis for f1 (8 terms)
        phi1 = torch.stack([
            delta_u_p, delta_u_p**2, v_p**2, r_p**2, v_p * r_p,
            delta_p**2, v_p * delta_p, r_p * delta_p
        ], dim=1)
        f1_const = self.m_prime * self.xG_prime * (r_p**2) + self.m_prime * v_p * r_p

        # Basis for f2 and f3 (14 terms)
        phi2 = torch.stack([
            v_p, v_p**3, r_p, r_p**3, delta_p, delta_p**2, delta_p**3,
            v_p * (r_p**2), (v_p**2) * r_p, delta_u_p * delta_p,
            v_p * (delta_p**2), (v_p**2) * delta_p, r_p * (delta_p**2), (r_p**2) * delta_p
        ], dim=1)
        f2_const = -self.m_prime * u_p * r_p
        f3_const = -self.m_prime * self.xG_prime * u_p * r_p

        detM = (self.m_prime - self.Y_vdot_p) * (self.Iz_prime - self.N_rdot_p) - \
               (self.m_prime * self.xG_prime - self.Y_rdot_p) * (self.m_prime * self.xG_prime - self.N_vdot_p)

        scale_lin = (U ** 2) / self.L
        scale_ang = (U ** 2) / (self.L ** 2)

        return {
            'phi1': phi1,
            'f1_const': f1_const,
            'phi2': phi2,
            'f2_const': f2_const,
            'f3_const': f3_const,
            'detM': detM,
            'scale_lin': scale_lin,
            'scale_ang': scale_ang
        }

    def compute_physics_accelerations(self, u, v, r, delta):
        """
        Differentiable computation of physical accelerations via Equations (4) and (5).
        Inputs are raw physical tensors: u [m/s], v [m/s], r [rad/s], delta [rad].
        """
        inputs_raw = torch.stack([u, v, r, delta], dim=1) if u.dim() > 0 else torch.stack([u, v, r, delta]).unsqueeze(0)
        basis = self.precompute_basis(inputs_raw)

        theta1 = torch.stack([self.hydro_params[k] for k in X_PARAM_NAMES])
        theta2 = torch.stack([self.hydro_params[k] for k in Y_PARAM_NAMES])
        theta3 = torch.stack([self.hydro_params[k] for k in N_PARAM_NAMES])

        f1_p = torch.matmul(basis['phi1'], theta1) + basis['f1_const']
        f2_p = torch.matmul(basis['phi2'], theta2) + basis['f2_const']
        f3_p = torch.matmul(basis['phi2'], theta3) + basis['f3_const']

        u_dot_p = f1_p / (self.m_prime - self.X_udot_p)
        v_dot_p = ((self.Iz_prime - self.N_rdot_p) * f2_p - (self.m_prime * self.xG_prime - self.Y_rdot_p) * f3_p) / basis['detM']
        r_dot_p = ((self.m_prime - self.Y_vdot_p) * f3_p - (self.m_prime * self.xG_prime - self.N_vdot_p) * f2_p) / basis['detM']

        u_dot = u_dot_p * basis['scale_lin']
        v_dot = v_dot_p * basis['scale_lin']
        r_dot = r_dot_p * basis['scale_ang']

        if u.dim() == 0:
            return u_dot.squeeze(0), v_dot.squeeze(0), r_dot.squeeze(0)
        return u_dot, v_dot, r_dot

    def compute_loss(self, inputs_norm, targets_norm, inputs_raw=None, tgt_mean=None, tgt_std=None, beta=0.01, basis=None):
        """
        Computes total loss, observation data loss, and physical residuals.
        Reference: Page 5, Equations (15)-(18).
        """
        # Neural network forward pass
        pred_norm = self.forward(inputs_norm)

        # 1. Observation Data Loss (Equation 17)
        loss_data = torch.mean(torch.sum((pred_norm - targets_norm) ** 2, dim=-1))

        # 2. Destandardize neural network predictions to physical units
        pred_raw = pred_norm * tgt_std + tgt_mean
        u_dot_nn = pred_raw[:, 0]
        v_dot_nn = pred_raw[:, 1]
        r_dot_nn = pred_raw[:, 2]

        # 3. Evaluate physical accelerations
        if basis is None:
            basis = self.precompute_basis(inputs_raw)

        theta1 = torch.stack([self.hydro_params[k] for k in X_PARAM_NAMES])
        theta2 = torch.stack([self.hydro_params[k] for k in Y_PARAM_NAMES])
        theta3 = torch.stack([self.hydro_params[k] for k in N_PARAM_NAMES])

        f1_p = torch.matmul(basis['phi1'], theta1) + basis['f1_const']
        f2_p = torch.matmul(basis['phi2'], theta2) + basis['f2_const']
        f3_p = torch.matmul(basis['phi2'], theta3) + basis['f3_const']

        u_dot_p = f1_p / (self.m_prime - self.X_udot_p)
        v_dot_p = ((self.Iz_prime - self.N_rdot_p) * f2_p - (self.m_prime * self.xG_prime - self.Y_rdot_p) * f3_p) / basis['detM']
        r_dot_p = ((self.m_prime - self.Y_vdot_p) * f3_p - (self.m_prime * self.xG_prime - self.N_vdot_p) * f2_p) / basis['detM']

        u_dot_phy = u_dot_p * basis['scale_lin']
        v_dot_phy = v_dot_p * basis['scale_lin']
        r_dot_phy = r_dot_p * basis['scale_ang']

        # 4. Physical Residuals (Equation 15):
        e1 = u_dot_phy - u_dot_nn
        e2 = v_dot_phy - v_dot_nn
        e3 = r_dot_phy - r_dot_nn

        # 5. Physics-based Loss (Equation 16), normalized by target scale:
        loss_phy_u = torch.mean((e1 / tgt_std[0]) ** 2)
        loss_phy_v = torch.mean((e2 / tgt_std[1]) ** 2)
        loss_phy_r = torch.mean((e3 / tgt_std[2]) ** 2)
        loss_phy = loss_phy_u + loss_phy_v + loss_phy_r

        # 6. Total Loss (Equation 18):
        total_loss = loss_data + beta * loss_phy

        return {
            'total_loss': total_loss,
            'loss_data': loss_data,
            'loss_phy': loss_phy,
            'loss_phy_u': loss_phy_u,
            'loss_phy_v': loss_phy_v,
            'loss_phy_r': loss_phy_r
        }
