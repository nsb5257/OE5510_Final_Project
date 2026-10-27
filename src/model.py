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

        # Fixed parameters (registered as buffers so they move with device but do not require gradients)
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

    def compute_physics_accelerations(self, u, v, r, delta):
        """
        Differentiable computation of physical accelerations via Equations (4) and (5).
        Inputs are raw physical tensors: u [m/s], v [m/s], r [rad/s], delta [rad].
        """
        U = torch.clamp(u, min=1e-4)

        # Non-dimensional states (Prime-I system)
        u_p = u / U
        v_p = v / U
        r_p = r * self.L / U
        delta_u_p = (u - self.U0) / U
        delta_p = delta

        p = self.hydro_params

        # Page 4, Equation (4) - Non-dimensional forces f1', f2', f3'
        f1_p = (
            p['Xu_prime'] * delta_u_p
            + p['Xuu_prime'] * (delta_u_p ** 2)
            + p['Xvv_prime'] * (v_p ** 2)
            + (self.m_prime * self.xG_prime + p['Xrr_prime']) * (r_p ** 2)
            + (self.m_prime + p['Xvr_prime']) * v_p * r_p
            + p['Xdd_prime'] * (delta_p ** 2)
            + p['Xvd_prime'] * v_p * delta_p
            + p['Xrd_prime'] * r_p * delta_p
        )

        f2_p = (
            -self.m_prime * u_p * r_p
            + p['Yv_prime'] * v_p
            + p['Yvvv_prime'] * (v_p ** 3)
            + p['Yr_prime'] * r_p
            + p['Yrrr_prime'] * (r_p ** 3)
            + p['Yd_prime'] * delta_p
            + p['Ydd_prime'] * (delta_p ** 2)
            + p['Yddd_prime'] * (delta_p ** 3)
            + p['Yvrr_prime'] * v_p * (r_p ** 2)
            + p['Yvvr_prime'] * (v_p ** 2) * r_p
            + p['Yud_prime'] * delta_u_p * delta_p
            + p['Yvdd_prime'] * v_p * (delta_p ** 2)
            + p['Yvvd_prime'] * (v_p ** 2) * delta_p
            + p['Yrdd_prime'] * r_p * (delta_p ** 2)
            + p['Yrrd_prime'] * (r_p ** 2) * delta_p
        )

        f3_p = (
            -self.m_prime * self.xG_prime * u_p * r_p
            + p['Nv_prime'] * v_p
            + p['Nvvv_prime'] * (v_p ** 3)
            + p['Nr_prime'] * r_p
            + p['Nrrr_prime'] * (r_p ** 3)
            + p['Nd_prime'] * delta_p
            + p['Ndd_prime'] * (delta_p ** 2)
            + p['Nddd_prime'] * (delta_p ** 3)
            + p['Nvrr_prime'] * v_p * (r_p ** 2)
            + p['Nvvr_prime'] * (v_p ** 2) * r_p
            + p['Nud_prime'] * delta_u_p * delta_p
            + p['Nvdd_prime'] * v_p * (delta_p ** 2)
            + p['Nvvd_prime'] * (v_p ** 2) * delta_p
            + p['Nrdd_prime'] * r_p * (delta_p ** 2)
            + p['Nrrd_prime'] * (r_p ** 2) * delta_p
        )

        # Page 4, Equation (5) - Mass matrix determinant & accelerations
        detM = (self.m_prime - self.Y_vdot_p) * (self.Iz_prime - self.N_rdot_p) - \
               (self.m_prime * self.xG_prime - self.Y_rdot_p) * (self.m_prime * self.xG_prime - self.N_vdot_p)

        u_dot_p = f1_p / (self.m_prime - self.X_udot_p)
        v_dot_p = ((self.Iz_prime - self.N_rdot_p) * f2_p - (self.m_prime * self.xG_prime - self.Y_rdot_p) * f3_p) / detM
        r_dot_p = ((self.m_prime - self.Y_vdot_p) * f3_p - (self.m_prime * self.xG_prime - self.N_vdot_p) * f2_p) / detM

        # Dimensional accelerations
        u_dot = u_dot_p * (U ** 2) / self.L
        v_dot = v_dot_p * (U ** 2) / self.L
        r_dot = r_dot_p * (U ** 2) / (self.L ** 2)

        return u_dot, v_dot, r_dot

    def compute_loss(self, inputs_norm, targets_norm, inputs_raw, tgt_mean, tgt_std, beta=0.01):
        """
        Computes total loss, observation data loss, and physical residuals.
        Reference: Page 5, Equations (15)-(18).
        """
        # Neural network forward pass
        pred_norm = self.forward(inputs_norm)

        # 1. Observation Data Loss (Equation 17):
        # Loss_data = (1/No) * sum( |u_dot_data - u_dot|^2 + |v_dot_data - v_dot|^2 + |r_dot_data - r_dot|^2 )
        # Summed over 3 components, averaged over No batch samples
        loss_data = torch.mean(torch.sum((pred_norm - targets_norm) ** 2, dim=-1))

        # 2. Destandardize neural network predictions to physical units
        pred_raw = pred_norm * tgt_std + tgt_mean
        u_dot_nn = pred_raw[:, 0]
        v_dot_nn = pred_raw[:, 1]
        r_dot_nn = pred_raw[:, 2]

        # 3. Evaluate physical accelerations from current 36 hydrodynamic derivatives
        u_raw = inputs_raw[:, 0]
        v_raw = inputs_raw[:, 1]
        r_raw = inputs_raw[:, 2]
        delta_raw = inputs_raw[:, 3]
        u_dot_phy, v_dot_phy, r_dot_phy = self.compute_physics_accelerations(u_raw, v_raw, r_raw, delta_raw)

        # 4. Physical Residuals (Equation 15):
        # e1 = |u_dot_phy - u_dot|, e2 = |v_dot_phy - v_dot|, e3 = |r_dot_phy - r_dot|
        e1 = u_dot_phy - u_dot_nn
        e2 = v_dot_phy - v_dot_nn
        e3 = r_dot_phy - r_dot_nn

        # 5. Physics-based Loss (Equation 16), normalized by target scale:
        loss_phy_u = torch.mean((e1 / tgt_std[0]) ** 2)
        loss_phy_v = torch.mean((e2 / tgt_std[1]) ** 2)
        loss_phy_r = torch.mean((e3 / tgt_std[2]) ** 2)
        loss_phy = loss_phy_u + loss_phy_v + loss_phy_r

        # 6. Total Loss (Equation 18):
        # Loss = Loss_data + beta * Loss_phy
        total_loss = loss_data + beta * loss_phy

        return {
            'total_loss': total_loss,
            'loss_data': loss_data,
            'loss_phy': loss_phy,
            'loss_phy_u': loss_phy_u,
            'loss_phy_v': loss_phy_v,
            'loss_phy_r': loss_phy_r
        }
