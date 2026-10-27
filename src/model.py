"""
model.py
Neural network model definition for 3-DOF ship maneuvering motion.
Reference:
  "Identification modeling of ship maneuvering motion based on physics-informed neural network"
  (Hu et al., 2026, Section 3.1, Section 4.2, Figure 2)
"""

import torch
import torch.nn as nn


class ShipMotionNN(nn.Module):
    """
    Standard feed-forward neural network for ship maneuvering prediction.
    Architecture:
      - Input layer: 4 features (u, v, r, rudder_angle)
      - 3 hidden layers, each with 10 neurons
      - Activation: Tanh on all hidden layers
      - Output layer: 3 features (du_dt, dv_dt, dr_dt)
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
