# Preprocessing Metadata & Normalization Specification

**Generated:** 27 October 2026
**Source:** `src/prepare_data.py`
**Paper Reference:** Page 6, Section 4.1, Equation (19)

## 1. Train / Test Split Strategy

Trajectory-level partition (no random row mixing, preventing time-series leakage):
- **Training Maneuvers (2,997 steps total):**
  - `turning_circle_35`: indices [0, 999) (999 rows)
  - `zigzag_10_10`: indices [999, 1998) (999 rows)
  - `zigzag_20_20`: indices [1998, 2997) (999 rows)
- **Held-out Test Maneuvers (1,000 steps each):**
  - `turning_circle_30`: 1,000 seconds
  - `zigzag_20_10`: 1,000 seconds
  - `zigzag_25_5`: 1,000 seconds

## 2. Feature & Target Ordering

- **Input Vector ($X_{in}$):** `[u, v, r, rudder_angle]` (4 features)
- **Target Acceleration Vector ($Y_{out}$):** `[du_dt, dv_dt, dr_dt]` (3 targets)

## 3. Normalization Statistics (Computed on Training Data Only)

Standardization formula: $x_{norm} = \frac{x - \mu}{\sigma}$

| Variable | Role | Physical Unit | Mean ($\mu$) | Std Dev ($\sigma$) |
|---|---|---|---|---|
| `u` | Input | physical | 5.840051e+00 | 1.546726e+00 |
| `v` | Input | physical | -4.467825e-01 | 8.358767e-01 |
| `r` | Input | physical | 3.052000e-03 | 5.584033e-03 |
| `rudder_angle` | Input | physical | -2.011354e-01 | 3.620423e-01 |
| `du_dt` | Target | physical | -2.621648e-03 | 6.046643e-03 |
| `dv_dt` | Target | physical | -1.492807e-04 | 9.240256e-03 |
| `dr_dt` | Target | physical | 2.558853e-07 | 6.803742e-05 |

## 4. Treatment of Physical Units in Loss Calculation

1. The neural network receives standardized inputs and outputs standardized accelerations.
2. Observation loss ($Loss_{data}$) compares predicted standardized accelerations with normalized target accelerations.
3. For physics-based loss ($Loss_{phy}$), predicted accelerations are destandardized: $\dot{u} = \dot{\bar{u}} \sigma + \mu$.
4. Physical residuals $e_1, e_2, e_3$ are evaluated in physical units and normalized by target variance so that $Loss_{data}$ and $Loss_{phy}$ maintain balanced unitless scales.
