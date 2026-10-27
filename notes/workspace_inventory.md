# Workspace Inventory

**Date of Audit:** 27 October 2026  
**Project:** Reproduction of *Identification modeling of ship maneuvering motion based on physics-informed neural network* (Hu et al., 2026)  
**Workspace:** `C:\Users\renuv\OneDrive\Desktop\OE5510_Final_project`

---

## 1. Discovered Files and Directory Structure

| File Path | Format | Size (Bytes) | Lines / Shape | Role / Description |
|---|---|---|---|---|
| `plan.md` | Markdown | 22,198 | 329 lines | Two-week execution plan with tasks T0–T11 and source-of-truth rules. |
| `README.md` | Markdown | 27 | 1 line | Minimal project header ("# OE5510_Final_Project"). |
| `antigravity_start_prompt.txt` | Text | 2,997 | 25 lines | Initial instructions and constraints for the reproduction project. |
| `Identification modeling of ship maneuvering motion based on physics-informed neural network.pdf` | Scanned PDF | 21,471,507 | 19 pages | Full original research paper (scanned, 1 full-page raster image per page). |
| `data/README.txt` | Text | 2,059 | 42 lines | Dataset documentation detailing maneuver types, sampling, and variable definitions. |
| `data/train_data/turning_circle_35.txt` | Whitespace-delimited TXT | 179,874 | 1,000 lines (999 data + 1 header) | Training: 35° turning circle simulation (1000 s @ 1 s interval). 7 columns. |
| `data/train_data/zigzag_10_10.txt` | Whitespace-delimited TXT | 178,931 | 1,000 lines (999 data + 1 header) | Training: 10°/10° zigzag simulation (1000 s @ 1 s interval). 7 columns. |
| `data/train_data/zigzag_20_20.txt` | Whitespace-delimited TXT | 178,917 | 1,000 lines (999 data + 1 header) | Training: 20°/20° zigzag simulation (1000 s @ 1 s interval). 7 columns. |
| `data/test_data/turning_circle_30.txt` | Whitespace-delimited TXT | 153,072 | 1,001 lines (1000 data + 1 header) | Test: 30° turning circle simulation (1000 s). 6 columns (`u, v, r, rudder_angle, y-position, x-position`). |
| `data/test_data/zigzag_20_10.txt` | Whitespace-delimited TXT | 128,140 | 1,001 lines (1000 data + 1 header) | Test: 20°/10° zigzag simulation (1000 s). 5 columns (`u, v, r, rudder_angle, heading_angle`). |
| `data/test_data/zigzag_25_5.txt` | Whitespace-delimited TXT | 128,060 | 1,001 lines (1000 data + 1 header) | Test: 25°/5° zigzag simulation (1000 s). 5 columns (`u, v, r, rudder_angle, heading_angle`). |

---

## 2. Dataset Roles, Columns, and Statistics

### Training Dataset (`data/train_data/`)
Total training points: $999 \times 3 = 2,997$ rows (paper cites 3,000 points sampled for 1000 s at 1 s intervals).
All 3 training files have identical 7 columns:
`u v r rudder_angle du_dt dv_dt dr_dt`

- `u`: Surge velocity ($m/s$), range $\approx 3.41$ to $7.956$ m/s ($u_0 = 7.956$ m/s).
- `v`: Sway velocity ($m/s$), range $\approx -1.72$ to $1.37$ m/s.
- `r`: Yaw rate ($rad/s$), range $\approx -0.009$ to $0.0117$ rad/s.
- `rudder_angle`: Rudder angle in radians ($rad$), turning rate $2.34^\circ/s = 0.04084$ rad/s.
  - $35^\circ$ turning circle: constant $-35^\circ = -0.610865$ rad after ramp.
  - $10^\circ/10^\circ$ zigzag: range $[-0.1745, +0.1745]$ rad.
  - $20^\circ/20^\circ$ zigzag: range $[-0.3491, +0.3491]$ rad.
- `du_dt`: Surge acceleration ($m/s^2$), range $\approx -0.0267$ to $0.0069$ m/s$^2$.
- `dv_dt`: Sway acceleration ($m/s^2$), range $\approx -0.0266$ to $0.0183$ m/s$^2$.
- `dr_dt`: Yaw acceleration ($rad/s^2$), range $\approx -2.57 \times 10^{-4}$ to $2.50 \times 10^{-4}$ rad/s$^2$.

### Test Dataset (`data/test_data/`)
All 3 test files have 1000 data rows (1001 lines with header).
- `turning_circle_30.txt`: Columns: `u, v, r, rudder_angle, y-position, x-position` (6 columns).
  - Note: Does not contain acceleration columns `du_dt, dv_dt, dr_dt`. Trajectory positions are non-dimensional ($x/L, y/L$).
- `zigzag_20_10.txt`: Columns: `u, v, r, rudder_angle, heading_angle` (5 columns).
  - Note: Does not contain accelerations or trajectory positions; contains `heading_angle` in radians.
- `zigzag_25_5.txt`: Columns: `u, v, r, rudder_angle, heading_angle` (5 columns).
  - Note: Contains `heading_angle` in radians, no accelerations.

### Verification of Physical Ground Truth
Using the paper's Abkowitz model equations and Table 2 original parameters, numerical integration of accelerations over the velocities reproduces `du_dt`, `dv_dt`, `dr_dt` across all training files with numerical precision $< 10^{-5}$.

---

## 3. Python Environment Audit

- **Python Version:** 3.13.3 (64-bit on Windows)
- **PyTorch:** 2.13.0+cpu (`torch.cuda.is_available() = False`, CPU training only)
- **NumPy:** 2.3.4
- **SciPy:** 1.16.3
- **Matplotlib:** 3.10.1
- **Pandas:** 3.0.6
- **Scikit-Learn:** 1.6.1
- **PyPDF:** 6.13.1 (installed; used to inspect PDF structures)
- **Pillow (PIL):** 11.1.0 (installed; used to inspect page images)
- **PyTest:** 9.1.1 (installed; available for unit tests)

*Note on hardware:*
The paper reported 120,000 iterations taking ~8 minutes on an Intel Core i7-12700K CPU. On our CPU with PyTorch CPU, the network is tiny (4 inputs, 3 hidden layers of 10 neurons, 3 outputs), making training manageable.

---

## 4. Unknowns, Discrepancies, and Observations

1. **PDF Format:** The supplied PDF file is image-based (scanned/rendered pages with no selectable text). Each page consists of a single high-resolution raster image.
2. **Missing Test Accelerations:** The test files do not include `du_dt, dv_dt, dr_dt`. The paper evaluates generalization by iterative multi-step forward time simulation (Equation 6), comparing trajectories of $u, v, r, \psi, x/L, y/L$, not one-step acceleration RMSE.
3. **Column Name Variance:** In `turning_circle_30.txt`, lateral position is labeled `y-position` and longitudinal position `x-position` (with hyphens), while `data/README.txt` writes `y_position` and `x_position`.
