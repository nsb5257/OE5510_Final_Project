# Dataset Audit & Profile Report

**Generated:** 27 October 2026
**Script:** `src/inspect_data.py`

## 1. Overview of Maneuver Files

| Split | File | Rows | Columns | Missing Values | Duplicate Rows |
|---|---|---|---|---|---|
| **train** | `turning_circle_35.txt` | 999 | `u`, `v`, `r`, `rudder_angle`, `du_dt`, `dv_dt`, `dr_dt` | 0 | 0 |
| **train** | `zigzag_10_10.txt` | 999 | `u`, `v`, `r`, `rudder_angle`, `du_dt`, `dv_dt`, `dr_dt` | 0 | 0 |
| **train** | `zigzag_20_20.txt` | 999 | `u`, `v`, `r`, `rudder_angle`, `du_dt`, `dv_dt`, `dr_dt` | 0 | 0 |
| **test** | `turning_circle_30.txt` | 1000 | `u`, `v`, `r`, `rudder_angle`, `y-position`, `x-position` | 0 | 0 |
| **test** | `zigzag_20_10.txt` | 1000 | `u`, `v`, `r`, `rudder_angle`, `heading_angle` | 0 | 0 |
| **test** | `zigzag_25_5.txt` | 1000 | `u`, `v`, `r`, `rudder_angle`, `heading_angle` | 0 | 0 |

## 2. Variable Definitions and Units Mapping

| Column Name | Variable | Physical Unit | Description |
|---|---|---|---|
| `u` | Surge velocity | m/s | Body-fixed forward velocity |
| `v` | Sway velocity | m/s | Body-fixed lateral velocity |
| `r` | Yaw rate | rad/s | Angular rate of rotation about z-axis |
| `rudder_angle` | Rudder angle | rad | Control input angle of the rudder |
| `heading_angle` | Heading angle | rad | Ship orientation in Earth-fixed frame |
| `du_dt` | Surge acceleration | m/s^2 | Time derivative of u |
| `dv_dt` | Sway acceleration | m/s^2 | Time derivative of v |
| `dr_dt` | Yaw acceleration | rad/s^2 | Time derivative of r |
| `x-position` | Longitudinal position | non-dim (x/L) | Position along Earth-fixed X normalized by L |
| `y-position` | Lateral position | non-dim (y/L) | Position along Earth-fixed Y normalized by L |

## 3. Summary Statistics by File

### `turning_circle_35.txt` (train)

| Column | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| `u` | 3.4060e+00 | 7.9560e+00 | 4.1052e+00 | 1.2434e+00 |
| `v` | -1.7232e+00 | 0.0000e+00 | -1.1993e+00 | 2.5687e-01 |
| `r` | 0.0000e+00 | 1.1737e-02 | 8.4940e-03 | 1.4831e-03 |
| `rudder_angle` | -6.1087e-01 | 0.0000e+00 | -6.0599e-01 | 4.5072e-02 |
| `du_dt` | -2.6654e-02 | 0.0000e+00 | -4.5546e-03 | 7.3497e-03 |
| `dv_dt` | -2.6571e-02 | 3.3503e-03 | -1.1300e-03 | 6.0015e-03 |
| `dr_dt` | -2.5653e-05 | 2.5026e-04 | 8.1566e-06 | 4.5371e-05 |

### `zigzag_10_10.txt` (train)

| Column | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| `u` | 6.5459e+00 | 7.9560e+00 | 7.1878e+00 | 4.0589e-01 |
| `v` | -1.1609e+00 | 1.1584e+00 | -8.7228e-02 | 6.5486e-01 |
| `r` | -7.0910e-03 | 7.0564e-03 | 5.2631e-04 | 3.7762e-03 |
| `rudder_angle` | -1.7453e-01 | 1.7453e-01 | -8.6760e-03 | 1.7211e-01 |
| `du_dt` | -8.3863e-03 | 3.5732e-03 | -1.2515e-03 | 3.2781e-03 |
| `dv_dt` | -1.0087e-02 | 1.0290e-02 | -3.6123e-04 | 7.8834e-03 |
| `dr_dt` | -1.6886e-04 | 1.6991e-04 | 5.8509e-07 | 5.6522e-05 |

### `zigzag_20_20.txt` (train)

| Column | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| `u` | 5.3435e+00 | 7.9560e+00 | 6.2271e+00 | 7.0013e-01 |
| `v` | -1.4626e+00 | 1.3669e+00 | -5.3824e-02 | 8.6708e-01 |
| `r` | -8.7591e-03 | 9.1885e-03 | 1.3571e-04 | 5.7100e-03 |
| `rudder_angle` | -3.4907e-01 | 3.4907e-01 | 1.1255e-02 | 3.3990e-01 |
| `du_dt` | -1.3844e-02 | 6.9069e-03 | -2.0589e-03 | 6.2497e-03 |
| `dv_dt` | -1.7175e-02 | 1.8287e-02 | 1.0434e-03 | 1.2479e-02 |
| `dr_dt` | -2.4388e-04 | 2.4931e-04 | -7.9741e-06 | 9.2266e-05 |

### `turning_circle_30.txt` (test)

| Column | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| `u` | 3.6240e+00 | 7.9560e+00 | 4.3354e+00 | 1.2130e+00 |
| `v` | -1.6788e+00 | 0.0000e+00 | -1.2009e+00 | 2.5199e-01 |
| `r` | 0.0000e+00 | 1.1118e-02 | 8.2153e-03 | 1.4137e-03 |
| `rudder_angle` | -5.2360e-01 | 0.0000e+00 | -5.1998e-01 | 3.6062e-02 |
| `y-position` | -2.1336e-03 | 3.5400e+00 | 1.6999e+00 | 1.1421e+00 |
| `x-position` | 0.0000e+00 | 3.3355e+00 | 1.8362e+00 | 1.0861e+00 |

### `zigzag_20_10.txt` (test)

| Column | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| `u` | 5.8600e+00 | 7.9560e+00 | 6.5477e+00 | 5.7050e-01 |
| `v` | -1.3096e+00 | 1.3013e+00 | 2.6493e-02 | 7.5744e-01 |
| `r` | -8.8848e-03 | 8.9608e-03 | 2.3541e-06 | 5.1198e-03 |
| `rudder_angle` | -3.4907e-01 | 3.4907e-01 | -2.8166e-02 | 3.3688e-01 |
| `heading_angle` | -4.5382e-01 | 4.5877e-01 | -4.0069e-02 | 3.1636e-01 |

### `zigzag_25_5.txt` (test)

| Column | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| `u` | 5.8752e+00 | 7.9560e+00 | 6.4612e+00 | 5.7379e-01 |
| `v` | -1.1843e+00 | 1.2111e+00 | 3.9188e-02 | 7.1140e-01 |
| `r` | -9.1880e-03 | 8.9924e-03 | -3.3446e-04 | 5.1441e-03 |
| `rudder_angle` | -4.3633e-01 | 4.3633e-01 | 3.7203e-03 | 4.1245e-01 |
| `heading_angle` | -3.6255e-01 | 3.5478e-01 | -7.0726e-03 | 2.4958e-01 |

## 4. Observations and Data Integrity Checks

1. **Completeness:** Zero missing or NaN values across all 6 files.
2. **Duplicates:** Zero duplicate rows found in any file.
3. **Sampling:** 1-second sampling interval (1 Hz) across 1000-second simulations (999 steps for train, 1000 for test).
4. **Accelerations:** Present in all 3 training files (`du_dt`, `dv_dt`, `dr_dt`); absent in test files.
5. **Trajectory Coordinates:** Available in test turning circle (`x-position`, `y-position` normalized by ship length $L = 320$ m).
6. **Heading Angle:** Available in test zigzag files (`heading_angle` in radians).
