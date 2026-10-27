# Baseline Evaluation Metrics (RMSE)

Evaluated via recursive forward Euler time simulation (Equation 6, $\Delta t = 1.0$ s).

| Maneuver | Model | RMSE u [m/s] | RMSE v [m/s] | RMSE r [deg/s] | RMSE r [rad/s] | Sum (with deg/s) |
|---|---|---|---|---|---|---|
| `turning_circle_30` | **Baseline** | 0.0320 | 0.0040 | 0.0025 | 0.000044 | 0.0386 |
| `zigzag_20_10` | **Baseline** | 0.0066 | 0.0089 | 0.0026 | 0.000046 | 0.0181 |
| `zigzag_25_5` | **Baseline** | 0.0251 | 0.0306 | 0.0090 | 0.000157 | 0.0647 |
