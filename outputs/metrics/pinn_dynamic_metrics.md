# Pinn_dynamic Evaluation Metrics (RMSE)

Evaluated via recursive forward Euler time simulation (Equation 6, $\Delta t = 1.0$ s).

| Maneuver | Model | RMSE u [m/s] | RMSE v [m/s] | RMSE r [deg/s] | RMSE r [rad/s] | Sum (with deg/s) |
|---|---|---|---|---|---|---|
| `turning_circle_30` | **Pinn_dynamic** | 0.0520 | 0.0079 | 0.0022 | 0.000039 | 0.0621 |
| `zigzag_20_10` | **Pinn_dynamic** | 0.0091 | 0.0121 | 0.0037 | 0.000064 | 0.0249 |
| `zigzag_25_5` | **Pinn_dynamic** | 0.0283 | 0.0356 | 0.0108 | 0.000188 | 0.0747 |
