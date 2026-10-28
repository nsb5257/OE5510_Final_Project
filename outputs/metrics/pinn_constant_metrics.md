# Pinn_constant Evaluation Metrics (RMSE)

Evaluated via recursive forward Euler time simulation (Equation 6, $\Delta t = 1.0$ s).

| Maneuver | Model | RMSE u [m/s] | RMSE v [m/s] | RMSE r [deg/s] | RMSE r [rad/s] | Sum (with deg/s) |
|---|---|---|---|---|---|---|
| `turning_circle_30` | **Pinn_constant** | 0.1068 | 0.0360 | 0.0030 | 0.000052 | 0.1458 |
| `zigzag_20_10` | **Pinn_constant** | 0.0292 | 0.0194 | 0.0067 | 0.000116 | 0.0552 |
| `zigzag_25_5` | **Pinn_constant** | 0.0567 | 0.0519 | 0.0155 | 0.000270 | 0.1241 |
