# Pinn_dynamic Evaluation Metrics (RMSE)

Evaluated via recursive forward Euler time simulation (Equation 6, $\Delta t = 1.0$ s).

| Maneuver | Model | RMSE u [m/s] | RMSE v [m/s] | RMSE r [deg/s] | RMSE r [rad/s] | Sum (with deg/s) |
|---|---|---|---|---|---|---|
| `turning_circle_30` | **Pinn_dynamic** | 0.0532 | 0.0955 | 0.0473 | 0.000826 | 0.1960 |
| `zigzag_20_10` | **Pinn_dynamic** | 0.1664 | 0.1378 | 0.0396 | 0.000692 | 0.3439 |
| `zigzag_25_5` | **Pinn_dynamic** | 0.2526 | 0.1509 | 0.0782 | 0.001366 | 0.4817 |
