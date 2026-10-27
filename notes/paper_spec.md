# Research Paper Specification

**Paper Title:** Identification modeling of ship maneuvering motion based on physics-informed neural network  
**Authors:** Yi Hu, Shuo Liu, Tao Wang, Yu Zhang, Yong Cai  
**Journal:** *Ships and Offshore Structures* (Taylor & Francis, 2026)  
**DOI:** 10.1080/17445302.2026.2648148  
**Document Inspection Date:** 27 October 2026  
**Document Source:** Local PDF `Identification modeling of ship maneuvering motion based on physics-informed neural network.pdf` (19 pages, scanned image raster)

---

## 1. Physical System and Benchmark Vessel

### Benchmark Vessel: KVLCC2 (Second Variant)
Reference: Page 6, Section 4.1, Table 1.

| Parameter | Symbol | Paper Value (Table 1) | Unit | Dimensionless Form (Prime-I) |
|---|---|---|---|---|
| Length between perpendiculars | $L_{pp}$ | 320.00 | $\text{m}$ | $1.0$ |
| Breadth | $B$ | 58.00 | $\text{m}$ | $B/L = 0.18125$ |
| Design draft | $d$ | 20.80 | $\text{m}$ | $d/L = 0.065$ |
| Block coefficient | $C_b$ | 0.81 | - | $0.81$ |
| Longitudinal coordinate of CG | $x_G$ | 11.136 | $\text{m}$ | $x_G' = x_G / L = 0.0346367$ (or $0.0348$) |
| Moment of inertia about Z axis | $I_z$ | $2.00 \times 10^{12}$ | $\text{kg}\cdot\text{m}^2$ (printed as $\text{kg/m}^2$) | $I_z' = I_z / (\frac{1}{2}\rho L^5) = 0.001198626$ ($\rho = 1000 \text{ kg/m}^3$) |
| Rudder turning rate | $\dot{\delta}_{max}$ | 2.34 | $^\circ/\text{s}$ | $0.0408407 \text{ rad/s}$ |
| Approach speed | $u_0$ | 7.956 | $\text{m/s}$ | $1.0$ (initial surge speed) |
| Ship mass | $m$ | *Unspecified in Table 1* | $\text{kg}$ | $m' = m / (\frac{1}{2}\rho L^3) = 0.019080625$ (derived from $C_b$ and data audit) |

---

## 2. Coordinate System and Mathematical Model

### Coordinate Reference Systems (Page 3, Figure 1)
- Earth-fixed coordinate system: $O_e - X_e Y_e Z_e$
- Body-fixed coordinate system: $O_b - X_b Y_b Z_b$ ($O_b$ at midship)
- Motion variables:
  - $u$: surge velocity ($\text{m/s}$)
  - $v$: sway velocity ($\text{m/s}$)
  - $r$: yaw rate ($\text{rad/s}$)
  - $\delta$: rudder angle ($\text{rad}$)
  - $\psi$: heading angle ($\text{rad}$)
  - $\dot{u}, \dot{v}$: linear accelerations ($\text{m/s}^2$)
  - $\dot{r}$: yaw acceleration ($\text{rad/s}^2$)

### Equations of Motion (Page 3, Equation 1)
Newton's 3-DOF equations of motion:
$$\begin{aligned}
m(\dot{u} - vr - x_G r^2) &= X \\
m(\dot{v} + ur + x_G \dot{r}) &= Y \\
I_z \dot{r} + m x_G (\dot{v} + ur) &= N
\end{aligned} \quad \text{(Equation 1)}$$

### Abkowitz-Type Hydrodynamic Force Model (Page 3, Equation 2 & Page 4, Equation 4)
Forces and yaw moment decomposed into acceleration derivatives and velocity/rudder terms:
$$\begin{aligned}
(m - X_{\dot{u}})\dot{u} &= f_1 \\
(m - Y_{\dot{v}})\dot{v} + (m x_G - Y_{\dot{r}})\dot{r} &= f_2 \\
(m x_G - N_{\dot{v}})\dot{v} + (I_z - N_{\dot{r}})\dot{r} &= f_3
\end{aligned} \quad \text{(Equation 3)}$$

where the non-acceleration hydrodynamic functions $f_1, f_2, f_3$ are (Page 4, Equation 4):
$$\begin{aligned}
f_1 = & \; X_u \Delta u + X_{uu}(\Delta u)^2 + X_{vv}v^2 + (m x_G + X_{rr})r^2 + (m + X_{vr})vr + X_{\delta\delta}\delta^2 + X_{v\delta}v\delta + X_{r\delta}r\delta \\
f_2 = & -m u r + Y_v v + Y_{vvv} v^3 + Y_r r + Y_{rrr} r^3 + Y_\delta \delta + Y_{\delta\delta}\delta^2 + Y_{\delta\delta\delta}\delta^3 \\
      & + Y_{vrr} v r^2 + Y_{vvr} v^2 r + Y_{u\delta} \Delta u \delta + Y_{v\delta\delta} v \delta^2 + Y_{vv\delta} v^2 \delta + Y_{r\delta\delta} r \delta^2 + Y_{rr\delta} r^2 \delta \\
f_3 = & -m x_G u r + N_v v + N_{vvv} v^3 + N_r r + N_{rrr} r^3 + N_\delta \delta + N_{\delta\delta}\delta^2 + N_{\delta\delta\delta}\delta^3 \\
      & + N_{vrr} v r^2 + N_{vvr} v^2 r + N_{u\delta} \Delta u \delta + N_{v\delta\delta} v \delta^2 + N_{vv\delta} v^2 \delta + N_{r\delta\delta} r \delta^2 + N_{rr\delta} r^2 \delta
\end{aligned} \quad \text{(Equation 4)}$$
*(Note: Visual inspection of page 4 confirms the presence of $Y_{\delta\delta}\delta^2$ and $N_{\delta\delta}\delta^2$, matching Table 2).*

### Explicit Acceleration Equations (Page 4, Equation 5)
Solving the linear algebraic system for accelerations:
$$\begin{aligned}
\dot{u} &= \frac{f_1}{m - X_{\dot{u}}} \\
\dot{v} &= \frac{(I_z - N_{\dot{r}})f_2 - (m x_G - Y_{\dot{r}})f_3}{(m - Y_{\dot{v}})(I_z - N_{\dot{r}}) - (m x_G - Y_{\dot{r}})(m x_G - N_{\dot{v}})} \\
\dot{r} &= \frac{(m - Y_{\dot{v}})f_3 - (m x_G - N_{\dot{v}})f_2}{(m - Y_{\dot{v}})(I_z - N_{\dot{r}}) - (m x_G - Y_{\dot{r}})(m x_G - N_{\dot{v}})}
\end{aligned} \quad \text{(Equation 5)}$$

### Euler Discretization Scheme (Page 4, Equation 6)
Time step $\Delta t = 1.0 \text{ s}$:
$$\begin{aligned}
u(t+1) &= u(t) + \dot{u}(t) \Delta t \\
v(t+1) &= v(t) + \dot{v}(t) \Delta t \\
r(t+1) &= r(t) + \dot{r}(t) \Delta t \\
\psi(t+1) &= \psi(t) + r(t) \Delta t \\
x(t+1) &= x(t) + (u(t)\cos\psi(t) - v(t)\sin\psi(t))\Delta t \\
y(t+1) &= y(t) + (u(t)\sin\psi(t) + v(t)\cos\psi(t))\Delta t
\end{aligned}$$

---

## 3. Non-Dimensionalization (Prime-I System Verification)

As verified against the workspace dataset, the dimensionless coefficients in Table 2 are defined using the standard Prime-I system with $U = u$ (instantaneous surge velocity) and $L = 320.0 \text{ m}$:
- Non-dimensional states:
  $u' = u / u = 1.0$, $v' = v / u$, $r' = r L / u$, $\Delta u' = (u - u_0)/u$, $\delta' = \delta \text{ (rad)}$
- Mass parameters:
  $m' = 0.019080625$, $x_G' = 0.034636727$, $I_z' = 0.001198626$
- Acceleration conversions:
  $\dot{u} = \dot{u}' \cdot u^2 / L$, $\dot{v} = \dot{v}' \cdot u^2 / L$, $\dot{r} = \dot{r}' \cdot u^2 / L^2$

The non-dimensional acceleration equations in Prime-I form:
$$\begin{aligned}
\dot{u}' &= \frac{f_1'}{m' - X_{\dot{u}}'} \\
\dot{v}' &= \frac{(I_z' - N_{\dot{r}}')f_2' - (m' x_G' - Y_{\dot{r}}')f_3'}{(m' - Y_{\dot{v}}')(I_z' - N_{\dot{r}}') - (m' x_G' - Y_{\dot{r}}')(m' x_G' - N_{\dot{v}}')} \\
\dot{r}' &= \frac{(m' - Y_{\dot{v}}')f_3' - (m' x_G' - N_{\dot{v}}')f_2'}{(m' - Y_{\dot{v}}')(I_z' - N_{\dot{r}}') - (m' x_G' - Y_{\dot{r}}')(m' x_G' - N_{\dot{v}}')}
\end{aligned}$$
This reproduces the raw dataset accelerations across all training files with error $< 10^{-5}$.

---

## 4. Hydrodynamic Derivatives

### Fixed Acceleration Derivatives (Page 7, left column)
These 5 parameters are known and fixed during identification:
- $X_{\dot{u}}' = -0.00114$
- $Y_{\dot{v}}' = -0.0145$
- $Y_{\dot{r}}' = -0.00121$
- $N_{\dot{v}}' = -0.000588$
- $N_{\dot{r}}' = -0.000564$

### Trainable Hydrodynamic Derivatives (36 Parameters, Page 7, Table 2)
Initial values for all 36 derivatives: set to $0.0$ (Page 6, Section 4.2).

| Group | Parameter | Original (Yao et al. 2021) | Identified by Paper PINN |
|---|---|---|---|
| **X-Coef (8)** | $X_u'$ | -0.00319 | -0.0032 |
| | $X_{uu}'$ | 0.00199 | 0.0020 |
| | $X_{vv}'$ | 0.00159 | -0.0028 |
| | $X_{rr}'$ | 0.000338 | -0.0009 |
| | $X_{vr}'$ | 0.01391 | 0.0092 |
| | $X_{\delta\delta}'$ | -0.00272 | -0.0027 |
| | $X_{v\delta}'$ | 0.001609 | 0.0008 |
| | $X_{r\delta}'$ | -0.001034 | -0.0015 |
| **Y-Coef (14)** | $Y_v'$ | -0.01902 | -0.0193 |
| | $Y_{vvv}'$ | -0.1287 | -0.1024 |
| | $Y_r'$ | 0.005719 | 0.0056 |
| | $Y_{rrr}'$ | -0.000048 | 0.000045 |
| | $Y_{vrr}'$ | -0.02429 | -0.0167 |
| | $Y_{vvr}'$ | 0.0211 | 0.0493 |
| | $Y_\delta'$ | 0.00408 | 0.0040 |
| | $Y_{\delta\delta}'$ | -0.000114 | -0.0001 |
| | $Y_{\delta\delta\delta}'$ | -0.003059 | -0.0029 |
| | $Y_{u\delta}'$ | -0.00456 | -0.0046 |
| | $Y_{v\delta\delta}'$ | 0.00326 | 0.0050 |
| | $Y_{vv\delta}'$ | 0.003018 | 0.0053 |
| | $Y_{r\delta\delta}'$ | -0.002597 | -0.0017 |
| | $Y_{rr\delta}'$ | 0.000895 | 0.0003 |
| **N-Coef (14)** | $N_v'$ | -0.007886 | -0.0078 |
| | $N_{vvv}'$ | 0.00175 | 0.0102 |
| | $N_r'$ | -0.003701 | -0.0037 |
| | $N_{rrr}'$ | -0.000707 | -0.0011 |
| | $N_{vrr}'$ | 0.003726 | 0.0054 |
| | $N_{vvr}'$ | -0.019 | -0.0100 |
| | $N_\delta'$ | -0.001834 | -0.0018 |
| | $N_{\delta\delta}'$ | -0.000056 | -0.000057 |
| | $N_{\delta\delta\delta}'$ | 0.001426 | 0.0014 |
| | $N_{u\delta}'$ | 0.00232 | 0.0023 |
| | $N_{v\delta\delta}'$ | -0.001504 | -0.0014 |
| | $N_{vv\delta}'$ | -0.001406 | 0.0001 |
| | $N_{r\delta\delta}'$ | 0.001191 | 0.0011 |
| | $N_{rr\delta}'$ | -0.000398 | -0.0010 |

---

## 5. Neural Network Architecture and Training Specification

### Architecture (Page 4, Section 3.1 & Page 6, Section 4.2, Figure 2)
- **Model Type:** Feedforward Neural Network (Multilayer Perceptron)
- **Inputs (4):** $u$ (surge velocity), $v$ (sway velocity), $r$ (yaw rate), $\delta$ (rudder angle)
- **Hidden Layers:** 3 fully connected hidden layers, 10 neurons per hidden layer
- **Activation Function:** Hyperbolic tangent ($\text{Tanh}$) on all hidden layers
- **Outputs (3):** $\dot{u}, \dot{v}, \dot{r}$ (accelerations)
- **Weight & Bias Initialization:** Xavier initialization (Glorot and Bengio 2010)
- **Trainable Parameters:**
  - Network weights and biases: $(4 \times 10 + 10) + (10 \times 10 + 10) + (10 \times 10 + 10) + (10 \times 3 + 3) = 50 + 110 + 110 + 33 = 303$ weights and biases.
  - Hydrodynamic derivatives: 36 scalar parameters.
  - Total trainable parameters: $303 + 36 = 339$.

### Optimizer & Schedule (Page 4, Equation 8 & Page 7, Section 4.2)
- **Optimizer:** Adam
- **Total Iterations:** 120,000 iterations
- **Learning Rate Schedule (Equation 8):**
  $$lr = 0.001 \cdot (0.1)^{\lfloor \frac{\text{iter}}{40000} \rfloor}$$
  - Iterations $0 \le \text{iter} < 40,000$: $lr = 1.0 \times 10^{-3}$
  - Iterations $40,000 \le \text{iter} < 80,000$: $lr = 1.0 \times 10^{-4}$
  - Iterations $80,000 \le \text{iter} \le 120,000$: $lr = 1.0 \times 10^{-5}$

### Data Preprocessing (Page 6, Equation 19)
Standardization formula:
$$x' = \frac{x - \mu}{\sigma}$$
Applied to each feature variable across training data.

---

## 6. Loss Function and Dynamic Weighting Strategy

### Formulation (Page 5, Equations 15–18)
Total loss:
$$Loss = Loss_{data} + \beta \cdot Loss_{phy} \quad \text{(Equation 18)}$$

1. **Observation Data Loss (Equation 17):**
   $$Loss_{data} = \frac{1}{N_o} \sum_{i=1}^{N_o} (|\dot{u}_{data, i} - \dot{u}_i|^2 + |\dot{v}_{data, i} - \dot{v}_i|^2 + |\dot{r}_{data, i} - \dot{r}_i|^2)$$

2. **Physics Residuals (Equation 15):**
   $$e_1 = \left| \dot{u}_{phy} - \dot{u} \right|, \quad e_2 = \left| \dot{v}_{phy} - \dot{v} \right|, \quad e_3 = \left| \dot{r}_{phy} - \dot{r} \right|$$
   where $\dot{u}, \dot{v}, \dot{r}$ are neural network predictions and $\dot{u}_{phy}, \dot{v}_{phy}, \dot{r}_{phy}$ are evaluated from Equation (5) using the 36 current hydrodynamic derivative parameters.

3. **Physics-based Loss (Equation 16):**
   $$Loss_{phy} = \frac{1}{N_o} \sum_{i=1}^{N_o} (e_{1, i}^2 + e_{2, i}^2 + e_{3, i}^2)$$

4. **Dynamic Loss Weight Schedule $\beta$ (Page 5, right column):**
   - $0 \le \text{iter} \le 10,000$: $\beta = 0.01$
   - $10,000 < \text{iter} \le 50,000$: $\beta = 0.1$
   - $50,000 < \text{iter} \le 120,000$: $\beta = 1.0$

5. **Constant Weight Ablation (Page 7, left column):**
   - $\beta = 1.0$ for all 120,000 iterations.

---

## 7. Baseline Neural Network Definition (T4)

Reference: Page 12, Section 5.3.
- "In addition to the PINN, a neural network (NN) without embedded physical constraints is used for comparison. This model shares the same parameter settings and training process as the PINN."
- Same architecture (4 inputs, 3 hidden layers of 10 neurons, Tanh, 3 outputs), Adam optimizer, lr schedule.
- Trained strictly on $Loss_{data}$ ($\beta = 0$), with no hydrodynamic parameters.

---

## 8. Evaluation Metrics and Paper-Reported Results

### Evaluation Protocol (Page 9, Section 4.4)
Given the initial state $(u_0, v_0, r_0, \psi_0 = 0, x_0 = 0, y_0 = 0)$ and the test rudder time history $\delta(t)$:
1. **Approach 1 (PINN):** At each step $t$, the neural network predicts $[\dot{u}, \dot{v}, \dot{r}]$ from $[u(t), v(t), r(t), \delta(t)]$. Forward Euler integrates state to $t+1$.
2. **Approach 2 (Identified Model):** At each step $t$, Equation (5) evaluates $[\dot{u}, \dot{v}, \dot{r}]$ using the 36 identified derivatives. Forward Euler integrates state to $t+1$.
3. **Approach 3 (Data-only NN):** Baseline neural network predicts $[\dot{u}, \dot{v}, \dot{r}]$ recursively.

### Metric: Root Mean Square Error (RMSE) (Page 9, Equation 28)
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2}$$

### Paper-Reported Simulation Test Results (Page 10, Table 3)

| Maneuver | Model | $u$ (m/s) | $v$ (m/s) | $r$ (deg/s or rad/s)* | Sum |
|---|---|---|---|---|---|
| **$25^\circ/5^\circ$ Zigzag** | PINN | 0.0398 | 0.0909 | 0.0419 | 0.1726 |
| | Identified | 0.0471 | 0.0669 | 0.0327 | 0.1467 |
| **$20^\circ/10^\circ$ Zigzag** | PINN | 0.0130 | 0.0145 | 0.0063 | 0.0338 |
| | Identified | 0.0281 | 0.0186 | 0.0083 | 0.0550 |
| **$30^\circ$ Turning Circle** | PINN | 0.0508 | 0.0041 | 0.0013 | 0.0562 |
| | Identified | 0.0376 | 0.0070 | 0.0025 | 0.0471 |

*\*Note on $r$ RMSE units:* The paper's text notes $r$ plotted in $\text{deg/s}$ (Figures 6–8); whether Table 3 $r$ is in $\text{rad/s}$ or $\text{deg/s}$ is analyzed below.

---

## 9. Implementation-Critical Details, Ambiguities, and Questions

### Ambiguity 1: Loss Calculation — Physical vs Standardized Units
- **Evidence:** Equation (17) writes $Loss_{data} = \frac{1}{N_o} \sum (|\dot{u}_{data} - \dot{u}|^2 + \dots)$.
  Figure 3(a) shows $Loss_{data}$ starting at $10^0 = 1$ and converging to $10^{-4}$.
  In unnormalized physical units, raw $|\dot{u}_{data}|^2 \sim 10^{-4}$ and $|\dot{r}_{data}|^2 \sim 10^{-8}$, so unnormalized initial loss would be $< 10^{-3}$.
  Figure 2 shows the network outputting $\dot{\bar{u}}, \dot{\bar{v}}, \dot{\bar{r}}$ (standardized accelerations).
- **Resolution for Implementation:** Standardize inputs and target accelerations by training set mean and standard deviation. The neural network predicts standardized accelerations. For $Loss_{phy}$, unstandardize network outputs to physical accelerations, compute physical residuals $e_1, e_2, e_3$, and normalize residuals by acceleration variances $\sigma_{\dot{u}}^2, \sigma_{\dot{v}}^2, \sigma_{\dot{r}}^2$ so that $Loss_{phy}$ and $Loss_{data}$ reside on the same dimensionless order-1 scale as shown in Figure 3.

### Ambiguity 2: Batch Size During Optimization
- **Evidence:** Equations (16) and (17) sum over $i = 1 \dots N_o$, where $N_o = 3000$ (total training points).
  Page 7 states training takes 120,000 iterations and finishes in ~8 minutes on an Intel Core i7 CPU.
  120,000 full-batch iterations in 8 min is ~250 iterations/s.
- **Specification:** Full-batch Adam optimization over all $N_o = 2997$ points, exactly matching Equations (16) and (17).

### Ambiguity 3: RMSE Units for $r$ in Table 3
- **Evidence:** In Table 3, $r$ RMSE for $25^\circ/5^\circ$ zigzag is $0.0419$.
  In `zigzag_25_5.txt`, the range of $r$ is $[-0.009, +0.009] \text{ rad/s}$ ($[-0.52^\circ/s, +0.52^\circ/s]$).
  If $r$ RMSE was $0.0419 \text{ rad/s}$, the error would be $4\times$ larger than the maximum signal amplitude!
  However, in $\text{deg/s}$, signal amplitude is $\approx 0.52 \text{ deg/s}$, and an RMSE of $0.0419 \text{ deg/s}$ represents $\sim 8\%$ relative error, perfectly matching the visual fit in Figure 6.
- **Specification:** Compute and report RMSE in both $\text{rad/s}$ and $\text{deg/s}$ for transparent comparison against Table 3.
