Dataset for the paper:

Identification modeling of ship maneuvering motion based on physics-informed neural network (PINN)

Authors:
Yi Hu et al.

Description
This dataset contains the simulation data used to train and evaluate the physics-informed neural network (PINN) model for ship maneuvering motion modeling.

The simulation data were generated using the Abkowitz-type ship maneuvering model for the KVLCC2 benchmark ship.

Simulation dataset structure
The simulation dataset is divided into two parts:

1. Training dataset
2. Test dataset

Training dataset
The training dataset is used to train the PINN model.

It includes maneuvering motion data from the following maneuvers:

- Turning circle maneuver(35)
- Zigzag maneuver (10/10)
- Zigzag maneuver (20/20)

Each maneuver was simulated for 1000 seconds with a sampling interval of 1 second.

Test dataset
The test dataset is used to evaluate the generalization performance of the trained model.

The test dataset includes maneuvering motions that were not used during the training stage:

- Turning circle maneuver (30)
- Zigzag maneuver (25/5)
- Zigzag maneuver (20/10)

Variables included in the dataset

u             surge velocity (m/s)
v             sway velocity (m/s)
r             yaw rate (rad/s)
rudder_angle  rudder angle (rad)
heading_angle heading_angle (rad)
du_dt         surge acceleration
dv_dt         sway acceleration
dr_dt         yaw acceleration
x_position  nondimensional longitudinal position (normalized by ship length L)
y_position  nondimensional lateral position (normalized by ship length L)

Experimental data
The experimental validation data used in the paper are obtained from the SIMMAN 2008 workshop database.

SIMMAN database:
http://www.simman2008.dk/

These experimental data are not included in this repository. Users should obtain them directly from the official SIMMAN database.

Purpose
The dataset is provided to support the reproducibility of the numerical experiments reported in the paper.
