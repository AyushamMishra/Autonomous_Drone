
"""
Assumptions for motion model :
1. 2D horizontal movement(x,y) with yaw as heading direction.
2. no roll and pitch changes.
3. Matched to UTM CRS.

forward velocity = v m/s
yaw rate = omega rad/s

state= [x, y, yaw]
control inputs= [v, omega]
dt = time interval between steps

"""


"""
Motion Model
State      : [x, y, yaw]
Control    : [v, omega]
Units      : meters, radians, seconds
"""

import numpy as np

class MotionModel:
    def __init__(self, sigma_v=0.5, sigma_yaw=0.01, sigma_omega=0.01):
        self.sigma_v = sigma_v
        self.sigma_yaw = sigma_yaw
        self.sigma_omega = sigma_omega

    def propagate(self, state, control_inputs, dt, add_noise=True):

        v_cmd, omega_cmd = control_inputs
        N = state.shape[0]

        state_new = state.copy()
        yaw = state[:, 2]

        if add_noise:
            v_noise = np.random.normal(0, self.sigma_v, N)
            omega_noise = np.random.normal(0, self.sigma_omega, N)
            yaw_noise = np.random.normal(0, self.sigma_yaw, N)
        else:
            v_noise = 0.0
            omega_noise = 0.0
            yaw_noise = 0.0

        # Effective controls
        v_eff = v_cmd + v_noise
        omega_eff = omega_cmd + omega_noise

        # Kinematic propagation
        state_new[:, 0] += v_eff * np.cos(yaw) * dt
        state_new[:, 1] += v_eff * np.sin(yaw) * dt
        state_new[:, 2] += omega_eff * dt + yaw_noise

        # Normalize yaw to [-pi, pi]
        state_new[:, 2] = (state_new[:, 2] + np.pi) % (2 * np.pi) - np.pi

        return state_new
