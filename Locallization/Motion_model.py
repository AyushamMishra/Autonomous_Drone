
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


import numpy as np
sigma_v = 0.5       #  velocity noise
sigma_yaw =0.01     # yaw noise

class MotionModel:
    def __init__(self, sigma_v, sigma_yaw):
        self.sigma_v = sigma_v
        self.sigma_yaw = sigma_yaw

    def propogate(self,state,control_inputs,dt):

        v,yaw_rate= control_inputs

        N=state.shape[0]  # number of particles

        ## Adding noise to control inputs
        v_noise= np.random.normal(0,self.sigma_v,N)
        yaw_noise= np.random.normal(0,self.sigma_yaw,N)

        v_eff=v+v_noise
        yaw=state[:,2]

        ## Kinematic updates for (x,y) coordinates

        state[:,0] += v_eff * np.cos(yaw) * dt
        state[:,1] += v_eff * np.sin(yaw) * dt
        state[:,2] += yaw_rate * dt + yaw_noise

        ## Normalize yaw to [-pi, pi]
        state[:,2] = (state[:,2] + np.pi) % (2 * np.pi) - np.pi

        return state