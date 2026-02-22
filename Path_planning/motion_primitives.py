from Path_planning.state import State
import numpy as np

class MotionPrimitives:
    def __init__(self, step=1.0, max_steer=np.deg2rad(30), num_yaws=16):
        self.step = step
        self.num_yaws = num_yaws

        self.steers = [-max_steer, -max_steer/2, 0.0, max_steer/2, max_steer]
        self.yaw_res = 2*np.pi / num_yaws

    def primitives(self):
        """
        Returns list of (dx, dy, dyaw_index)
        Motion in local vehicle frame
        """
        motions = []

        for delta in self.steers:
            dx = self.step
            dy = 0.0

            dyaw_index = int(round(delta / self.yaw_res))

            motions.append((dx, dy, dyaw_index))

        return motions