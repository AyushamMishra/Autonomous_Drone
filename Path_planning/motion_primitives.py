from sre_parse import State
import numpy as np

class MotionPrimitives:
    def __init__(self, step=2.5, max_steer=np.deg2rad(30)):
        self.step = step
        self.steers = [-max_steer, -max_steer/2, 0.0, max_steer/2, max_steer]

    def primitives(self,s):
        """
        dx, dy, dyaw (dy relative to heading frame)
        """
        states = []
        for delta in self.steers:
            yaw = s.yaw + delta
            x = s.x + self.step * np.cos(yaw)
            y = s.y + self.step * np.sin(yaw)

            nxt = State(x, y, yaw)
            nxt.g = s.g + self.step
            states.append(nxt)
        return states
