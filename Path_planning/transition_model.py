import numpy as np

class TransitionEvaluator:
    def __init__(self, costmap, yaw_angles):
        self.costmap = costmap
        self.yaw_angles = yaw_angles
        self.H = costmap.shape[1]
        self.W = costmap.shape[2]
        self.Y = len(yaw_angles)

    def propagate(self, state, primitive):
        dx, dy, dyaw = primitive

        yaw = state.yaw
        theta = self.yaw_angles[yaw]

        nx = int(round(state.x + dx * np.cos(theta) - dy * np.sin(theta)))
        ny = int(round(state.y + dx * np.sin(theta) + dy * np.cos(theta)))
        nyaw = (yaw + dyaw) % self.Y

        # bounds
        if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
            return None

        cost = self.costmap[nyaw, ny, nx]
        if not np.isfinite(cost):
            return None

        return nx, ny, nyaw, cost
