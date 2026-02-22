import numpy as np

from Path_planning import state

class TransitionEvaluator:
    def __init__(self, costmap, yaw_angles):
        self.costmap = costmap
        self.yaw_angles = yaw_angles
        self.H = costmap.shape[1]
        self.W = costmap.shape[2]
        self.Y = len(yaw_angles)

    def propagate(self, state, primitive):

        dx, dy, dyaw = primitive

        # Convert continuous yaw to index
        yaw_idx = int(round((state.yaw + np.pi) / (2*np.pi) * self.Y)) % self.Y

        theta = self.yaw_angles[yaw_idx]

        nx = int(round(state.x + dx * np.cos(theta) - dy * np.sin(theta)))
        ny = int(round(state.y + dx * np.sin(theta) + dy * np.cos(theta)))
        nyaw = (yaw_idx + dyaw) % self.Y

        if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
            return None

        cost = self.costmap[nyaw, ny, nx]

        if not np.isfinite(cost):
            return None

        return nx, ny, nyaw, cost
