print("state.py loaded")

class State:
    def __init__(self, x, y, yaw, g=0.0, parent=None):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.g = g
        self.parent = parent

    def __repr__(self):
        return f"State(x={self.x}, y={self.y}, yaw={self.yaw}, g={self.g:.2f})"

