import numpy as np

class MotionPrimitives:
    def __init__(self, step_cells=1):
        self.step = step_cells

    def primitives(self):
        """
        dx, dy, dyaw (dy relative to heading frame)
        """
        return [
            (self.step, 0,  0),   # straight
            (self.step, 0,  1),   # slight left
            (self.step, 0, -1),   # slight right
        ]
