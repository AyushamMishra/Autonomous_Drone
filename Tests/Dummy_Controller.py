class DummyController:

    def __init__(self):
        self.compute_calls = 0

    def compute_control(self, state, path):
        self.compute_calls += 1
        return {"v": 1.0, "omega": 0.0}