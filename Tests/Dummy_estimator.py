class DummyEstimator:

    def __init__(self):
        self.predict_calls = 0
        self.update_calls = 0
        self.state = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def predict(self, sensor_data):
        self.predict_calls += 1

    def update(self, sensor_data):
        self.update_calls += 1

    def get_state(self):
        return self.state