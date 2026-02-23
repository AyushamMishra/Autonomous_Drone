class DummyPerception:

    def __init__(self):
        self.process_calls = 0

    def process(self, sensor_data):
        self.process_calls += 1
        return {"detected": False}