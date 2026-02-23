class DummyBehavior:

    def __init__(self):
        self.update_calls = 0

    def update(self, state, perception_output):
        self.update_calls += 1
        return (10, 10)