class DummyPlanner:

    def __init__(self):
        self.plan_calls = 0

    def set_goal(self, goal):
        self.goal = goal

    def needs_replan(self):
        return True

    def plan(self, state):
        self.plan_calls += 1
        return [(0, 0), (1, 1)]