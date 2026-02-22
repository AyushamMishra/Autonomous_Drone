class PlannerOutput:
    """
    Standard container for planner outputs.
    """

    def __init__(
        self,
        global_path=None,
        local_path=None,
        executed_trajectory=None,
        cost_to_goal=None,
        status="UNKNOWN",
        planning_time=0.0,
    ):
        """
        global_path         : list[(x,y)]
        local_path          : list[State]
        executed_trajectory : list[State]
        cost_to_goal        : [H,W] grid
        status              : str
        planning_time       : float (seconds)
        """

        self.global_path = global_path
        self.local_path = local_path
        self.executed_trajectory = executed_trajectory
        self.cost_to_goal = cost_to_goal

        self.status = status
        self.planning_time = planning_time

    def success(self):
        return self.status == "SUCCESS"

    def failed(self):
        return self.status == "FAILED"