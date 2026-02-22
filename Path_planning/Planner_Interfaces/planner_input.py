from Path_planning.state import State
class PlannerInput:
    """
    Standard container for all planning-related inputs.
    This prevents direct coupling between modules.
    """

    def __init__(
        self,
        start_state: State,
        goal_state: State,
        costmap_3d,
        yaw_angles,
        obstacle_threshold,
        global_path=None,
        corridor_mask=None,
        cost_to_goal=None,
    ):
        """
        start_state        : State
        goal_state         : State
        costmap_3d         : [Y, H, W] numpy array
        yaw_angles         : list/array of yaw angles
        obstacle_threshold : float
        global_path        : list[(x,y)]
        corridor_mask      : [H,W] bool mask
        cost_to_goal       : [H,W] heuristic grid
        """

        self.start_state = start_state
        self.goal_state = goal_state

        self.costmap_3d = costmap_3d
        self.costmap_2d = costmap_3d[0]   # your global planners use slice 0

        self.yaw_angles = yaw_angles
        self.obstacle_threshold = obstacle_threshold

        self.global_path = global_path
        self.corridor_mask = corridor_mask
        self.cost_to_goal = cost_to_goal

    def has_global_path(self):
        return self.global_path is not None

    def has_heuristic(self):
        return self.cost_to_goal is not None

    def has_corridor(self):
        return self.corridor_mask is not None