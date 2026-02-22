import time
from Path_planning.Responsive_replanner import DStarLite
from Path_planning.LocalPath_planner import HybridAStarPlanner
from Path_planning.GlobalPath_planner import AStarGlobalPlanner
from Path_planning.Planner_Interfaces.planner_output import PlannerOutput


class PlannerManager:
    """
    Orchestrates:
    - Global A*
    - D* Lite (responsive replanning)
    - Hybrid A* local planning
    """

    def __init__(
        self,
        global_planner,
        responsive_replanner,
        local_planner,
    ):
        """
        global_planner       : AStarGlobalPlanner instance
        responsive_replanner : DStarLite instance
        local_planner        : HybridAStarPlanner instance
        """

        self.global_planner = global_planner
        self.responsive_replanner = responsive_replanner
        self.local_planner = local_planner

        self.initialized = False

    # ------------------------------------------------------
    # INITIAL GLOBAL PLAN
    # ------------------------------------------------------
    def initialize_global_plan(self, planner_input):
        """
        Runs initial A* and initializes D* Lite.
        """

        start_xy = (int(planner_input.start_state.x),
                    int(planner_input.start_state.y))

        goal_xy = (int(planner_input.goal_state.x),
                   int(planner_input.goal_state.y))

        # Run A*
        global_path = self.global_planner.global_plan(start_xy, goal_xy)

        if global_path is None:
            return None

        # Initialize D* Lite
        self.responsive_replanner.initialize(
            planner_input.costmap_2d,
            start_xy,
            goal_xy
        )

        self.initialized = True
        return global_path

    # ------------------------------------------------------
    # UPDATE COSTMAP (for D*)
    # ------------------------------------------------------
    def update_costmap(self, new_costmap_2d):
        """
        Update D* with changed costs.
        """
        self.responsive_replanner.update_costmap(new_costmap_2d)

    # ------------------------------------------------------
    # LOCAL PLANNING STEP
    # ------------------------------------------------------
    def compute_local_plan(self, planner_input):
        """
        Calls Hybrid A* using available heuristics.
        """

        local_path = self.local_planner.plan(
            planner_input.start_state,
            planner_input.goal_state,
            planner_input.global_path
        )

        return local_path

    # ------------------------------------------------------
    # MAIN ENTRY FUNCTION
    # ------------------------------------------------------
    def step(self, planner_input):
        """
        Main planning loop.
        Returns PlannerOutput.
        """

        t0 = time.time()

        # 1️⃣ If not initialized → run initial global planning
        if not self.initialized:

            global_path = self.initialize_global_plan(planner_input)

            if global_path is None:
                return PlannerOutput(status="FAILED")

            planner_input.global_path = global_path

        # 2️⃣ Run D* to update cost-to-go map
        cost_to_goal = self.responsive_replanner.compute_shortest_path()

        planner_input.cost_to_goal = cost_to_goal

        # 3️⃣ Run local planner (Hybrid A*)
        local_path = self.compute_local_plan(planner_input)

        if local_path is None:
            return PlannerOutput(status="FAILED")

        planning_time = time.time() - t0

        return PlannerOutput(
            global_path=planner_input.global_path,
            local_path=local_path,
            cost_to_goal=cost_to_goal,
            status="SUCCESS",
            planning_time=planning_time
        )

    # ------------------------------------------------------
    # RESET SYSTEM
    # ------------------------------------------------------
    def reset(self):
        self.initialized = False