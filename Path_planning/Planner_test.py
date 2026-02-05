from Locallization.map_server import Map_Server
from Path_planning.Costmap_builder import CostmapBuilder
from Path_planning.motion_primitives import MotionPrimitives
from Path_planning.transition_model import TransitionEvaluator as TransitionModel
from Path_planning.state import State
from Path_planning.State_expander import StateExpander

def main():
    # Load map
    map_server = Map_Server(
        dem_path=r"D:\Autonomous_Drone\Search_map\cdnh43w\cdnh43w_utm.tif"
    )

    # Build costmap
    cost_builder = CostmapBuilder(map_server)
    costmap = cost_builder.build(
        sigma_x=2.0,
        sigma_y=1.5,
        vehicle_length=2.0,
        vehicle_width=0.5
    )

    # Planning components
    primitives = MotionPrimitives(step_cells=1)
    transition = TransitionModel(costmap, cost_builder.yaw_angles)
    expander = StateExpander(primitives, transition)

    # Initial state
    s0 = State(x=120, y=80, yaw=4)

    # Expand
    neighbors = expander.expand(s0)

    print(f"Expanded {len(neighbors)} states:")
    for s in neighbors:
        print(s)

if __name__ == "__main__":
    main()
