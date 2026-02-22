import numpy as np
import matplotlib.pyplot as plt

from Locallization.map_server import Map_Server
from Locallization import DEM_reprojection

from Path_planning import motion_primitives
from Path_planning.Costmap_builder import CostmapBuilder
from Path_planning.state import State
from Path_planning.GlobalPath_planner import AStarGlobalPlanner
from Path_planning.Responsive_replanner import DStarLite
from Path_planning.LocalPath_planner import HybridAStarPlanner

from Path_planning.Planner_Interfaces.planner_input import PlannerInput
from Path_planning.planner_manager import PlannerManager
from Path_planning.State_expander import StateExpander
from Path_planning.motion_primitives import MotionPrimitives
from Path_planning.transition_model import TransitionEvaluator


def planner_test():

    print("🚀 FULL STACK PLANNER TEST")

    # ----------------------------------------------------
    # 1️⃣ LOAD DEM
    # ----------------------------------------------------
    map_server = Map_Server(DEM_reprojection.output_dem)

    r0, c0 = 2050, 25
    crop_size = 40

    dem_crop = map_server.dem[r0:r0+crop_size,
                              c0:c0+crop_size].filled(np.nan)

    print(f"✅ DEM crop: {dem_crop.shape}")

    class CropMap:
        def __init__(self, dem):
            self.dem = dem
            self.res_x = self.res_y = 29.0
            self.height, self.width = dem.shape

    crop_map = CropMap(dem_crop)

    # ----------------------------------------------------
    # 2️⃣ BUILD COSTMAP
    # ----------------------------------------------------
    print("🗺️ Building costmap...")
    cost_builder = CostmapBuilder(crop_map)
    costmap_3d = cost_builder.build(w_slope=0.2, w_rough=0.1)

    Y, H, W = costmap_3d.shape
    print(f"✅ Costmap: {Y} yaw layers | {H}x{W}")

    # ----------------------------------------------------
    # 3️⃣ DEFINE START / GOAL
    # ----------------------------------------------------
    start_px = (8, 8)
    goal_px = (32, 32)

    start_state = State(start_px[0], start_px[1], yaw=0)
    goal_state = State(goal_px[0], goal_px[1], yaw=0)


    # ----------------------------------------
    # Build motion + transition
    # ----------------------------------------

    yaw_angles = np.linspace(0, 2*np.pi, Y, endpoint=False)

    motion_primitives = MotionPrimitives()

    transition_model = TransitionEvaluator(costmap_3d,yaw_angles)

    state_expander = StateExpander(motion_primitives,transition_model)
    # ----------------------------------------------------
    # 4️⃣ INITIALIZE PLANNERS
    # ----------------------------------------------------
    print("⚙️ Initializing planners...")

    global_planner = AStarGlobalPlanner(
        costmap_3d[0],
        obstacle_threshold=3.0
    )

    dstar = DStarLite()
    local_planner = HybridAStarPlanner(costmap=costmap_3d,yaw_angles=yaw_angles,state_expander=state_expander)

    planner_manager = PlannerManager(
        global_planner,
        dstar,
        local_planner
    )

    # ----------------------------------------------------
    # 5️⃣ BUILD PLANNER INPUT
    # ----------------------------------------------------
    planner_input = PlannerInput(
        start_state=start_state,
        goal_state=goal_state,
        costmap_3d=costmap_3d,
        yaw_angles=np.linspace(0, 2*np.pi, Y, endpoint=False),
        obstacle_threshold=3.0
    )

    # ----------------------------------------------------
    # 6️⃣ RUN FULL PIPELINE
    # ----------------------------------------------------
    print("🧠 Running integrated planner...")
    output = planner_manager.step(planner_input)

    if not output.success():
        print("❌ Planning failed.")
        return False

    print(f"✅ Planning success | Time: {output.planning_time:.3f}s")

    global_path = output.global_path
    local_path = output.local_path

    # ----------------------------------------------------
    # 7️⃣ VISUALIZATION
    # ----------------------------------------------------
    print("📊 Visualizing results...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Terrain + paths
    im = axes[0,0].imshow(dem_crop,
                          cmap='terrain',
                          origin='lower')

    plt.colorbar(im, ax=axes[0,0], shrink=0.6)

    gx, gy = list(zip(*global_path))
    axes[0,0].plot(gx, gy, 'b-o', label="Global A*")

    lx = [s.x for s in local_path]
    ly = [s.y for s in local_path]
    axes[0,0].plot(lx, ly, 'lime', linewidth=3, label="Hybrid A*")

    axes[0,0].scatter(*start_px, c='green', s=150, marker='*')
    axes[0,0].scatter(*goal_px, c='red', s=150, marker='*')

    axes[0,0].set_title("Integrated Planning Stack")
    axes[0,0].legend()
    axes[0,0].axis("equal")

    # Costmap slice
    axes[0,1].imshow(costmap_3d[0], cmap='Reds', origin='lower')
    axes[0,1].plot(gx, gy, 'b-')
    axes[0,1].set_title("Costmap (Yaw 0 Slice)")

    # Cost-to-go from D*
    axes[1,0].imshow(output.cost_to_goal,
                     cmap='viridis',
                     origin='lower')
    axes[1,0].set_title("D* Cost-To-Go Map")

    # Stats
    axes[1,1].axis("off")

    stats = f"""
MISSION SUCCESS

Grid: {H}x{W}
Yaw layers: {Y}
Global waypoints: {len(global_path)}
Local states: {len(local_path)}
Planning time: {output.planning_time:.3f}s
"""

    axes[1,1].text(
        0.05, 0.95, stats,
        transform=axes[1,1].transAxes,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='lightgreen',
                  alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig("planner_test_FULL_STACK.png", dpi=200)
    plt.show()

    print("🎉 FULL STACK TEST COMPLETE")
    print("Saved: planner_test_FULL_STACK.png")

    map_server.close()
    return True


if __name__ == "__main__":
    planner_test()