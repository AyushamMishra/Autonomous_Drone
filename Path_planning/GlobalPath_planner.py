import numpy as np
import math 
import heapq 
from Path_planning.state import State 


class AStarGlobalPlanner:
    def __init__(self, costmap, obstacle_threshold):

        """
        costmap : [H, W] 2D cost map (same resolution as Hybrid A*)
        obstacle_threshold : cells >= this are treated as obstacles
        """

        self.costmap = costmap
        self.H, self.W = costmap.shape
        self.obs_th = obstacle_threshold

        self.moves = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
    def heuristics (self,a,b):
            return np.hypot(a[0]-b[0],a[1]-b[1])
        
    def in_bounds(self,x,y):
            return 0<= x < self.W and 0 <= y < self.H
        
    def is_free(self,x,y):
            return np.isfinite(self.costmap[y, x]) and self.costmap[y, x] < self.obs_th
        
    def global_plan(self,start_xy,goal_xy):

            # Returns list of (x,y) tuples og global waypoints or None if no path found

            open_set = []
            heapq.heappush(open_set, (0.0, start_xy))

            came_from = {}
            g_score = {start_xy: 0.0}

            while open_set:
                _, current = heapq.heappop(open_set)

                if current == goal_xy:
                    return self.reconstruct_path(came_from,current)
                
                for dx,dy in self.moves:
                    nx, ny = current[0] + dx, current[1] + dy

                    if not self.in_bounds(nx,ny) or not self.is_free(nx,ny):
                        continue

                    step_cost = np.hypot(dx, dy) + self.costmap[ny, nx]
                    new_g = g_score[current] + step_cost
                    neighbor = (nx, ny)

                    if neighbor not in g_score or new_g<g_score[neighbor]:
                        g_score[neighbor] = new_g
                        f = new_g + self.heuristics(neighbor, goal_xy)
                        heapq.heappush(open_set, (f, neighbor))
                        came_from[neighbor] = current

            return None  # No path found
    @staticmethod
    def compute_cost_to_goal(costmap,goal_xy,obs_th):
        H, W = costmap.shape
        cost_to_goal = np.full((H, W), np.inf)
        pq = []
        gx, gy = goal_xy
        cost_to_goal[gy, gx] = 0.0
        heapq.heappush(pq, (0.0, (gx, gy)))

        moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while pq:
            g, (x, y) = heapq.heappop(pq)
            for dx, dy in moves:
                nx, ny = x + dx, y + dy

                # 1. Bounds check (hard stop)
                if not (0 <= nx < W and 0 <= ny < H):
                    continue

                # 2. Obstacle check
                if costmap[ny, nx] >= obs_th:
                    continue

                new_g = g + np.hypot(dx, dy)

                if new_g < cost_to_goal[ny, nx]:
                    cost_to_goal[ny, nx] = new_g
                    heapq.heappush(pq, (new_g, (nx, ny)))


        return cost_to_goal
    
    @staticmethod  
    def build_corridor(path,H,W,radius=10):
        mask=np.zeros((H,W),dtype=bool)
        for x,y in path:
              for dx in range(-radius,radius+1):
                   for dy in range(-radius,radius+1):
                        nx,ny=x+dx,y+dy
                        if 0 <= nx < W and 0 <= ny < H:
                            mask[ny, nx] = True

        return mask
        
    @staticmethod  
    def reconstruct_path(came_from,current):
        path=[current]
        while current in came_from:
            current = came_from[current]
            path.append(current)

        return path[::-1]


# ==========================================================
# TEMPORARY TEST SCRIPT FOR GLOBAL A*
# ==========================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # -------------------------------
    # 1. Create a toy costmap
    # -------------------------------
    H, W = 120, 160
    costmap = np.zeros((H, W), dtype=float)

    # Add obstacles
    costmap[40:80, 60:65] = 100.0   # vertical wall
    costmap[20:25, 20:120] = 100.0  # horizontal wall
    costmap[90:110, 100:140] = 100.0

    obstacle_threshold = 50.0

    # -------------------------------
    # 2. Start / Goal
    # -------------------------------
    start = (10, 10)
    goal = (140, 100)

    # -------------------------------
    # 3. Run Global A*
    # -------------------------------
    planner = AStarGlobalPlanner(costmap, obstacle_threshold)
    path = planner.global_plan(start, goal)

    if path is None:
        print("[TEST] ❌ No path found")
        exit(0)

    print(f"[TEST] ✅ Path found with {len(path)} waypoints")

    # -------------------------------
    # 4. Compute cost-to-go map
    # -------------------------------
    cost_to_goal = AStarGlobalPlanner.compute_cost_to_goal(
        costmap, goal, obstacle_threshold
    )

    # -------------------------------
    # 5. Build corridor
    # -------------------------------
    corridor = AStarGlobalPlanner.build_corridor(
        path, H, W, radius=6
    )

    # -------------------------------
    # 6. Visualization
    # -------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # --- Costmap + Path
    axs[0].imshow(costmap, cmap="gray_r")
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    axs[0].plot(xs, ys, "r-", linewidth=2, label="Global Path")
    axs[0].scatter(*start, c="green", s=60, label="Start")
    axs[0].scatter(*goal, c="blue", s=60, label="Goal")
    axs[0].set_title("Global A* Path")
    axs[0].legend()

    # --- Corridor mask
    axs[1].imshow(corridor, cmap="Blues")
    axs[1].plot(xs, ys, "r-", linewidth=2)
    axs[1].set_title("Global Corridor Mask")

    # --- Cost-to-go
    im = axs[2].imshow(cost_to_goal, cmap="viridis")
    axs[2].set_title("Cost-to-Goal Map")
    plt.colorbar(im, ax=axs[2], shrink=0.8)

    plt.tight_layout()
    plt.show()
