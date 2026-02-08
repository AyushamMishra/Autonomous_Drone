import heapq
from tracemalloc import start
import numpy as np
from Path_planning.state import State
import time
from Path_planning.GlobalPath_planner import AStarGlobalPlanner


class HybridAStarPlanner:
    def __init__(self,state_expander,costmap,yaw_angles,goal_tolerance=8.0,corridor_mask=None,global_cost_to_goal=None):


        """
        state_expander : StateExpander
        costmap        : [yaw,H,W]    numpy array of cost values for each yaw angle and grid cell
        yaw_angles     : [yaw]        list of yaw angles in radians
        goal_tolerence : float         distance in meters to consider goal reached
        """

        self.expander = state_expander
        self.costmap = costmap
        self.yaw_angles = yaw_angles
        self.goal_tol = goal_tolerance

        self.H = costmap.shape[1]
        self.W = costmap.shape[2]
        self.Y = len(yaw_angles)
        self.corridor_mask = corridor_mask
        self.global_cost_to_goal = global_cost_to_goal
        self.yaw_res = abs(self.yaw_angles[1] - self.yaw_angles[0])    # Yaw resolution for discretization


    # Heuristics  ( )

    def heuristic(self, s: State, goal: State):
        if self.global_cost_to_goal is None:
            return np.hypot(s.x - goal.x, s.y - goal.y)
        x, y = int(s.x), int(s.y)
        
        # Bounding x,y to be within costmap bounds
        x = np.clip(x, 0, self.W - 1)
        y = np.clip(y, 0, self.H - 1)
        
        h = self.global_cost_to_goal[y, x]
        if goal.yaw is not None:
            yaw_cost = 0.1 * abs((s.yaw - goal.yaw + np.pi) % (2*np.pi) - np.pi)
            h+=0.1*yaw_cost
        return h
    
    # Goal Condition 

    def is_goal(self,s:State,waypoint):
        return np.hypot(s.x - waypoint[0], s.y - waypoint[1]) < self.goal_tol
    
    # Path planning function 

    def construct_path(self, state):
        path = []
        while state is not None:
            path.append(state)
            state = state.parent
        return path[::-1]
    
    # Discetrized state (ensures unique states in open/closed sets)
    def discretize (self,s:State):
        yaw_idx = int(round((s.yaw + np.pi) / self.yaw_res)) % self.Y
        return (
        int(round(s.x)),
        int(round(s.y)),
        yaw_idx
    )

    # Local waypoint goal 
    def select_local_goal(self, current, global_path, lookahead):
        lookahead=int(0.5 * len(global_path))
        dists = [np.hypot(current.x - x, current.y - y) for x, y in global_path]
        idx = np.argmin(dists)

        for k in range(idx + lookahead, idx, -1):
            x, y = global_path[min(k, len(global_path)-1)]
            if self.corridor_mask is None or self.corridor_mask[y, x]:
                return (x, y)

        return global_path[min(idx + lookahead, len(global_path)-1)]


    
    # Hybrid A* Search 
    def plan(self,start:State,goal:State,global_path):
        # Returns list of State forming path from start to goal, or None if no path found
        open_set =[]
        closed =set()
        max_iter=50000
        iter=0
        self.counter=0

        
        t0 = time.time()
        MAX_TIME = 10.0  # seconds

        # Setting local goal 
        lookahead = int(0.5 * len(global_path)) 
        local_goal_xy = self.select_local_goal(start, global_path, lookahead)

        dx = local_goal_xy[0] - start.x
        dy = local_goal_xy[1] - start.y
        goal_yaw = np.arctan2(dy, dx)
        goal = State(local_goal_xy[0], local_goal_xy[1], None)

        start.g=0.0
        start.h=self.heuristic(start,goal)
        start.f=start.g+start.h
        skey = self.discretize(start)
        g_score = {}
        g_score[skey] = 0.0
        heapq.heappush(open_set, (start.f,self.counter, start))

        



        

        while open_set:
            iter+=1
            if iter>max_iter:
                print("[Planner] ❌ Max iterations reached")
                return None
 
            if iter % 500 == 0:
                print(f"[Planner] iter={iter}, open={len(open_set)}, closed={len(closed)}")

            _,_, current = heapq.heappop(open_set)
            ckey = self.discretize(current)

            if ckey in closed:
                continue
            closed.add(ckey)

            # Heading alignment near goal
            pos_ok = np.hypot(current.x - goal.x, current.y - goal.y) < self.goal_tol
            yaw_ok = True if goal.yaw is None else abs((current.yaw - goal.yaw + np.pi) % (2*np.pi) - np.pi) < np.deg2rad(10)

            if pos_ok and yaw_ok:
                return self.construct_path(current)

            
            # Time limit check
            if time.time() - t0 > MAX_TIME:
                print("[Planner] ⏱ Time limit reached")
                return None

            
            # Restricting expansion to global corridor
            CORRIDOR_PENALTY =  3.0  # tune

            inside_corridor = True
            if self.corridor_mask is not None:
                inside_corridor = self.corridor_mask[int(current.y), int(current.x)]


            # Expand using motion primitives
            successors = self.expander.expand(current)

            # Guard against too many successors (e.g. from bad primitives)
            successors.sort(key=lambda s: s.g)
            # successors = successors[:50]

            for nxt in successors:
                nyaw = int(round((nxt.yaw + np.pi) / self.yaw_res)) % self.Y
                nxt.yaw = self.yaw_angles[nyaw]
                nx, ny = int(round(nxt.x)), int(round(nxt.y))


                if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
                    continue


                # Adding penalty for leaving corridor
                inside_corridor = True
                if self.corridor_mask is not None:
                    inside_corridor = self.corridor_mask[ny, nx]

                if self.corridor_mask is not None and not self.corridor_mask[ny, nx]:
                    nxt.g += 0.05 * self.global_cost_to_goal[ny, nx]

                # Yaw index bounds check
                if nyaw < 0 or nyaw >= self.Y:
                    continue

                if not np.isfinite(self.costmap[nyaw, ny, nx]):
                    continue

                skey = self.discretize(nxt)
                
                # Penalizing yaw changes or steering 
                STEER_PENALTY = 0.5         # Tune 
                yaw_diff = abs((nxt.yaw - current.yaw + np.pi) % (2*np.pi) - np.pi)
                nxt.g += STEER_PENALTY * yaw_diff
                
                g_new=nxt.g

                if skey in g_score and g_new >= g_score[skey]:
                    continue

                # Cost from costmap already included in transition
                g_new = nxt.g
                g_score[skey] = g_new
                nxt.h = self.heuristic(nxt, goal)
                nxt.f = g_new + nxt.h
                nxt.parent = current

                self.counter += 1
                heapq.heappush(open_set, (nxt.f, self.counter, nxt))
        print("[planner] No path found")
        return None
    @staticmethod
    def receding_horizon_execute( start_state, goal_xy, global_path,
                                    planner, exec_steps=3, goal_tol=8.0,
                                    max_cycles=200
                                ):
        current = start_state
        executed_path = [current]

        for cycle in range(max_cycles):

            # 1. Check global termination
            if np.hypot(current.x - goal_xy[0], current.y - goal_xy[1]) < goal_tol:
                print(f"[EXEC] ✅ Global goal reached in {cycle} cycles")
                break

            # 2. Plan local path
            local_path = planner.plan(
                current,
                State(goal_xy[0], goal_xy[1], None),
                global_path
                                    )

            if local_path is None or len(local_path) < 2:
                print("[EXEC] ❌ Local planner failed")
                break

            #    3. Execute first N states
            for s in local_path[1:exec_steps+1]:
                current = State(s.x, s.y, s.yaw)
                executed_path.append(current)

            print(
                f"[EXEC] cycle={cycle}, "
                f"pos=({current.x:.1f},{current.y:.1f}), "
                f"yaw={np.rad2deg(current.yaw):.1f}°"
              )

        return executed_path

    

# -------------------------------------------
# Temporary Test script for locak path planner 
# -------------------------------------------


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # -------------------------
    # Dummy map + parameters
    # -------------------------
    H, W = 100, 100
    Y = 16

    costmap = np.zeros((Y, H, W))
    costmap[:] = 1.0

    # Obstacles
    costmap[:, 40:60, 45:55] = np.inf

    # Yaw angles
    yaw_angles = np.linspace(-np.pi, np.pi, Y, endpoint=False)

    # -------------------------
    # Global A* (2D)
    # -------------------------
    global_planner = AStarGlobalPlanner(costmap[0], obstacle_threshold=20.0)
    start_xy = (10, 10)
    goal_xy = (90, 90)

    global_path = global_planner.global_plan(start_xy, goal_xy)

    # Corridor mask
    corridor_mask = np.zeros((H, W), dtype=bool)
    for x, y in global_path:
        corridor_mask[max(0,y-3):min(H,y+3), max(0,x-3):min(W,x+3)] = True

    # Cost-to-go heuristic
    global_cost_to_goal = global_planner.compute_cost_to_goal(costmap[0], goal_xy, obs_th=20.0)

    # -------------------------
    # Dummy motion primitive expander
    # -------------------------
    class SimpleExpander:
        def expand(self, s):
            steps = []
            for d in [-0.3, 0.0, 0.3]:
                yaw = s.yaw + d
                x = s.x + np.cos(yaw)
                y = s.y + np.sin(yaw)
                nxt = State(x, y, yaw)
                nxt.g = s.g + 1.0
                steps.append(nxt)
            return steps

    expander = SimpleExpander()

    # -------------------------
    # Hybrid A*
    # -------------------------
    planner = HybridAStarPlanner(
        expander,
        costmap,
        yaw_angles,
        goal_tolerance=2.0,
        corridor_mask=corridor_mask,
        global_cost_to_goal=global_cost_to_goal
    )

    start = State(start_xy[0], start_xy[1], 0.0)
    goal = State(goal_xy[0], goal_xy[1], 0.0)

    local_path = planner.plan(start, goal, global_path)

    # -------------------------
    # Receding horizon execution loop
    # -------------------------

    executed_path = HybridAStarPlanner.receding_horizon_execute(
                    start_state=start,
                    goal_xy=goal_xy,
                    global_path=global_path,
                    planner=planner,
                    exec_steps=5,
                    goal_tol=8.0
                             )


    # -------------------------
    # Visualization
    # -------------------------
    plt.figure(figsize=(8, 8))
    plt.imshow(costmap[0] == np.inf, cmap="gray", origin="lower")

    # Global path
    gx, gy = zip(*global_path)
    plt.plot(gx, gy, "b--", label="Global A* path")

    # Corridor
    cy, cx = np.where(corridor_mask)
    plt.scatter(cx, cy, s=1, c="cyan", alpha=0.3, label="Corridor")

    # Local path
    if local_path:
        lx = [s.x for s in local_path]
        ly = [s.y for s in local_path]
        plt.plot(lx, ly, "r", linewidth=2, label="Hybrid A* path")

        for s in local_path[::5]:
            plt.arrow(
                s.x, s.y,
                1.5*np.cos(s.yaw), 1.5*np.sin(s.yaw),
                head_width=0.8, color="red"
            )

    # Executed Trajectory 
    if executed_path:
        xs = [s.x for s in executed_path]
        ys = [s.y for s in executed_path]
        plt.plot(xs, ys, "r-", linewidth=2, label="Executed trajectory")

        for s in executed_path[::8]:
            plt.arrow(
                s.x, s.y,
                1.5*np.cos(s.yaw),
                1.5*np.sin(s.yaw),
                head_width=0.6,
                color="red"
                      )


    plt.scatter(*start_xy, c="green", s=100, label="Start")
    plt.scatter(*goal_xy, c="magenta", s=100, label="Goal")

    plt.legend()
    plt.title("Global + Hybrid A* Planner Test")
    plt.show()
