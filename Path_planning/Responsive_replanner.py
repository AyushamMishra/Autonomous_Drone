import numpy as np
import heapq
from Path_planning.motion_primitives import MotionPrimitives


INF=1e9

class DStarLite:
    
    def __init__(self,costmap,goal,start):
        self.costmap=costmap
        self.goal=goal
        self.H,self.W=costmap.shape
        self.start=start

        self.g={}
        self.rhs={}
        self.U=[]
        self.km=0.0 

        self.moves=[(-1,0),(1,0),(0,-1),(0,1),
                    (-1,-1),(-1,1),(1,-1),(1,1)
                    ]

        # Initializze goal
        self.rhs[goal]=0.0
        self.g[goal] = INF
        heapq.heappush(self.U, (self.key(goal), goal))

    
    def heuristic(self,a,b):
        return np.hypot(a[0]-b[0], a[1]-b[1])
    
    def key(self,s):
        g_rhs = min(self.g.get(s, INF), self.rhs.get(s, INF))
        return (g_rhs + self.heuristic(self.start, s) + self.km,g_rhs)
    
    def in_bounds(self,x,y):
        return 0 <= x < self.W and 0 <= y < self.H
    
    def cost (self,s,sp):
        dx = abs(sp[0] - s[0])
        dy = abs(sp[1] - s[1])

        if dx == 1 and dy == 1:
            base = 1.414
        else:
            base = 1.0

        return base + self.costmap[sp[1], sp[0]]
    
    def neighbours(self, s):
        x, y = s
        for dx, dy in self.moves:
            nx, ny = x+dx, y+dy
            if self.in_bounds(nx, ny):
                yield (nx, ny)

    def update_vertex(self,u):
        if u != self.goal:
            self.rhs[u] = min(
                self.g.get(sp, INF) + self.cost(u, sp)
                for sp in self.neighbours(u)
            )

        self.U = [(k,s) for k,s in self.U if s != u]
        heapq.heapify(self.U)

        if self.g.get(u, INF) != self.rhs.get(u, INF):
            heapq.heappush(self.U, (self.key(u), u))

    def compute_shortest_path(self,start):
        while self.U:
            k_old, u = heapq.heappop(self.U)
            if k_old >= self.key(start) and \
               self.rhs.get(start, INF) == self.g.get(start, INF):
                break

            if self.g.get(u, INF) > self.rhs.get(u, INF):
                self.g[u] = self.rhs[u]
                for s in self.neighbours(u):
                    self.update_vertex(s)
            else:
                self.g[u] = INF
                self.update_vertex(u)
                for s in self.neighbours(u):
                    self.update_vertex(s)

    def replan(self,start):
        old_start = self.start 
        self.km += self.heuristic(old_start, start)
        self.start = start
        self.compute_shortest_path(start)
        return self.get_cost_map()
    
    def update_cell(self,x,y,new_cost):
        self.costmap[y, x] = new_cost
        self.update_vertex((x, y))
        for s in self.neighbours((x, y)):
            self.update_vertex(s)
  
    def get_cost_map(self):
        costmap = np.full((self.H, self.W), INF)
        for (x,y), v in self.g.items():
            costmap[y, x] = v
        return costmap

