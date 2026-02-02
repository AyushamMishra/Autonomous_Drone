
import numpy as np
from Locallization.map_server import Map_Server


class CostmapBuilder:
    def __init__(self, Map_Server):
        self.map = Map_Server
        self.height = Map_Server.height
        self.width = Map_Server.width
        self.res = max(Map_Server.res_x, Map_Server.res_y)

        # Final costmap
        self.costmap = np.full((self.height, self.width), np.inf)

    # LAYER 1: Base traversability (NaN / no-data)
    
    def compute_base_layer(self):
        base = np.zeros_like(self.map.dem, dtype=float)

        # Mask invalid DEM
        invalid = np.isnan(self.map.dem)
        base[invalid] = np.inf

        return base

    
    # LAYER 2: Slope cost

    def compute_slope_layer(self):
        slope_cost = np.zeros((self.height, self.width))

        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                zc = self.map.dem[r, c]
                if np.isnan(zc):
                    slope_cost[r, c] = np.inf
                    continue

                dzdx = (self.map.dem[r, c+1] - self.map.dem[r, c-1]) / (2 * self.map.res_x)
                dzdy = (self.map.dem[r+1, c] - self.map.dem[r-1, c]) / (2 * self.map.res_y)

                if np.isnan(dzdx) or np.isnan(dzdy):
                    slope_cost[r, c] = np.inf
                    continue

                grad_mag = np.hypot(dzdx, dzdy)

                # Nonlinear penalty
                slope_cost[r, c] = grad_mag ** 2

        return slope_cost

    
    # LAYER 3: Terrain roughness (local variance)
   
    def compute_roughness_layer(self, window=3):
        rough = np.zeros((self.height, self.width))

        half = window // 2
        for r in range(half, self.height - half):
            for c in range(half, self.width - half):
                patch = self.map.dem[
                    r-half:r+half+1,
                    c-half:c+half+1
                ]

                if np.any(np.isnan(patch)):
                    rough[r, c] = np.inf
                    continue

                rough[r, c] = np.var(patch)

        return rough

    
    # LAYER 4: Feature richness (PF observability)
    
    def compute_feature_layer(self):
        feature = np.zeros((self.height, self.width))

        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                dzdx = (self.map.dem[r, c+1] - self.map.dem[r, c-1])
                dzdy = (self.map.dem[r+1, c] - self.map.dem[r-1, c])

                if np.isnan(dzdx) or np.isnan(dzdy):
                    feature[r, c] = 0.0
                else:
                    feature[r, c] = np.hypot(dzdx, dzdy)

        # Inversion: cinverting high features to low cost
        return 1.0 / (feature + 1e-3)

    
    # BUILD FINAL COSTMAP
    
    def build(self,
              w_slope=2.0,
              w_rough=1.5,
              w_feature=1.0):

        base = self.compute_base_layer()
        slope = self.compute_slope_layer()
        rough = self.compute_roughness_layer()
        feature = self.compute_feature_layer()

        self.costmap = (
            base +
            w_slope * slope +
            w_rough * rough +
            w_feature * feature
        )

        # Normalize finite costs
        finite = np.isfinite(self.costmap)
        self.costmap[finite] /= np.max(self.costmap[finite])

        return self.costmap
    
