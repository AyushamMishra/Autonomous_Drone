
import numpy as np
from Locallization.map_server import Map_Server


class CostmapBuilder:
    def __init__(self, map_server, yaw_bins=16):
        self.map = map_server
        self.height = map_server.height
        self.width = map_server.width
        self.res_x = map_server.res_x
        self.res_y = map_server.res_y

        self.yaw_bins = yaw_bins
        self.yaw_angles = np.linspace(0, 2*np.pi, yaw_bins, endpoint=False)

        # 3D costmap: [yaw, row, col]
        self.costmap = np.full((yaw_bins, self.height, self.width), np.inf)

    
    # LAYER 1: Base traversability
    
    def compute_base_layer(self):
        base = np.zeros_like(self.map.dem, dtype=float)
        base[np.isnan(self.map.dem)] = np.inf
        return base

    
    # Precompute terrain gradients
    
    def compute_gradients(self):
        dzdx = np.full((self.height, self.width), np.nan)
        dzdy = np.full((self.height, self.width), np.nan)

        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                if np.isnan(self.map.dem[r, c]):
                    continue

                dzdx[r, c] = (
                    self.map.dem[r, c+1] - self.map.dem[r, c-1]
                ) / (2 * self.res_x)

                dzdy[r, c] = (
                    self.map.dem[r+1, c] - self.map.dem[r-1, c]
                ) / (2 * self.res_y)

        return dzdx, dzdy

    
    # LAYER 2: Anisotropic slope cost
    
    def compute_anisotropic_slope_layer(self, dzdx, dzdy):
        slope_cost = np.zeros(
            (self.yaw_bins, self.height, self.width)
        )

        for k, theta in enumerate(self.yaw_angles):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Directional slope
            s_theta = dzdx * cos_t + dzdy * sin_t

            # Penalize uphill more than downhill
            uphill = np.maximum(s_theta, 0.0)

            slope_cost[k] = uphill ** 2

            slope_cost[k][np.isnan(s_theta)] = np.inf

        return slope_cost

    
    # LAYER 3: Terrain roughness
    
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
                else:
                    rough[r, c] = np.var(patch)

        return rough

    
    # LAYER 4: Feature richness
    
    def compute_feature_layer(self, dzdx, dzdy):
        feature = np.hypot(dzdx, dzdy)
        feature[np.isnan(feature)] = 0.0
        return 1.0 / (feature + 1e-3)
    

    # Layer 5: Uncertainity Inflation Layer

    def inflate_with_uncertainty(self,
                             sigma_x,
                             sigma_y,
                             inflation_scale=2.5):
        """
            sigma_x, sigma_y : localization std dev in meters
            inflation_scale  : number of sigmas to cover (2–3 typical)
        """

        # Convert meters → pixels
        sx = sigma_x / self.res_x
        sy = sigma_y / self.res_y

        rx = int(np.ceil(inflation_scale * sx))
        ry = int(np.ceil(inflation_scale * sy))
        if rx == 0 and ry == 0:
            return self.costmap

        # Gaussian kernel
        x = np.arange(-rx, rx + 1)
        y = np.arange(-ry, ry + 1)
        X, Y = np.meshgrid(x, y)

        G = np.exp(-0.5 * ((X / sx) ** 2 + (Y / sy) ** 2))
        G /= np.max(G)

        inflated = np.copy(self.costmap)

        for k in range(self.yaw_bins):
            slice_k = self.costmap[k]

            for r in range(self.height):
                for c in range(self.width):
                    if not np.isfinite(slice_k[r, c]):
                        continue

                    r0 = max(0, r - ry)
                    r1 = min(self.height, r + ry + 1)
                    c0 = max(0, c - rx)
                    c1 = min(self.width, c + rx + 1)

                    kr0 = ry - (r - r0)
                    kr1 = kr0 + (r1 - r0)
                    kc0 = rx - (c - c0)
                    kc1 = kc0 + (c1 - c0)

                    local = slice_k[r0:r1, c0:c1]
                    kernel = G[kr0:kr1, kc0:kc1]

                    inflated[k, r, c] = np.max(local * kernel)

        self.costmap = inflated
        return self.costmap
    
    # Layer 6:  Computing vehicle footprint cost 
    def compute_vehicle_footprint_cost(self,vehicle_length,vehicle_width):
        # Penalizes terrain features that could interfere with the vehicle's footprint.
        L = int(np.ceil(vehicle_length / self.res_x))
        W = int(np.ceil(vehicle_width  / self.res_y))

        footprint_cost = np.zeros_like(self.costmap)

        for k, theta in enumerate(self.yaw_angles):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            for r in range(self.height):
                for c in range(self.width):
                    if not np.isfinite(self.costmap[k, r, c]):
                        footprint_cost[k, r, c] = np.inf
                        continue

                    max_cost = 0.0
                    for dx in range(-L//2, L//2 + 1):
                        for dy in range(-W//2, W//2 + 1):

                            rr = int(r + dx * sin_t + dy * cos_t)
                            cc = int(c + dx * cos_t - dy * sin_t)
                            if rr < 0 or rr >= self.height or cc < 0 or cc >= self.width:
                                max_cost = np.inf
                                break

                            val = self.costmap[k, rr, cc]
                            if not np.isfinite(val):
                                max_cost = np.inf
                                break

                            max_cost = max(max_cost, val)

                    footprint_cost[k, r, c] = max_cost

        return footprint_cost
    
    # Layer 7: Heading alignment cost

    def compute_heading_alignment_cost(self,dzdx,dzdy):
        # Penalizes vehicle heading that misaligns with terrain gradients.
        align_cost = np.zeros(
        (self.yaw_bins, self.height, self.width)
                             )

        grad_mag = np.hypot(dzdx, dzdy)
        grad_dir = np.arctan2(dzdy, dzdx)

        for k, yaw in enumerate(self.yaw_angles):
            dtheta = np.abs(
                np.arctan2(np.sin(grad_dir - yaw),np.cos(grad_dir - yaw))
            )

            align_cost[k] = grad_mag * np.sin(dtheta)
            align_cost[k][np.isnan(align_cost[k])] = 0.0

        return align_cost
    
    # Layer 8: Turn rate cost

    def compute_turn_rate_cost(self,w_turn=1.0):
        # Penalizes sharp turns based on yaw bin differences.
        turn_cost = np.zeros_like(self.costmap)

        for k in range(self.yaw_bins):
            prev_k = (k - 1) % self.yaw_bins
            next_k = (k + 1) % self.yaw_bins

            dtheta = min(abs(self.yaw_angles[k] - self.yaw_angles[prev_k]),
                         abs(self.yaw_angles[next_k] - self.yaw_angles[k])
                 )

            turn_cost[k] = w_turn * (dtheta ** 2)

        return turn_cost




    
    # BUILD FINAL COSTMAP
    
    def build(self,w_slope=2.0,w_rough=1.5,
              w_feature=1.0,sigma_x=None, sigma_y=None,
              w_align=1.2,w_turn=0.8,
              vehicle_length=2,vehicle_width=0.5
              ):

        base = self.compute_base_layer()
        dzdx, dzdy = self.compute_gradients()
        slope = self.compute_anisotropic_slope_layer(dzdx, dzdy)
        rough = self.compute_roughness_layer()
        feature = self.compute_feature_layer(dzdx, dzdy)
        align = self.compute_heading_alignment_cost(dzdx, dzdy)
        turn  = self.compute_turn_rate_cost(w_turn)


        for k in range(self.yaw_bins):
            self.costmap[k] = (
                base +
                w_slope * slope[k] +
                w_rough * rough +
                w_feature * feature +
                w_align * align[k]+
                turn[k]
            )

        # Uncertainity inflation

        if sigma_x is not None and sigma_y is not None:
            self.inflate_with_uncertainty(sigma_x, sigma_y)
            
        # Vehicle footprint cost
        footprint = self.compute_vehicle_footprint_cost(vehicle_length,vehicle_width)

        self.costmap = np.maximum(self.costmap, footprint)  

            # Normalize per yaw slice
        for k in range(self.yaw_bins):
            finite = np.isfinite(self.costmap[k])
            if np.any(finite):
                self.costmap[k][finite] /= np.max(self.costmap[k][finite])

    

        return self.costmap
