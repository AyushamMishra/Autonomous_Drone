import numpy as np
import map_server
from Motion_model import MotionModel

# Angle wrapping function:
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ParticleFilter:
    def __init__(self, map_server, num_particles,sigma_altitude, start_x, start_y, start_yaw, spread):
        self.map_server = map_server
        self.motion_model = MotionModel(sigma_v=0.5, sigma_yaw=0.01)
        self.num_particles = num_particles

        # Particle state: [x, y, yaw, last_elevation,last_dzdx,last_dzdy,last_x,last_y,prev_z]
        self.particles = np.zeros((num_particles, 9))

        # Initialize position + yaw
        self.particles[:, 0] = start_x + np.random.uniform(-spread, spread, num_particles)
        self.particles[:, 1] = start_y + np.random.uniform(-spread, spread, num_particles)

        yaw_spread = np.deg2rad(10)

        self.particles[:, 2] = start_yaw + np.random.uniform(-yaw_spread, yaw_spread, num_particles)
        # Initialize gradient and History
        dz_dx, dz_dy = self.map_server.get_gradient(start_x, start_y)
        self.particles[:, 4] = dz_dx
        self.particles[:, 5] = dz_dy
        self.particles[:, 6] = self.particles[:, 0]
        self.particles[:, 7] = self.particles[:, 1]

        # Initialize last_elevation from DEM
        for i in range(num_particles):
            z = self.map_server.get_elevation(
                self.particles[i, 0],
                self.particles[i, 1]
            )

            if z is None or np.isnan(z):
                self.particles[i, 3] = np.nan
            else:
                self.particles[i, 3] = z

        # Uniform weights
        self.weights = np.ones(num_particles) / num_particles

    # Predict
    def predict(self,control_inputs,dt):

        # Store previous postion BEFORE motion
        self.particles[:, 6] = self.particles[:, 0]   # x_last
        self.particles[:, 7] = self.particles[:, 1]   # y_last
        # Store previous elevation before motion
        self.particles[:, 8] = self.particles[:, 3]   # prev_z

        
        
        self.particles[:, :3] = self.motion_model.propagate(
            self.particles[:, :3],
            control_inputs,
            dt
        )


    # UPDATE

    def update_weights(self, measured_slope, measured_altitude, measured_yaw, sigma_altitude, sigma_slope, sigma_heading):
        terrain_flatness_threshold = 0.02  # Flat if grad_mag < 2%
        eps=1e-12
    
        for i in range(self.num_particles):
            x, y = self.particles[i, 0], self.particles[i, 1]
            last_z = self.particles[i, 3]
            current_z = self.map_server.get_elevation(x, y)
             
            # Invalid DEM Lookup 
            if np.isnan(current_z) or np.isnan(last_z):
                self.weights[i] = eps
                continue
            
            # Map derived Slope
            yaw = self.particles[i, 2]
        
        #             CORE LIKELIHOODS (Always active) 
        #    Altitude (primary)
            altitude_error = measured_altitude - current_z    
            altitude_likelihood = np.exp(-0.5 * (altitude_error)**2 / sigma_altitude**2)
        
        #    Heading  
            heading_error = wrap_angle(measured_yaw - yaw)
            heading_likelihood = np.exp(-0.5 * (heading_error)**2 / sigma_heading**2)
        
        #              CONDITIONAL LIKELIHOODS (Terrain-dependent)(only works when terrain is not flat and has features )
            base_likelihood = altitude_likelihood * heading_likelihood
        
        # Check terrain texture
            dz_dx, dz_dy = self.map_server.get_gradient(x, y)
            if np.isnan(dz_dx) or np.isnan(dz_dy):
                dz_dx, dz_dy = 0.0, 0.0
            
            max_grad=0.15          # Cap max gradient to avoid extreme cases (Change as needed)
            dz_dx = np.clip(dz_dx, -max_grad, max_grad)
            dz_dy = np.clip(dz_dy, -max_grad, max_grad)
            grad_mag = np.hypot(dz_dx, dz_dy)

            #   Along-track elevation consistency
            prev_z = self.particles[i, 8]
            dxm = x - self.particles[i, 6]
            dym = y - self.particles[i, 7]

            if not np.isnan(prev_z):
                dz_obs = current_z - prev_z
                dz_pred = dz_dx * dxm + dz_dy * dym

                sigma_dz = 2.0                  # meters (DEM-limited)
                L_dz = np.exp(-0.5 * (dz_obs - dz_pred)**2 / sigma_dz**2)
                base_likelihood *= L_dz ** 0.6

            # 3. Slope (only if terrain has slope features)

            if grad_mag > terrain_flatness_threshold:
                ds = 0.5 * min(self.map_server.res_x, self.map_server.res_y)
                dx = ds * np.cos(yaw)
                dy = ds * np.sin(yaw)
                z_forward = self.map_server.get_elevation(x + dx, y + dy)

                if z_forward is not None and not np.isnan(z_forward):
                    map_slope = z_forward - current_z
                    # Slope Error
                    slope_error = measured_slope - map_slope    
                    slope_likelihood = np.exp(-0.5 * (slope_error)**2 / sigma_slope**2)
                    base_likelihood *= slope_likelihood ** 0.7          # Reduced weight
        
        # 4. Cross-track (always gentle)
            motion_norm = np.hypot(dxm, dym)
            
            nx, ny = -np.sin(yaw), np.cos(yaw)
            cross_disp = abs(dxm * nx + dym * ny)
            
            # Tie sigma to motion scale 
            sigma_cross=max(1.0,0.2*np.hypot(dxm,dym))
            L_cross = np.exp(-0.5 * (cross_disp)**2 / sigma_cross**2)
            base_likelihood *= L_cross ** 0.8  # Very gentle weight

            self.weights[i] = min(base_likelihood, 1.0) + eps

        # Update aux
            self.particles[i, 3] = current_z
            self.particles[i, 4] = dz_dx
            self.particles[i, 5] = dz_dy
    
    # Normalize
        weight_sum = np.sum(self.weights)
        self.weights /= weight_sum if weight_sum > 0 else self.num_particles
        N_eff = 1.0 / np.sum(self.weights**2)
        print(f"N_eff: {N_eff:.0f}/{self.num_particles}")



    # RESAMPLE

    def resample(self):
        if np.any(np.isnan(self.weights)):
            raise RuntimeError("Nan weights detected during resampling.")
        
        N_eff=1.0/np.sum(self.weights**2)
        if N_eff>0.3*self.num_particles:
            return   # No resampling needed
        

        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )

        self.particles = self.particles[indices]
        # JITTTER: Prevents collapse
        for i in range(self.num_particles):
            yaw = self.particles[i, 2]
            # Along-track jitter (keep observability)
            self.particles[i, 0] += np.random.normal(0, 1.5) * np.cos(yaw)
            self.particles[i, 1] += np.random.normal(0, 1.5) * np.sin(yaw)
            # Very small cross-track
            self.particles[i, 0] += np.random.normal(0, 0.3) * -np.sin(yaw)
            self.particles[i, 1] += np.random.normal(0, 0.3) *  np.cos(yaw)

        self.particles[:, 2] += np.random.normal(0, np.deg2rad(1), self.num_particles)  # 1° yaw
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])

        self.weights.fill(1.0 / self.num_particles)
        print("RESAMPLE + JITTER")

        

    # ESTIMATE

    def estimate(self):
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)
        mean_yaw = np.average(self.particles[:, 2], weights=self.weights)

        return mean_x, mean_y, mean_yaw
