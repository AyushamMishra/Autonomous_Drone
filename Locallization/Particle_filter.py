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

        # Particle state: [x, y, yaw, last_elevation]
        self.particles = np.zeros((num_particles, 4))

        # Initialize position + yaw
        self.particles[:, 0] = start_x + np.random.uniform(-spread, spread, num_particles)
        self.particles[:, 1] = start_y + np.random.uniform(-spread, spread, num_particles)
        yaw_spread = np.deg2rad(10)
        self.particles[:, 2] = start_yaw + np.random.uniform(-yaw_spread, yaw_spread, num_particles)


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
        self.particles[:, :3] = self.motion_model.propagate(
            self.particles[:, :3],
            control_inputs,
            dt
        )


    # UPDATE

    def update_weights(self, measured_slope, measured_altitude,measured_yaw, sigma_altitude, sigma_slope,sigma_heading):
        for i in range(self.num_particles):
            x, y = self.particles[i, 0], self.particles[i, 1]
            last_z = self.particles[i, 3]

            current_z = self.map_server.get_elevation(x, y)

        # Invalid DEM lookup
            if current_z is None or np.isnan(current_z) or np.isnan(last_z):
                self.weights[i] = 1e-6
                continue

        # Map-derived slope
            yaw = self.particles[i, 2]

            # expected forward motion
            ds = 1.0  # meters (can be v*dt, but keep small)
            dx = ds * np.cos(yaw)
            dy = ds * np.sin(yaw)

            z_forward = self.map_server.get_elevation(x + dx, y + dy)
            if z_forward is None or np.isnan(z_forward):
                self.weights[i] = 1e-6
                continue

            map_slope = z_forward - current_z

        # Slope error
            slope_error = measured_slope - map_slope

        # Gaussian Slope likelihood
            slope_likelihood = np.exp(
            -0.5 * (slope_error ** 2) / (sigma_slope ** 2)
        )
            
        ## Altitude Error
            altitude_error = measured_altitude - current_z

        # Gaussian Altitude likelihood
            Altitude_likelihood = np.exp(
                -0.5 * (altitude_error ** 2) / (sigma_altitude ** 2)
            )

        ## Heading(Yaw) Error and Likelihood
        
            heading_error = wrap_angle(measured_yaw - self.particles[i, 2])

            heading_likelihood = np.exp(
                -0.5 * (heading_error ** 2) / (sigma_heading ** 2)
            )   


        # Combined Likelihood
            self.weights[i] = (slope_likelihood * Altitude_likelihood * heading_likelihood) + 1e-12

        # Update stored elevation
            self.particles[i, 3] = current_z

    # NORMALIZE ONCe
        weight_sum = np.sum(self.weights)

        if weight_sum == 0 or np.isnan(weight_sum):
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights /= weight_sum

    # Effective Particle count
        N_eff = 1.0 / np.sum(self.weights**2)
        print(f"Effective Particle Count: {N_eff:.2f}")

    # RESAMPLE

    def resample(self):
        if np.any(np.isnan(self.weights)):
            raise RuntimeError("Nan weights detected during resampling.")
        

        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )

        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    # ESTIMATE

    def estimate(self):
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)
        mean_yaw = np.average(self.particles[:, 2], weights=self.weights)

        return mean_x, mean_y, mean_yaw
