import numpy as np
import map_server


class ParticleFilter:
    def __init__(self, map_server, num_particles, start_x, start_y, start_yaw, spread):
        self.map_server = map_server
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


    # PREDICTION

    def predict(self, move_x, move_y, move_yaw, sigma_move):
        noise_x = np.random.normal(0, sigma_move, self.num_particles)
        noise_y = np.random.normal(0, sigma_move, self.num_particles)
        noise_yaw = np.random.normal(0, sigma_move / 10, self.num_particles)

        self.particles[:, 0] += move_x + noise_x
        self.particles[:, 1] += move_y + noise_y
        self.particles[:, 2] += move_yaw + noise_yaw
        
    
    # UPDATE

    def update_weights(self, measured_slope, sigma_slope):
        for i in range(self.num_particles):
            x, y = self.particles[i, 0], self.particles[i, 1]
            last_z = self.particles[i, 3]

            current_z = self.map_server.get_elevation(x, y)

        # Invalid DEM lookup
            if current_z is None or np.isnan(current_z) or np.isnan(last_z):
                self.weights[i] = 1e-6
                continue

        # Map-derived slope
            map_slope = current_z - last_z

        # Slope error
            slope_error = measured_slope - map_slope

        # Gaussian Slope likelihood
            self.weights[i] = np.exp(
            -0.5 * (slope_error ** 2) / (sigma_slope ** 2)
        ) + 1e-6

        # Update stored elevation
            self.particles[i, 3] = current_z

    # NORMALIZE ONCe
        weight_sum = np.sum(self.weights)

        if weight_sum == 0 or np.isnan(weight_sum):
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights /= weight_sum

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
