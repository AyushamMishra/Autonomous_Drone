from py_compile import main
import rasterio
import numpy as np
#np.random.seed(42)  # (only for debugging use cases)



## Sanity test for DEM Projection
""" 
with rasterio.open("D:\Autonomous_Drone\Search_map\cdnh43w\cdnh43w_utm.tif") as ds:
    print(ds.crs)
    print(ds.transform)
    print(ds.bounds)
    x_test = (ds.bounds.left + ds.bounds.right) / 2
    y_test = (ds.bounds.bottom + ds.bounds.top) / 2
"""

import numpy as np
from map_server import Map_Server
from Particle_filter import ParticleFilter
from Projection import projection
import DEM_reprojection

# ---------------- CONFIG ----------------
START_LAT = 28.0005
START_LON = 76.0005

NUM_PARTICLES = 500
SPREAD = 30.0              # meters
SIGMA_MOVE = 0.5           # meters
SIGMA_SLOPE = 0.3          # elevation-delta noise (meters)
SIGMA_ALT = 3.0            # altitude noise (meters), barometric altitude
SIGMA_HEADING=np.deg2rad(1.0)  # heading noise (radians)


MOTION_X = 1.0             # meters/step
MOTION_Y = 0.2
MOTION_YAW = 0.01

NUM_STEPS = 10

velocity=1.0
yaw_rate=0.01
dt=1.0
# ---------------------------------------


def main():

    print("\n--- PF SANITY CHECK (SLOPE + ALT + Heading + Gradient LIKELIHOOD) START ---\n")

    # Load DEM
    map_server = Map_Server(DEM_reprojection.output_dem)
    proj = projection(map_server.crs)

    # Ground-truth initial state
    gt_x, gt_y = proj.lat_lon_to_xy(START_LAT, START_LON)
    gt_yaw = 0.0

    gt_z = map_server.get_elevation(gt_x, gt_y)
    if np.isnan(gt_z):
        raise RuntimeError("Invalid start elevation")

    print(f"Ground truth start elevation: {gt_z:.2f} m\n")

    # Initialize Particle Filter
    pf = ParticleFilter(
        map_server=map_server,
        num_particles=NUM_PARTICLES,
        start_x=gt_x,
        start_y=gt_y,
        start_yaw=gt_yaw,
        spread=SPREAD,
        sigma_altitude=SIGMA_ALT
    )

    # ---------------- SIM LOOP ----------------
    for step in range(NUM_STEPS):

        # --- Ground truth motion ---
        gt_x += velocity * np.cos(gt_yaw) * dt
        gt_y += velocity * np.sin(gt_yaw) * dt
        gt_yaw += yaw_rate * dt

        prev_z = gt_z
        true_z = map_server.get_elevation(gt_x, gt_y)

        if np.isnan(true_z):
            print(f"Step {step:02d} | Invalid GT elevation — skipping")
            continue

        # Altimeter slope measurement (Δz)
        measured_slope = (true_z - prev_z) + np.random.normal(0, SIGMA_SLOPE)
        measured_altitude = true_z+np.random.normal(0,SIGMA_ALT)
        measured_yaw = gt_yaw + np.random.normal(0, SIGMA_HEADING)
        gt_z = true_z

        # --- PF ---
        pf.predict(
            control_inputs=(velocity, yaw_rate),
            dt=dt
        )

        pf.update_weights(
            measured_slope=measured_slope,
            sigma_slope=SIGMA_SLOPE,
            measured_altitude=measured_altitude,
            sigma_altitude=SIGMA_ALT,
            measured_yaw=measured_yaw,
            sigma_heading=SIGMA_HEADING
        )

        pf.resample()

        # --- Estimate ---
        est_x, est_y, est_yaw = pf.estimate()

        pos_error = np.sqrt(
            (est_x - gt_x) ** 2 +
            (est_y - gt_y) ** 2
        )

        print(
            f"Step {step:02d} | "
            f"GT: ({gt_x:.2f}, {gt_y:.2f}) | "
            f"EST: ({est_x:.2f}, {est_y:.2f}) | "
            f"Err: {pos_error:.2f} m | "
            f"dZ: {measured_slope:.3f}"
        )

    print("\n--- PF SANITY CHECK END ---\n")
    map_server.close()


if __name__ == "__main__":
    main()
