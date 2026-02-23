import time
import numpy as np
from Core.Interfaces import SensorData


def fake_sensor() -> SensorData:
    return SensorData(
        timestamp=time.time(),
        altitude=10.0,
        slope=0.0,
        heading=0.0,
        velocity=1.0,
        yaw_rate=0.0,
        accel=np.zeros(3),
        gyro=np.zeros(3),
        image=None
    )