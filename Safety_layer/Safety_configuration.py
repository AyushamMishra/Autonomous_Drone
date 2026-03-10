from dataclasses import dataclass

@dataclass
class SafetyConfiguration:
    # velocity limits
    max_velocity_xy: float = 5.0
    max_velocity_z: float = 2.0

    # altitude limits
    min_altitude: float = 1.0
    max_altitude: float = 130.0

    # geofence
    geofence_radius: float = 120.0

    # failsafe
    target_timeout: float = 3.0

    # control timestep assumption
    command_dt: float = 0.1