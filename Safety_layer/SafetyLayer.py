"""
Safety Layer

Filters control commands to ensure system safety constraints
before commands reach actuators or PX4.

This layer is stateless except for timeout monitoring.
"""

import math
import time
import numpy as np
import logging
logger=logging.getLogger("SafetyLayer")

from Core.Interfaces import State, ControlCommand
from Safety_layer.Safety_configuration import SafetyConfiguration as SafetyConfig


class SafetyLayer:
    
    ##  Enforces safety constraints on control commands.

    def __init__(self, config: SafetyConfig, home_position=(0.0, 0.0)):

        self.config = config
        self.home_x = home_position[0]
        self.home_y = home_position[1]

        self.last_target_seen = time.time()

    
    # Public API
   

    def enforce(self, state: State, command: ControlCommand) -> ControlCommand:
    
       ## Main entrypoint for safety enforcement.
       

        command = self._sanitize_command(command)
        command = self._limit_velocity(command)
        command = self._enforce_altitude(state, command)
        command = self._enforce_geofence(state, command)
        command = self._enforce_target_timeout(command)

        return command

    def update_target_detection(self):
        
        ##Called by perception module when a target is detected.
        
        self.last_target_seen = time.time()

    
    # Safety Checks

    def _sanitize_command(self, command: ControlCommand) -> ControlCommand:
        """
        Prevent invalid control values.
        """

        velocity = command.velocity
        yaw_rate = command.yaw_rate

        if velocity is not None and (math.isnan(velocity) or math.isinf(velocity)):
            velocity = 0.0

        if yaw_rate is not None and (math.isnan(yaw_rate) or math.isinf(yaw_rate)):
            yaw_rate = 0.0

        return ControlCommand(
            velocity=velocity,
            yaw_rate=yaw_rate,
            thrust=command.thrust,
            body_rates=command.body_rates
        )

   

    def _limit_velocity(self, command: ControlCommand) -> ControlCommand:
        """
        Clamp horizontal velocity magnitude.
        """

        if command.velocity is None:
            return command

        v = command.velocity
        max_v = self.config.max_velocity_xy

        if abs(v) > max_v:
            logger.warning("Velocity limit exceeded: %.2f > %.2f. Clamping.",v,max_v)
            v = max_v * np.sign(v)

        return ControlCommand(
            velocity=v,
            yaw_rate=command.yaw_rate,
            thrust=command.thrust,
            body_rates=command.body_rates
        )

    

    def _enforce_altitude(self, state: State, command: ControlCommand) -> ControlCommand:
        """
        Prevent altitude violations using predicted altitude.
        """

        if state.z is None or command.velocity is None:
            return command

        predicted_z = state.z + command.velocity * self.config.command_dt

        if predicted_z > self.config.max_altitude:
            logger.warning("Altitude limit exceeded: predicted_z=%.2f > max_altitude=%.2f",predicted_z,self.config.max_altitude)
            return ControlCommand(
                velocity=-abs(command.velocity),
                yaw_rate=command.yaw_rate
            )

        if predicted_z < self.config.min_altitude:
            logger.warning("Altitude limit exceeded: predicted_z=%.2f < min_altitude=%.2f",predicted_z,self.config.min_altitude)
            return ControlCommand(
                velocity=abs(command.velocity),
                yaw_rate=command.yaw_rate
            )

        return command

    

    def _enforce_geofence(self, state: State, command: ControlCommand) -> ControlCommand:
        """
        Prevent leaving mission boundary.
        """

        dx = state.x - self.home_x
        dy = state.y - self.home_y

        distance = math.sqrt(dx**2 + dy**2)

        if distance > self.config.geofence_radius:

            direction_x = -dx / distance
            direction_y = -dy / distance

            safe_velocity = self.config.max_velocity_xy * 0.5

            logger.error("Geofence violation: distance %.2f > radius %.2f. Returning to home.",distance,self.config.geofence_radius)

            return ControlCommand(
                velocity=safe_velocity,
                yaw_rate=math.atan2(direction_y, direction_x)
            )

        return command

   

    def _enforce_target_timeout(self, command: ControlCommand) -> ControlCommand:
        """
        Hover if target lost for too long.
        """

        if time.time() - self.last_target_seen > self.config.target_timeout:

            logger.warning("Target timeout triggered. Entering hover failsafe.")

            return ControlCommand(
                velocity=0.0,
                yaw_rate=0.0
            )

        return command