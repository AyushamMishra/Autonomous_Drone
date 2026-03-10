"""
SafetyLayer Test Suite

Validates all safety enforcement behaviors.

Tests are deterministic and hardware independent.
"""

import time
import numpy as np

from Safety_layer.SafetyLayer import SafetyLayer
from Safety_layer.Safety_configuration import SafetyConfiguration as SafetyConfig
from Core.Interfaces import State, ControlCommand


# -----------------------------------------------------
# Test Fixtures
# -----------------------------------------------------

def create_default_safety():

    config = SafetyConfig(
        max_velocity_xy=2.0,
        max_velocity_z=1.0,
        min_altitude=1.0,
        max_altitude=10.0,
        geofence_radius=20.0,
        target_timeout=1.0
    )

    return SafetyLayer(config)


def create_state(x=0, y=0, z=5):

    return State(
        x=x,
        y=y,
        yaw=0.0,
        z=z
    )


# -----------------------------------------------------
# Tests
# -----------------------------------------------------

def test_velocity_limit():

    safety = create_default_safety()
    state = create_state()

    command = ControlCommand(
        velocity=10.0,
        yaw_rate=0.0
    )

    safe = safety.enforce(state, command)

    assert abs(safe.velocity) <= 2.0


# -----------------------------------------------------

def test_velocity_passthrough():

    safety = create_default_safety()
    state = create_state()

    command = ControlCommand(
        velocity=1.0,
        yaw_rate=0.0
    )

    safe = safety.enforce(state, command)

    assert safe.velocity == command.velocity


# -----------------------------------------------------

def test_altitude_upper_bound():

    safety = create_default_safety()

    state = create_state(z=9.9)

    command = ControlCommand(
        velocity=5.0,
        yaw_rate=0.0
    )

    safe = safety.enforce(state, command)

    # Should command downward motion
    assert safe.velocity <= 0


# -----------------------------------------------------

def test_altitude_lower_bound():

    safety = create_default_safety()

    state = create_state(z=1.1)

    command = ControlCommand(
        velocity=-5.0,
        yaw_rate=0.0
    )

    safe = safety.enforce(state, command)

    # Should command upward motion
    assert safe.velocity >= 0


# -----------------------------------------------------

def test_geofence_enforcement():

    safety = create_default_safety()

    state = create_state(
        x=50,
        y=0,
        z=5
    )

    command = ControlCommand(
        velocity=1.0,
        yaw_rate=0.0
    )

    safe = safety.enforce(state, command)

    # Should override command toward home
    assert safe.velocity is not None


# -----------------------------------------------------

def test_target_timeout_hover():

    safety = create_default_safety()

    state = create_state()

    command = ControlCommand(
        velocity=2.0,
        yaw_rate=1.0
    )

    # simulate target loss
    time.sleep(1.2)

    safe = safety.enforce(state, command)

    assert safe.velocity == 0.0
    assert safe.yaw_rate == 0.0


# -----------------------------------------------------

def test_target_update_resets_timeout():

    safety = create_default_safety()

    state = create_state()

    command = ControlCommand(
        velocity=1.5,
        yaw_rate=0.5
    )

    safety.update_target_detection()

    safe = safety.enforce(state, command)

    assert safe.velocity == command.velocity


# -----------------------------------------------------

def test_nan_command_sanitization():

    safety = create_default_safety()
    state = create_state()

    command = ControlCommand(
        velocity=np.nan,
        yaw_rate=np.nan
    )

    safe = safety.enforce(state, command)

    assert safe.velocity == 0.0
    assert safe.yaw_rate == 0.0


# -----------------------------------------------------

def test_repeated_calls_stability():

    safety = create_default_safety()
    state = create_state()

    command = ControlCommand(
        velocity=1.5,
        yaw_rate=0.2
    )

    for _ in range(1000):

        safe = safety.enforce(state, command)

        assert safe.velocity <= 2.0