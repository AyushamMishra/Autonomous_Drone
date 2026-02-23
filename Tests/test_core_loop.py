import pytest
from Core.Autonomy_manager import AutonomyManager
from Core.Clock import AutonomyClock as Clock

from Tests.Dummy_perception import DummyPerception
from Tests.Dummy_estimator import DummyEstimator
from Tests.Dummy_planner import DummyPlanner
from Tests.Dummy_Controller import DummyController
from Tests.Dummy_sensor import fake_sensor as DummySensor
from Tests.Dummy_behavior import DummyBehavior
def build_manager():

    return AutonomyManager(
        loop_rate_hz=20,
        perception=DummyPerception(),
        estimator=DummyEstimator(),
        planner=DummyPlanner(),
        controller=DummyController(),
        behavior=DummyBehavior()
    )


def test_single_step_execution():
    manager = build_manager()

    sensor_data = DummySensor()
    command = manager.step_once(sensor_data)
    state = manager.get_state()

    assert state is not None

def test_state_update():
    manager = build_manager()

    

    sensor_data = DummySensor()
    manager.step_once(sensor_data)

    state = manager.get_state()

    assert state is not None