import time
from Core.Autonomy_manager import AutonomyManager

from Tests.Dummy_estimator import DummyEstimator
from Tests.Dummy_planner import DummyPlanner
from Tests.Dummy_Controller import DummyController
from Tests.Dummy_sensor import fake_sensor
from Tests.Dummy_behavior import DummyBehavior
from Tests.Dummy_perception import DummyPerception


def build_manager():
    return AutonomyManager(
        estimator=DummyEstimator(),
        planner=DummyPlanner(),
        controller=DummyController(),
        perception=DummyPerception(),
        behavior=DummyBehavior(),
        loop_rate_hz=50
    )


def test_full_integration_run():
    manager = build_manager()

    manager.start(fake_sensor)

    time.sleep(1.0)

    manager.stop()

    state = manager.get_state()

    assert state is not None


def test_graceful_shutdown():
    manager = build_manager()

    manager.start(fake_sensor)

    time.sleep(0.5)

    manager.stop()

    assert manager._running is False