import time
from Core.Clock import AutonomyClock


def test_delta_time_accuracy():
    target_freq = 50
    clock = AutonomyClock(loop_rate_hz=target_freq)

    clock.start()

    dt_values = []

    for _ in range(20):
        ts = clock.tick()
        dt_values.append(ts.dt)

    avg_dt = sum(dt_values[1:]) / len(dt_values[1:])
    expected_dt = 1.0 / target_freq

    assert abs(avg_dt - expected_dt) < 0.005


def test_timestamp_monotonic():
    clock = AutonomyClock(loop_rate_hz=20)
    clock.start()

    timestamps = []

    for _ in range(10):
        ts = clock.tick()
        timestamps.append(ts.now)

    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1]


def test_loop_frequency_enforcement():
    target_freq = 30
    clock = AutonomyClock(loop_rate_hz=target_freq)
    clock.start()

    start = time.time()

    for _ in range(30):
        clock.tick()

    end = time.time()
    elapsed = end - start

    expected_time = 30 / target_freq

    assert abs(elapsed - expected_time) < 0.1