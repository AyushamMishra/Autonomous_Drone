"""
core/clock.py

High-resolution deterministic clock for autonomy stack.

Provides:
- Delta time (dt)
- Loop frequency enforcement
- Timestamp synchronization
- Overrun detection
- Real-time and simulation modes
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeState:
    """Container for timing information."""
    now: float
    dt: float
    cycle_count: int
    overrun: bool


class AutonomyClock:
    """
    Deterministic autonomy clock.

    Modes:
    - real-time mode (default)
    - simulation mode (manual stepping)
    """

    def __init__(self, loop_rate_hz: float = 50.0, simulation: bool = False):
        self.loop_rate_hz = loop_rate_hz
        self.loop_period = 1.0 / loop_rate_hz

        self.simulation = simulation

        self._start_time = None
        self._last_time = None
        self._cycle_count = 0
        self._next_cycle_time = None


    # ==========================================================
    # -------------------- INITIALIZATION ----------------------
    # ==========================================================

    def start(self):
        now = time.perf_counter()
        self._start_time = now
        self._last_time = now
        self._next_cycle_time = now + self.loop_period
        self._cycle_count = 0


    # ==========================================================
    # -------------------- REAL-TIME STEP ----------------------
    # ==========================================================

    def tick(self) -> TimeState:
        """
        Advance one loop cycle (real-time mode).
        Enforces frequency and computes dt.
        """

        if self._last_time is None:
            self.start()

        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        overrun = False

        if not self.simulation:
            sleep_time = self._next_cycle_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                overrun = True
                self._next_cycle_time = time.perf_counter()

            self._next_cycle_time += self.loop_period

        self._cycle_count += 1

        return TimeState(
            now=now,
            dt=dt,
            cycle_count=self._cycle_count,
            overrun=overrun
        )


    # ==========================================================
    # -------------------- SIMULATION STEP ---------------------
    # ==========================================================

    def step_simulation(self, fixed_dt: float) -> TimeState:
        """
        Deterministic fixed-step simulation mode.
        """

        if self._last_time is None:
            self._last_time = 0.0

        self._last_time += fixed_dt
        self._cycle_count += 1

        return TimeState(
            now=self._last_time,
            dt=fixed_dt,
            cycle_count=self._cycle_count,
            overrun=False
        )


    # ==========================================================
    # -------------------- UTILITIES ---------------------------
    # ==========================================================

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    def frequency(self) -> float:
        if self._last_time is None or self._cycle_count < 2:
            return 0.0
        return self._cycle_count / self.elapsed()