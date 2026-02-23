"""
Production-grade autonomy orchestration layer.

Coordinates:
- State Estimation
- Planning
- Control
- Perception (optional)
- Behavior (optional)

This module does NOT implement math.
It only orchestrates existing modules through interfaces.
"""
import time
import threading
from typing import Optional
from Core.Clock import AutonomyClock, TimeState

from Core.Interfaces import (
    StateEstimator,
    Planner,
    Controller,
    PerceptionModule,
    BehaviorModule,
    SensorData,
    State,
    Path,
    ControlCommand
)


class AutonomyManager:
    """
    Production-grade autonomy orchestrator.

    Features:
    - Deterministic control loop
    - Thread-safe state access
    - Optional perception + behavior layers
    - Replan gating
    - Graceful shutdown
    """

    def __init__(
        self,
        estimator: StateEstimator,
        planner: Planner,
        controller: Controller,
        perception: Optional[PerceptionModule] = None,
        behavior: Optional[BehaviorModule] = None,
        loop_rate_hz: float = 50.0
    ):
        self.estimator = estimator
        self.planner = planner
        self.controller = controller
        self.perception = perception
        self.behavior = behavior

        self.loop_rate_hz = loop_rate_hz
        self.loop_period = 1.0 / loop_rate_hz
        self.clock = AutonomyClock(loop_rate_hz=loop_rate_hz)

        self._running = False
        self._lock = threading.Lock()

        self._current_state: Optional[State] = None
        self._current_path: Optional[Path] = None
        self._current_command: Optional[ControlCommand] = None

        self._goal = None


    # -------------------- PUBLIC API --------------------------
    def set_goal(self, goal):
        with self._lock:
            self._goal = goal
            self.planner.set_goal(goal)

    def get_state(self) -> Optional[State]:
        with self._lock:
            return self._current_state

    def get_command(self) -> Optional[ControlCommand]:
        with self._lock:
            return self._current_command

    def start(self, sensor_callback):
        """
        Starts real-time autonomy loop.

        sensor_callback must return SensorData.
        """
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(sensor_callback,),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join()

    # -------------------- CORE LOOP ---------------------------
    

    def _run_loop(self, sensor_callback):

        self.clock.start() 

        while self._running:

            # 1. Tick clock (dt + sync)
      
            time_state = self.clock.tick()
 
            # 1. Read Sensors
           
            sensor_data: SensorData = sensor_callback()
            # Synchronize timestamp
            sensor_data.timestamp = time_state.now

            # 2. Estimation

            self.estimator.predict(sensor_data)
            self.estimator.update(sensor_data)

            state = self.estimator.get_state()

            # 3. Perception (optional)
            perception_output = None
            if self.perception is not None:
                perception_output = self.perception.process(sensor_data)

            
            # 4. Behavior (optional)
            
            if self.behavior is not None:
                behavior_goal = self.behavior.update(state, perception_output)
                if behavior_goal is not None:
                    self.planner.set_goal(behavior_goal)

            
            # 5. Planning
            
            if self._goal is not None:
                if self._current_path is None or self.planner.needs_replan():
                    self._current_path = self.planner.plan(state)

            # 6. Control
            if self._current_path is not None:
                command = self.controller.compute_control(
                    state,
                    self._current_path
                )
            else:
                command = None

            # 7. Thread-safe state update
            with self._lock:
                self._current_state = state
                self._current_command = command

            # 8. Overrun Monitoring
            if time_state.overrun:
                print("[WARNING] Control loop overrun")
    # -------------------- SINGLE STEP MODE --------------------

    def step_once(self, sensor_data: SensorData) -> Optional[ControlCommand]:
        """
        Deterministic step (useful for simulation or testing).
        """

        self.estimator.predict(sensor_data)
        self.estimator.update(sensor_data)

        state = self.estimator.get_state()

        perception_output = None
        if self.perception:
            perception_output = self.perception.process(sensor_data)

        if self.behavior:
            behavior_goal = self.behavior.update(state, perception_output)
            if behavior_goal is not None:
                self.planner.set_goal(behavior_goal)

        if self._goal is not None:
            if self._current_path is None or self.planner.needs_replan():
                self._current_path = self.planner.plan(state)

        if self._current_path is not None:
            command = self.controller.compute_control(state, self._current_path)
        else:
            command = None

        with self._lock:
            self._current_state = state
            self._current_command = command

        return command