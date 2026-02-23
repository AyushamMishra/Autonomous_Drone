"""
Autonomy Stack Interfaces:

Defines clean contracts between:
- Estimation
- Planning
- Control
- Perception
- Behavior

These interfaces allow modular expansion without modifying
core mathematical modules (PF, Costmap, Planner).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

# -------------------- DATA CONTAINERS -----------------------
@dataclass
class State:
    """
    Unified system state representation.

    This wraps PF output without modifying it.
    """
    x: float
    y: float
    yaw: float

    # Optional extensions (do not break current stack)
    z: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    roll: Optional[float] = None
    pitch: Optional[float] = None

    covariance: Optional[np.ndarray] = None


@dataclass
class SensorData:
    """
    Unified sensor input packet.

    Compatible with:
    - Terrain PF (altitude, slope, heading, etc.)
    - Future IMU
    - Future camera
    """
    timestamp: float

    # Terrain PF inputs
    altitude: Optional[float] = None
    slope: Optional[float] = None
    heading: Optional[float] = None
    velocity: Optional[float] = None
    yaw_rate: Optional[float] = None

    # IMU (future)
    accel: Optional[np.ndarray] = None
    gyro: Optional[np.ndarray] = None

    # Camera / perception (future)
    image: Optional[Any] = None


@dataclass
class Path:
    """
    Planner output path.

    Compatible with A*/costmap output.
    """
    waypoints: List[np.ndarray]
    cost: Optional[float] = None


@dataclass
class ControlCommand:
    """
    Output command to actuators.

    Ground robot:
        velocity + yaw_rate

    Drone:
        thrust + body rates
    """
    velocity: Optional[float] = None
    yaw_rate: Optional[float] = None

    thrust: Optional[float] = None
    body_rates: Optional[np.ndarray] = None

# -------------------- CORE INTERFACES -----------------------

class StateEstimator(ABC):
    """
    Interface for all state estimators.
    Terrain PF will implement this directly.
    """

    @abstractmethod
    def predict(self, sensor_data: SensorData) -> None:
        pass

    @abstractmethod
    def update(self, sensor_data: SensorData) -> None:
        pass

    @abstractmethod
    def get_state(self) -> State:
        pass

    @abstractmethod
    def reset(self, state: State) -> None:
        pass


class Planner(ABC):
    """
    Interface for global or local planners.
    Wrap  planner.
    """

    @abstractmethod
    def set_goal(self, goal: np.ndarray) -> None:
        pass

    @abstractmethod
    def plan(self, state: State) -> Path:
        pass

    @abstractmethod
    def needs_replan(self) -> bool:
        pass


class Controller(ABC):
    """
    Converts path/trajectory into actuator commands.
    """

    @abstractmethod
    def compute_control(
        self,
        state: State,
        reference: Path
    ) -> ControlCommand:
        pass


class PerceptionModule(ABC):
    """
    YOLO or future vision modules implement this.
    """

    @abstractmethod
    def process(self, sensor_data: SensorData) -> Any:
        pass


class BehaviorModule(ABC):
    """
    Mission logic / mode switching.
    """

    @abstractmethod
    def update(self, state: State, perception: Any) -> Any:
        pass

# -------------------- COSTMAP LAYER -------------------------

class CostmapLayer(ABC):
    """
    Plugin interface for costmap layers.
    Allows dynamic obstacle layers without
    rewriting costmap core.
    """

    @abstractmethod
    def update(self, state: State) -> None:
        pass

    @abstractmethod
    def get_cost(self, x: float, y: float) -> float:
        pass