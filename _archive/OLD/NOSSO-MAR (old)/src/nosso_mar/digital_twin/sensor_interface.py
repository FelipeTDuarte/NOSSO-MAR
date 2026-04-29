"""
Sensor interface abstractions for NOSSO-MAR digital twin.

Provides a unified API for ingesting real-time observations from:
    - Wave buoys (Datawell, Spotter)
    - LiDAR wave scanners
    - ADCP current profilers
    - Satellite altimetry (SWOT, Sentinel-6)
    - WEC onboard sensors (displacement, force)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Observation:
    sensor_id:  str
    timestamp:  datetime
    variables:  Dict[str, float]
    quality:    float = 1.0       # 0=bad, 1=good
    position:   Optional[tuple] = None


class SensorInterface(ABC):
    def __init__(self, sensor_id: str, position: tuple):
        self.sensor_id = sensor_id
        self.position  = position
        self._buffer: List[Observation] = []

    @abstractmethod
    def read(self) -> Observation:
        ...

    def push(self, obs: Observation):
        self._buffer.append(obs)

    def flush(self) -> List[Observation]:
        data, self._buffer = self._buffer, []
        return data


class WaveBuoySensor(SensorInterface):
    """
    Datawell Waverider or Spotter buoy interface.
    Provides: Hs, Tp, Dp, eta_timeseries, spectral density S(f)
    """

    def read(self) -> Observation:
        # In production: connect to MQTT / REST API / serial port
        return Observation(
            sensor_id  = self.sensor_id,
            timestamp  = datetime.utcnow(),
            variables  = {"Hs": 2.0, "Tp": 8.0, "Dp": 180.0},
            position   = self.position,
        )


class LiDARSensor(SensorInterface):
    """
    Wave LiDAR scanner — provides η(x, y) surface maps.
    """

    def __init__(self, sensor_id: str, position: tuple, grid_shape: tuple):
        super().__init__(sensor_id, position)
        self.grid_shape = grid_shape

    def read(self) -> Observation:
        eta_map = np.zeros(self.grid_shape, dtype=np.float32)
        return Observation(
            sensor_id = self.sensor_id,
            timestamp = datetime.utcnow(),
            variables = {"eta_map": eta_map.tolist()},
            position  = self.position,
        )
