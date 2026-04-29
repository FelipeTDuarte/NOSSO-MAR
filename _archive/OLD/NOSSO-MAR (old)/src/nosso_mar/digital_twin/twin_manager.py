"""
Digital Twin Manager — orchestrates the full NOSSO-MAR digital twin.

Responsibilities:
    1. Ingest real-time sensor observations
    2. Run EnKF state estimation cycle
    3. Run NO forecast for horizon prediction
    4. Trigger MARL agents for WEC control decisions
    5. Detect anomalies and raise alerts
    6. Log and persist twin state to storage backend
"""
from __future__ import annotations
from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DigitalTwinManager:
    """
    Central orchestrator for the NOSSO-MAR ocean digital twin.

    cfg keys:
        site_name          : str
        grid_shape         : [H, W]
        assimilation_cycle : float  (seconds between DA cycles)
        forecast_horizon   : float  (seconds)
        storage_backend    : str    (local | s3 | zarr)
    """

    def __init__(self, cfg: Dict,
                 wave_no=None, fsi_module=None,
                 state_estimator=None, anomaly_detector=None,
                 marl_controller=None):
        self.cfg               = cfg
        self.wave_no           = wave_no
        self.fsi_module        = fsi_module
        self.state_estimator   = state_estimator
        self.anomaly_detector  = anomaly_detector
        self.marl_controller   = marl_controller
        self.sensors: Dict     = {}
        self._running          = False

    def register_sensor(self, sensor):
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Registered sensor: {sensor.sensor_id}")

    async def _assimilation_cycle(self):
        while self._running:
            obs = {}
            for sid, sensor in self.sensors.items():
                raw = sensor.read()
                if raw.quality > 0.5:
                    import torch
                    obs[sid] = torch.tensor(list(raw.variables.values()))

            if self.state_estimator and obs:
                import torch
                R_diag = torch.ones(sum(len(v) for v in obs.values())) * 0.01
                self.state_estimator.update(obs, R_diag)
                logger.debug(f"DA cycle completed at {datetime.utcnow()}")

            await asyncio.sleep(self.cfg.get("assimilation_cycle", 300))

    async def _forecast_cycle(self):
        while self._running:
            if self.state_estimator and self.wave_no:
                state = self.state_estimator.get_state()
                # Run forecast
                forecast = self.wave_no.run({
                    "spectrum":   {"Hs": state.Hs, "Tp": state.Tp},
                    "bathymetry": state.bathy,
                })
                logger.debug("Forecast updated")
            await asyncio.sleep(self.cfg.get("forecast_cycle", 600))

    def start(self):
        self._running = True
        loop = asyncio.get_event_loop()
        loop.create_task(self._assimilation_cycle())
        loop.create_task(self._forecast_cycle())
        logger.info(f"Digital twin started: {self.cfg.get('site_name', 'NOSSO-MAR')}")

    def stop(self):
        self._running = False
        logger.info("Digital twin stopped.")
