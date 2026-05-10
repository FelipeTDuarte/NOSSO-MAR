"""
Digital Twin Dashboard — lightweight web interface using FastAPI + Plotly.
Exposes REST endpoints for state, forecast, and alerts.
"""
from __future__ import annotations
from typing import Dict
import json

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app(twin_manager) -> "FastAPI":
    if not HAS_FASTAPI:
        raise ImportError("pip install fastapi uvicorn to use the dashboard")

    app = FastAPI(title="NOSSO-MAR Digital Twin", version="0.1.0")

    @app.get("/state")
    async def get_state():
        if twin_manager.state_estimator:
            st = twin_manager.state_estimator.get_state()
            return JSONResponse({"Hs": st.Hs, "Tp": st.Tp,
                                  "eta_mean": float(st.eta.mean())})
        return JSONResponse({"error": "No state estimator attached"})

    @app.get("/sensors")
    async def list_sensors():
        return JSONResponse({"sensors": list(twin_manager.sensors.keys())})

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "running",
                              "site": twin_manager.cfg.get("site_name")})

    return app
