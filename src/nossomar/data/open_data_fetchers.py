"""Small fetchers for open metocean bootstrap data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
import json

import requests


DEFAULT_TIMEOUT = 60
USER_AGENT = "NOSSO-MAR/0.1 (+https://example.invalid/nosso-mar)"


@dataclass(slots=True)
class DownloadedArtifact:
    """Description of a locally materialised remote artifact."""

    source_id: str
    local_path: str
    retrieved_at: str
    content_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def fetch_ndbc_realtime(output_dir: str | Path, station_id: str = "41009") -> DownloadedArtifact:
    """Download a lightweight NDBC realtime station text feed."""

    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    response = _session().get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    target = Path(output_dir) / "ndbc" / f"{station_id}.txt"
    _write_bytes(target, response.content)
    return DownloadedArtifact(
        source_id="ndbc_stdmet",
        local_path=str(target),
        retrieved_at=datetime.now(UTC).isoformat(),
        content_type=response.headers.get("Content-Type"),
        metadata={"station_id": station_id, "url": url},
    )


def fetch_coops_product(
    output_dir: str | Path,
    station_id: str = "9414290",
    product: str = "water_level",
    days: int = 2,
    units: str = "metric",
    time_zone: str = "gmt",
) -> DownloadedArtifact:
    """Download a NOAA CO-OPS JSON product for a single station."""

    end_date = datetime.now(UTC)
    begin_date = end_date - timedelta(days=days)
    params = {
        "product": product,
        "application": "NOSSO_MAR",
        "station": station_id,
        "begin_date": begin_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "units": units,
        "time_zone": time_zone,
        "format": "json",
    }
    if product == "water_level":
        params["datum"] = "MSL"
    response = _session().get(
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        params=params,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()

    payload = response.json()
    target = Path(output_dir) / "coops" / f"{station_id}_{product}.json"
    _write_json(target, payload)
    return DownloadedArtifact(
        source_id="noaa_coops_water_level",
        local_path=str(target),
        retrieved_at=datetime.now(UTC).isoformat(),
        content_type=response.headers.get("Content-Type"),
        metadata={"station_id": station_id, "product": product, "params": params},
    )


def fetch_oisst_daily_file(
    output_dir: str | Path,
    target_date: date | None = None,
) -> DownloadedArtifact:
    """Download one OISST daily netCDF file."""

    candidate_dates = []
    if target_date is not None:
        candidate_dates.append(target_date)
    candidate_dates.extend(
        [
            date.today() - timedelta(days=14),
            date.today() - timedelta(days=365),
            date(2025, 4, 14),
        ]
    )

    last_error: Exception | None = None
    for chosen_date in candidate_dates:
        yyyymm = chosen_date.strftime("%Y%m")
        yyyymmdd = chosen_date.strftime("%Y%m%d")
        url = (
            "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/"
            f"v2.1/access/avhrr/{yyyymm}/oisst-avhrr-v02r01.{yyyymmdd}.nc"
        )
        try:
            response = _session().get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            target = Path(output_dir) / "oisst" / f"oisst-avhrr-v02r01.{yyyymmdd}.nc"
            _write_bytes(target, response.content)
            return DownloadedArtifact(
                source_id="oisst_v2_1",
                local_path=str(target),
                retrieved_at=datetime.now(UTC).isoformat(),
                content_type=response.headers.get("Content-Type"),
                metadata={"date": chosen_date.isoformat(), "url": url},
            )
        except Exception as exc:  # pragma: no cover - network failures are environment-specific
            last_error = exc
            continue

    assert last_error is not None
    raise last_error


def fetch_remote_metadata_snapshot(
    output_dir: str | Path,
    source_id: str,
    url: str,
) -> DownloadedArtifact:
    """Persist a remote landing page or catalogue page as an HTML snapshot."""

    response = _session().get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    safe_name = source_id.replace("/", "_")
    target = Path(output_dir) / f"{safe_name}.html"
    _write_bytes(target, response.content)
    return DownloadedArtifact(
        source_id=source_id,
        local_path=str(target),
        retrieved_at=datetime.now(UTC).isoformat(),
        content_type=response.headers.get("Content-Type"),
        metadata={"url": url},
    )
