# Data Assimilation — NOSSO-MAR

> **Status**: Phase 2 (planned). Reference code in `_archive/OLD/`.
> Design decisions approved in thread `v20260419` (see `THREADS_ARCHIVE.md`).

## Strategy

Data assimilation with neural operators as the forecast model replaces the
classical numerical model in the analysis-forecast cycle. This reduces the
cost of each cycle by 100×–1000×.

## EnKF — Ensemble Kalman Filter

**Reference file**: `_archive/OLD/NOSSO-MAR (old)/src/nosso_mar/assimilation/enkf.py`

- Stochastic EnKF with perturbed observations
- Covariance inflation (configurable factor)
- Optional localisation (for large domains)
- Forecast model = any NOSSO-MAR module with `.run()`

**Example config** (Phase 2):
```yaml
enkf:
  n_ensemble: 100
  inflation: 1.05
  localization_radius_km: 50.0
```

## 4D-Var

**Reference file**: `_archive/OLD/NOSSO-MAR (old)/src/nosso_mar/assimilation/four_dvar.py`

- Automatic differentiation through neural operators replaces the classical adjoint
- Minimiser: L-BFGS
- Configurable assimilation window

## Observation operators

| Sensor | Planned class |
|--------|--------------|
| Wave buoy | `WaveBuoyObsOperator` |
| Satellite altimeter | `SatelliteSwathOperator` |
| Water level (COOPS) | `TideGaugeObsOperator` |

## State vector

```python
WaveStateVector = [eta(x, y), u, v, h, Hs, Tp]
# Serialised as flat tensor for EnKF
```

## Observation data already available (Phase 1)

Downloaded in `data/open_database/downloads/`:
- `coops/9414290_water_level.json` — water level, SF Bay
- `coops/9414290_wind.json` — wind
- `ndbc/41009.txt` — offshore buoy

These datasets will be the first inputs to the EnKF in Phase 2.

## References

- Evensen (2009) — Data Assimilation: The Ensemble Kalman Filter
- Chen et al. (2021) — Neural networks for data assimilation
- `THREADS_ARCHIVE.md` — Thread v20260419: design decisions
