# Specification 01: IO Contracts & Data Types

## Purpose
Define the canonical data structures, tensor conventions, coordinate systems, and IO interfaces between modules. Freezing these early prevents architecture drift and enables modular development.

## Scope
- **Phase(s)**: Phase 0 (Task 0) → Phase 1 foundation
- **Status**: Draft → Implement in `src/nossomar/core/contracts.py`
- **Dependencies**: None (foundational)

---

## Key Abstractions

### 1. WECState (Wave Energy Converter State)

**Purpose**: Encapsulate hydrodynamic properties and response of a single device or array.

**Fields** (xarray.Dataset):
```
- A(ω):              Radiation added mass [kg], per frequency bin
- B(ω):              Radiation damping [N·s/m], per frequency bin  
- Fex(ω):            Excitation force amplitude [N], per frequency bin
- RAO(ω, heading):   Response amplitude operator [m/N], frequency × heading bins
- motion(t):         Position/velocity/acceleration of body [m, m/s, m/s²]
- power_absorbed(t): Instantaneous power [W]
- power_reactive(t): Reactive power [VAR]
- interaction(i,j):  Hydrodynamic interaction kernel between device i and j
```

**Coordinates**:
- `freq`: Frequency axis [Hz] — typically 0.1–2.0 Hz, resolution ≥ 0.01 Hz
- `heading`: Wave direction [deg] — 0–360°, resolution ≥ 15°
- `time`: Temporal axis [s]
- `device_id`: Device index in array

**Attributes** (metadata):
```
- device_type: str ("cylinder", "sphere", "heave_plate", etc.)
- radius: float [m]
- draft: float [m]
- mass: float [kg]
- bpto: float [N·s/m] — PTO damping
- depth: float [m] — water depth at device location
- mooring_stiffness: float [N/m] or None if fixed
```

---

### 2. WaveField (Phase-Resolved Wave State)

**Purpose**: Represent the spatial and temporal wave field over the computational domain.

**Fields** (xarray.DataArray):
```
- η(x, y, t):        Surface elevation [m] — regular grid
- u(x, y, z, t):     Horizontal velocity [m/s]
- v(x, y, z, t):     Vertical velocity [m/s]
- w(x, y, z, t):     Normal velocity (if z in sigma coords) [m/s]
- p(x, y, z, t):     Pressure (relative to hydrostatic) [Pa]
```

**Coordinates**:
- `x`, `y`: Cartesian spatial grid [m] — resolution depends on application (0.1–10 m)
- `z` or `sigma`: Vertical coordinate — depth or normalized (sigma) coordinates
- `time`: [s] — synchronized with WECState.time

**Attributes**:
```
- bathymetry_file: str — reference to bathymetry source
- depth_field: xr.DataArray [m] — depth at each (x,y)
- solver_reference: str — "OceanWave3D", "SWASH", "FUNWAVE", etc.
- wave_regime: str — "linear", "nonlinear", "breaking"
- coordinate_system: str — "cartesian_xy" or "spherical_lonlat"
```

---

### 3. OceanState (Full Coupled System)

**Purpose**: Combine WaveField + WECState + environmental forcing into one portable dataset.

**Structure**:
```python
OceanState = {
    "wave_field": WaveField,           # Phase-resolved η, velocities, pressure
    "wec_states": [WECState, ...],     # Array of device states
    "bathymetry": xr.DataArray,        # Depth field h(x,y)
    "forcing": {                       # External boundary conditions
        "wind": xr.DataArray,          # 10m wind [U, V] [m/s]
        "pressure": xr.DataArray,      # Atmospheric pressure [Pa]
        "open_bc": xr.DataArray,       # Open boundary wave spectrum [m²/Hz]
    },
    "metadata": {
        "case_id": str,                # Unique identifier
        "timestamp": datetime,         # Real-world time or simulation start
        "domain_center": (lat, lon),   # For georeferenced domains
        "config": dict,                # All hyperparams + solver settings
    }
}
```

---

## Coordinate Conventions

### Horizontal
- **Cartesian (small domain)**: x [m], y [m] — origin at domain corner
- **Spherical (large domain)**: lon [deg], lat [deg] — WGS84

### Vertical
- **Absolute depth z**: z = 0 at surface, z < 0 downward (z = -h at bottom)
- **Sigma coordinates σ**: σ ∈ [0, 1], σ=0 at surface, σ=1 at bottom
  - Conversion: `z(σ) = σ(h(x,y) - η(x,y,t)) + η(x,y,t)`
  - Advantage: Fixed grid resolution, handles wetting/drying naturally

### Time
- Absolute: seconds from simulation start, or UNIX timestamp
- Resolution: Δt ≤ 0.1 T_peak (T_peak = peak wave period)

---

## IO Formats

### Interchange Format: **NetCDF/Zarr**

**Why**: Self-documenting, structured grid native, chunked I/O for large datasets.

**Requirements**:
```
- One .nc/.zarr file per case per time chunk
- Dimensions: (time, x, y, z, freq, heading)
- Coordinates stored explicitly
- Attributes: units, long_name, solver_ref, creation_date
- Compression: zlib level 4 (balance speed/size)
```

### Python API

```python
# Load from file
ocean_state = OceanState.from_netcdf("case_001.nc")

# Access WaveField
eta = ocean_state.wave_field.η  # xr.DataArray
u, v = ocean_state.wave_field.u, ocean_state.wave_field.v

# Access WECState
for wec in ocean_state.wec_states:
    A_omega = wec.A  # xr.DataArray (freq,)
    power = wec.power_absorbed  # xr.DataArray (time,)

# Save
ocean_state.to_netcdf("output.nc", encoding_profile="phase1")
```

---

## Physical Units & Sign Conventions

| Variable | Unit | Sign |
|----------|------|------|
| η (elevation) | m | η > 0 above SWL |
| u, v (horiz. vel) | m/s | u > 0 westward, v > 0 northward |
| w (vert. vel) | m/s | w > 0 upward |
| A(ω) (added mass) | kg | Always ≥ 0 |
| B(ω) (damping) | N·s/m | Always ≥ 0 |
| Fex (excitation) | N | Convention: positive upward |
| Power | W | Power > 0 ⟹ absorbed by PTO |
| Depth h | m | h > 0 (always positive, measure downward) |

---

## Validation Rules

1. **Temporal consistency**: WaveField.time ⊂ WECState.time (wave field is denser or equal)
2. **Spatial consistency**: WEC locations must be within domain bounds
3. **Physical bounds**: 
   - A(ω) ≤ ρ·g·A_geom (added mass ≤ water mass of displaced volume)
   - B(ω) ≥ 0 (damping always dissipative)
   - η ≤ depth (surface elevation below water surface)
4. **Metadata completeness**: device_type, depth, solver_reference required

---

## Implementation Checklist

- [ ] Define WECState class (xarray-backed)
- [ ] Define WaveField class (xarray-backed)
- [ ] Define OceanState container
- [ ] Write to_netcdf() / from_netcdf() methods
- [ ] Validate coordinate conventions
- [ ] Unit tests: 10 cases (canonical problems)
- [ ] Document in `docs/API/contracts.md`

---

## Related Files

- **GitHub**: `src/nossomar/core/contracts.py`
- **Tests**: `tests/test_contracts.py`
- **Config template**: `configs/phase1_domain.yaml`
