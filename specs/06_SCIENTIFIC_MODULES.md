# Specification 06: Scientific Modules (Wetting, Forcing, Objects)

## Purpose
Define modular subsystems that handle domain-specific physics (wetting/drying, atmospheric forcing, object interaction) outside the core neural operator, but explicitly coupled into the forward pass in a disciplined order.

## Scope
- **Phase(s)**: Phase 1 foundation layer, refined in Phase 2
- **Status**: Architecture → Prototyping
- **Dependencies**: IO Contracts (Spec 01), Wave Field Operator (Spec 04)

---

## Module 1: Wetting & Drying

### Purpose
Regularize water column thickness in shallow regions, suppress velocity in dry cells, and handle dynamic coastline transitions.

### Implementation

**Water Mask** (binary indicator):
```
wet(x, y, t) = {
  1 if η(x, y, t) + h(x, y) > h_min,
  0 otherwise
}

h_min = 0.1 m (minimum water depth to avoid numerical issues)
```

**Regularized Depth**:
```
h_reg(x, y, t) = max(η(x, y, t) + h(x, y), h_min) · wet(x, y, t)
```

**Velocity Suppression** (dry-cell velocity zeroed):
```
u_wet(x, y, z, t) = u(x, y, z, t) · wet(x, y, t)
v_wet(x, y, z, t) = v(x, y, z, t) · wet(x, y, t)
```

### Training Considerations
- Include wetting/drying events in training data (~10% of time steps in shallow domains)
- Learn physical-consistent velocity behavior near wet/dry front
- Loss penalty for velocity > 0 in dry regions: L_dry = || u_wet · (1 - wet) ||²

### Related File
- `src/nossomar/modules/wetting_drying.py`

---

## Module 2: Atmospheric Forcing

### Purpose
Inject air-sea stresses (wind) and heat-like surface forcing into the wave propagation operator.

### Wind Stress
```
τ_x = ρ_air · C_d(u_10) · u_10 · |u_10|
τ_y = ρ_air · C_d(u_10) · v_10 · |u_10|

where:
  u_10, v_10 = 10 m wind components [m/s]
  C_d = drag coefficient (typical: 0.0012–0.0025)
  ρ_air = 1.225 kg/m³
```

**Integration into propagation**:
```
∂η/∂t + ∇·(u,v) = 0
∂u/∂t + u·∇u + g·∂η/∂x = τ_x / h + friction
∂v/∂t + v·∇v + g·∂η/∂y = τ_y / h + friction
```

### Data Source
- **ERA5** wind reanalysis (10 m wind, 0.25° resolution, hourly)
- Convert to local domain via bilinear interpolation
- Typically constant over domain for Phase 1 (spatially uniform wind)

### Training Data
- Include 20–30% cases with strong wind forcing (wind-wave coupling)
- Include 10% cases with opposing wind (wave decay)

### Related File
- `src/nossomar/modules/atmospheric_forcing.py`

---

## Module 3: Object Interaction (Fixed & Floating Bodies)

### Purpose
Handle fixed obstacles and floating bodies as source/sink terms in the wave propagation operator. Couple object dynamics (motion, forces) back to wave field.

### Fixed Objects (Breakwaters, Jetties)
```
Representation: solid_mask(x, y) = {1 if obstacle, 0 otherwise}

Boundary condition:
  u·n = 0 (no normal flow through structure)
  
In operator:
  ∇η is masked inside structure
  velocity u, v set to 0 inside
```

### Floating Bodies (WECs, Ships)
```
Body state: [x(t), y(t), z(t), θ(t), vx(t), vy(t), vz(t)]

Forces on body:
  F_hydro = F_excitation + F_radiation + F_buoyancy
  F_pto = -damping · velocity
  F_mooring = -stiffness · displacement
  
Equation of motion:
  m · ∂v/∂t = F_hydro + F_pto + F_mooring

Body-to-wave coupling:
  - Body displaces water (solid mask in wave equation)
  - Radiated wave from body motion affects incident field
  - Radiation damping appears as dissipation term
```

### Implementation Strategy

**Phase 1A (WSI operator only)**:
- Operator learns F_hydro(η, device_props) directly from data
- No explicit wave-body feedback yet

**Phase 1C (Coupling)**:
- Body motion creates radiation wave η_rad(t)
- Added to incident field: η_total = η_incident + η_rad
- Feedback through iterative coupling loop

**Phase 2 (Full object-aware operator)**:
- Objects become explicit nodes in graph neural operator
- Message passing between wave field and bodies
- Joint gradient flow during training

### Practical Implementation
```python
class ObjectInteractionModule:
    def __init__(self, wec_operator, wave_operator):
        self.f1a = wec_operator  # Hydrodynamic forces
        self.f1b = wave_operator # Wave propagation
    
    def forward(self, waves, bathymetry, bodies):
        """
        Args:
            waves: WaveField
            bathymetry: depth field h(x, y)
            bodies: list of Body objects (position, geometry, properties)
        
        Returns:
            waves_updated: WaveField after object effects
            body_response: forces, motion, power for each body
        """
        
        # Step 1: Extract wave kinematics at body locations
        for body in bodies:
            eta_local = waves.sample_at(body.position)
            u_local = waves.u.sample_at(body.position)
            body.kinematics = (eta_local, u_local)
        
        # Step 2: Compute hydrodynamic loads (F1A)
        for body in bodies:
            body.forces = self.f1a.compute_forces(body.kinematics, body.properties)
        
        # Step 3: Update body motion (time integration)
        for body in bodies:
            body.motion = self.time_integrate(body.forces, body.motion)
        
        # Step 4: Compute radiation field (analytical or learned)
        eta_radiation = self.compute_radiation_field(bodies, bathymetry)
        
        # Step 5: Update wave field with radiation
        waves_updated = self.f1b(waves.incident + eta_radiation, bathymetry)
        
        return waves_updated, body_response
```

### Related Files
- `src/nossomar/modules/fixed_obstacles.py`
- `src/nossomar/modules/floating_bodies.py`
- `src/nossomar/modules/multi_object_fsi.py` (array interactions)

---

## Coupling Order (Phase 1 Standard)

**The order matters for physical consistency. Adopted order**:

```
1. Wetting/Drying Regularization
   → ensure h_reg ≥ h_min everywhere
   → prevents division by zero

2. Atmospheric Forcing
   → apply wind stress at surface
   → uniform or spatially-varying (from ERA5)

3. Object Interaction (Optional for Phase 1)
   → sample wave at object locations
   → compute F_hydro (F1A operator)
   → add radiation field
   → (iteratively or end-to-end)

4. Operator Propagation (Core)
   → propagate wave field forward in time
   → F1B operator on modified wave equation

5. Post-Processing
   → apply wet/dry masks to velocity output
   → enforce energy conservation check
   → output final η, u, v, power
```

### Rationale
- **Step 1 first**: Prevents numerical singularities downstream
- **Step 2 before Step 4**: Forcing is boundary condition
- **Step 3 inside 4**: Objects are part of domain, not external
- **Step 5 last**: Clean output, catch inconsistencies

---

## Testing Strategy

### Unit Tests (per module)
```
test_wetting_drying.py:
  - Shallow water case: verify wet/dry front moves correctly
  - Velocity suppression: no flow in dry regions
  
test_atmospheric_forcing.py:
  - Wind-wave growth: wave energy increases with wind
  - Opposing wind: wave energy decays
  
test_object_interaction.py:
  - Single cylinder: computed forces match Capytaine within 5%
  - Two-body array: interaction kernel matches analytical solution
```

### Integration Tests
```
test_coupling_order.py:
  - Full forward pass with all modules
  - Compare against reference solver (OceanWave3D + WEC-Sim)
  - Total energy balance within 5%
```

---

## Configuration Parameters

**Phase 1 Recommended**:
```yaml
wetting_drying:
  enabled: true
  h_min: 0.1 m

atmospheric_forcing:
  enabled: true  (but typically weak wind for Phase 1)
  wind_source: "era5"  or "constant"
  wind_speed: 5 m/s (if constant)

object_interaction:
  enabled: true  (for F1A integration)
  include_radiation: false  (Phase 1 decoupled)
  include_feedback: false
  
coupling_order: ["wetting", "forcing", "objects", "propagation", "postprocess"]
```

---

## Related Files

- **Core modules**: `src/nossomar/modules/`
- **Tests**: `tests/test_modules_*.py`
- **Config**: `configs/modules.yaml`

---

## Future Extensions (Phase 2+)

- **Sediment transport module**: Suspended sediment + bed load evolution
- **Vegetation module**: Seagrass/coral drag
- **Nonlinear breaking module**: Energy dissipation in shoaling zone
- **Viscous damping module**: Bottom friction, surface viscosity
