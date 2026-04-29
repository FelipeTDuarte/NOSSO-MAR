# Phase 1 — Wave Propagation NO + WEC FSI NO

## Two-module coupled architecture

```
S(ω,θ), h(x,y), {WEC positions}
         ↓
  WavePropagationNO  (WNO / FNO2d / GeoFNO)   ← OceanWave3D data
         ↓  η(x,y,ω), p_i(ω), u_i(ω)
    WecFSIModule     (DeepONet)                 ← Capytaine / HAMS data
         ↓  A(ω), B(ω), F_ex(ω), RAO, P_i
         └──── F_rad, F_diff ──→ Module 1  (iterative)

P_total = Σ ∫ P_i(ω) dω  →  GA fitness
```

## Key references
- Li et al. (2021) FNO: arxiv.org/abs/2010.08895
- Navaneeth & Chakraborty (2023) WNO: arxiv.org/abs/2205.02191
- Lu et al. (2021) DeepONet: arxiv.org/abs/1910.03193
- Ancellin & Dias (2019) Capytaine
