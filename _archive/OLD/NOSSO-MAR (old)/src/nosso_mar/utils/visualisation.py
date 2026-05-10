"""
Visualisation utilities for NOSSO-MAR outputs.
Requires: matplotlib, optionally cartopy for geographic plots.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def plot_wave_field(eta: np.ndarray, x: np.ndarray, y: np.ndarray,
                    wec_positions: Optional[List] = None,
                    title: str = "Wave Field η [m]",
                    save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.pcolormesh(x, y, eta, cmap="RdBu_r",
                       vmin=-np.abs(eta).max(), vmax=np.abs(eta).max())
    plt.colorbar(im, ax=ax, label="η [m]")

    if wec_positions:
        xs = [p[0] for p in wec_positions]
        ys = [p[1] for p in wec_positions]
        ax.scatter(xs, ys, c="black", s=80, zorder=5, label="WEC")
        ax.legend()

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_bem_response(omega: np.ndarray, A: np.ndarray, B: np.ndarray,
                       F_ex: np.ndarray, rao: np.ndarray,
                       save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [("Added mass A(ω) [kg]", A),
             ("Radiation damping B(ω) [N·s/m]", B),
             ("|F_ex(ω)| [N]", F_ex),
             ("RAO(ω) [m/m]", rao)]
    for ax, (label, data) in zip(axes.flat, pairs):
        ax.plot(omega, data, linewidth=2)
        ax.set_xlabel("ω [rad/s]"); ax.set_ylabel(label); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_farm_layout(positions: List[Tuple], farm_bounds: List,
                     power: Optional[List] = None,
                     save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    c  = power if power else "steelblue"
    sc = ax.scatter(xs, ys, c=c, s=200, cmap="viridis", zorder=5)
    if power:
        plt.colorbar(sc, ax=ax, label="Absorbed power [W]")
    ax.set_xlim(farm_bounds[0]); ax.set_ylim(farm_bounds[1])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("WEC Farm Layout")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
