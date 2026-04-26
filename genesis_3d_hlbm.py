"""
3D HLBM-inspired fluid simulator for a 3x3x3 I-Ching cube.

The grid represents three vertical layers: Earth, Human, and Heaven.
Energy follows a 27-palace flying-star path with 順/逆 (forward/reverse)
cycling: each block of 27 time steps walks the full ring in one direction,
then the next block reverses. A HeTu (河圖) pair constraint keeps symmetric
palaces (through center 5) summing to a balance constant.

Optional stress test: a large pulse at an Earth corner excites the field;
layer sums and pre-constraint HeTu deviations are recorded, with stability
coefficients σ, ω for boundedness diagnostics.

Brute sweep: amplitudes from 88 to 1000; for each, record max |grad E|,
energy dissipation rate, and spatial std σ_E, then plot boundedness of σ_E
under extreme pulses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

GRID_SIZE = 3
CENTER = np.array([1, 1, 1], dtype=float)
BALANCE_CONSTANT = 10.0
TIME_STEPS = 100
OUTPUT_FILE = Path("genesis_3d_result.png")
OUTPUT_GIF = Path("genesis_3d_time.gif")
OUTPUT_MP4 = Path("genesis_3d_time.mp4")
OUTPUT_HETU_PLOT = Path("hetu_stability.png")
OUTPUT_STRESS_LAYERS = Path("stress_test_layers.png")
OUTPUT_BRUTAL_SWEEP = Path("brute_sweep_stability.png")
OUTPUT_PULSE_1000_MP4 = Path("pulse_1000_aftershock.mp4")
OUTPUT_CLINICAL_COMPARE = Path("clinical_lbm_hlbm_sigma.png")
CLINICAL_PULSE = {
    "enabled": True,
    "coord": (0, 0, 0),
    "at_step_0based": 9,
    "amplitude": 1000.0,
}
CLINICAL_TIME_STEPS = 100

PULSE_1000_HIRES_STEPS = 400
PULSE_1000_FPS = 30
PULSE_1000_DPI = 200

STRESS_PULSE = {
    "enabled": True,
    "coord": (0, 0, 0),
    "at_step_0based": 4,
    "amplitude": 1000.0,
}
PULSE_AMP_SWEEP = np.unique(
    np.round(
        np.logspace(np.log10(88.0), np.log10(1000.0), 18),
        4,
    )
)
STRESS_BLOWUP_ENERGY = 1.0e9
STRESS_BLOWUP_MSE = 1.0e12


@dataclass(frozen=True)
class Palace:
    index: int
    name: str
    coord: tuple[int, int, int]


def build_twenty_seven_palaces() -> list[Palace]:
    layer_names = ("Earth", "Human", "Heaven")
    spiral_2d = (
        (0, 0),(1, 0),(2, 0),(2, 1),(2, 2),(1, 2),(0, 2),(0, 1),(1, 1),
    )
    palaces: list[Palace] = []
    for z, layer_name in enumerate(layer_names):
        path = spiral_2d if z % 2 == 0 else tuple(reversed(spiral_2d))
        for x, y in path:
            palaces.append(Palace(index=len(palaces)+1, name=f"{layer_name}-{len(palaces)+1:02d}", coord=(x,y,z)))
    return palaces


def opposite_coord(coord: tuple[int, int, int]) -> tuple[int, int, int]:
    opposite = 2 * CENTER - np.array(coord, dtype=float)
    return tuple(int(round(v)) for v in opposite)


def apply_hetu_constraint(energy: np.ndarray) -> np.ndarray:
    constrained = energy.copy()
    visited: set[tuple[int, int, int]] = set()
    center = tuple(int(c) for c in CENTER)
    for coord in np.ndindex(energy.shape):
        if coord in visited or coord == center:
            continue
        pair = opposite_coord(coord)
        pair_sum = constrained[coord] + constrained[pair]
        correction = 0.5 * (BALANCE_CONSTANT - pair_sum)
        constrained[coord] += correction
        constrained[pair] += correction
        visited.add(coord)
        visited.add(pair)
    constrained[center] = 5.0
    return constrained


def star_step_direction(t: int) -> int:
    return 1 if (t // 27) % 2 == 0 else -1


def compute_hetu_metrics(energy: np.ndarray) -> dict[str, float]:
    center = tuple(int(c) for c in CENTER)
    seen: set[tuple[int, int, int]] = set()
    pair_sq: list[float] = []
    for coord in np.ndindex(energy.shape):
        if coord in seen or coord == center:
            continue
        pair = opposite_coord(coord)
        s = float(energy[coord] + energy[pair])
        pair_sq.append((s - BALANCE_CONSTANT) ** 2)
        seen.add(coord)
        seen.add(pair)
    e_pair = float(np.mean(pair_sq)) if pair_sq else 0.0
    h_tot = float(np.sum(energy))
    return {"H_tot": h_tot, "E_pair_mse": e_pair, "max_pair_abs_dev": float(np.sqrt(max(pair_sq))) if pair_sq else 0.0}


def max_pair_abs_dev_for_heaven_pairs(energy: np.ndarray) -> float:
    center = tuple(int(c) for c in CENTER)
    seen: set[tuple[int, int, int]] = set()
    mxd = 0.0
    for coord in np.ndindex(energy.shape):
        if coord in seen or coord == center:
            continue
        pair = opposite_coord(coord)
        if int(coord[2]) != 2 and int(pair[2]) != 2:
            seen.add(coord); seen.add(pair)
            continue
        s = float(energy[coord] + energy[pair])
        mxd = max(mxd, abs(s - BALANCE_CONSTANT))
        seen.add(coord); seen.add(pair)
    return mxd


def layer_sums(energy: np.ndarray) -> tuple[float, float, float]:
    return (float(np.sum(energy[:,:,0])), float(np.sum(energy[:,:,1])), float(np.sum(energy[:,:,2])))


def compute_stability_coefficient(max_pair_dev_pre, mse_pre_max, h_tots, energies):
    b = float(BALANCE_CONSTANT)
    any_bad = any((not np.isfinite(e).all() or (np.abs(e) > STRESS_BLOWUP_ENERGY).any()) for e in energies)
    if not h_tots or not all(np.isfinite(h) for h in h_tots):
        return 0.0, False, "H_tot 或場含非有限值"
    if mse_pre_max > STRESS_BLOWUP_MSE or max_pair_dev_pre > 1.0e6 * b or any_bad:
        return 0.0, False, "偏離或場值超過安全閾值，視為未守住"
    m = max(float(max_pair_dev_pre), 0.0)
    sigma = float(np.exp(-m / b))
    return sigma, True, f"sup|pair-B|_（約束前）= {m:.4f} (B={b}); σ=exp(-M/B)={sigma:.6f}"


def max_energy_gradient(energy: np.ndarray) -> float:
    gx, gy, gz = np.gradient(energy)
    return float(np.sqrt(np.max(gx*gx + gy*gy + gz*gz)))


def energy_dissipation_rate(energies: list[np.ndarray]) -> float:
    if len(energies) < 2:
        return 0.0
    s = sum(float(np.sum(np.abs(energies[t] - energies[t-1]))) / 27.0 for t in range(1, len(energies)))
    return s / float(len(energies) - 1)


def energy_field_std(energy: np.ndarray) -> float:
    return float(np.std(energy.ravel()))


def simulation_collapsed(energies: list[np.ndarray]) -> bool:
    for e in energies:
        if not np.isfinite(e).all() or (np.abs(e) > STRESS_BLOWUP_ENERGY).any():
            return True
    return False


def run_brutal_sweep(path_coords):
    base = {**STRESS_PULSE, "enabled": True}
    records = []
    all_ok = True
    for A in PULSE_AMP_SWEEP:
        cfg = {**base, "amplitude": float(A)}
        en, _ve, _si, _mh = run_time_loops(path_coords, stress_pulse=cfg, verbose=False)
        if not en:
            records.append({"amplitude": float(A), "max_grad": float("nan"), "edr": float("nan"), "std_E_final": float("nan"), "std_E_max": float("nan"), "ok": False})
            all_ok = False
            break
        mg = max(max_energy_gradient(e) for e in en)
        edr = energy_dissipation_rate(en)
        s_final = energy_field_std(en[-1])
        s_max = max(energy_field_std(e) for e in en)
        collapsed = simulation_collapsed(en)
        ok = not collapsed
        if not ok:
            all_ok = False
        records.append({"amplitude": float(A), "max_grad": mg, "edr": edr, "std_E_final": s_final, "std_E_max": s_max, "ok": ok})
    return all_ok, records


def save_brutal_sweep_plot(records):
    if not records:
        return
    fin = [r for r in records if np.isfinite(r.get("max_grad", float("nan")))]
    if not fin:
        return
    a = np.array([float(r["amplitude"]) for r in fin])
    g = np.array([float(r["max_grad"]) for r in fin])
    d = np.array([float(r["edr"]) for r in fin])
    sef = np.array([float(r["std_E_final"]) for r in fin])
    semx = np.array([float(r["std_E_max"]) for r in fin])
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    ax0, ax1, ax2 = axes
    ax0.semilogy(a, np.maximum(g, 1e-12), "o-", color="navy", markersize=4)
    ax0.set_ylabel("Max |grad E|"); ax0.set_title("Brute: max gradient"); ax0.grid(True, alpha=0.3)
    ax1.plot(a, d, "s-", color="darkgreen", markersize=4)
    ax1.set_ylabel("EDR (L1 / step)"); ax1.set_title("Energy dissipation (mixing) rate"); ax1.grid(True, alpha=0.3)
    ax2.plot(a, sef, "D-", color="crimson", label=r"$\sigma_E$ at $t=T$", markersize=4)
    ax2.plot(a, semx, "^--", color="darkorange", alpha=0.85, label=r"$\max_t \sigma_E(t)$")
    ax2.set_ylabel(r"$\sigma_E$ (spatial std)"); ax2.set_xlabel("STRESS_PULSE amplitude (88 → 1000)")
    ax2.set_title(r"$\sigma_E$ vs pulse: stay in finite range?"); ax2.grid(True, alpha=0.3)
    lo = min(float(np.min(sef)), float(np.min(semx)))
    hi = max(float(np.max(sef)), float(np.max(semx)))
    ax2.axhspan(lo, hi, color="plum", alpha=0.12, zorder=0)
    ax2.legend(loc="best", fontsize=8)
    fig.suptitle("Brute force sweep: stress pulse amplitude", y=0.995)
    fig.tight_layout()
    fig.savefig(OUTPUT_BRUTAL_SWEEP, dpi=150, bbox_inches="tight")
    plt.close(fig)


def running_sigma_cumulative(metrics_history, dev_key="max_pair_abs_dev"):
    b = float(BALANCE_CONSTANT)
    mrun = 0.0
    out = []
    for m in metrics_history:
        v = float(m.get(dev_key, m["max_pair_abs_dev"]))
        mrun = max(mrun, v)
        out.append(float(np.exp(-mrun / b)))
    return out


def run_clinical_control_experiment(path_coords):
    print("\n========== 臨床對照實驗：Standard LBM vs HLBM ==========")
    sp = {**CLINICAL_PULSE, "enabled": True}
    _, _, _, m_std = run_time_loops(path_coords, stress_pulse=sp, verbose=False, time_steps=CLINICAL_TIME_STEPS, hlbm=False)
    _, _, _, m_hl = run_time_loops(path_coords, stress_pulse=sp, verbose=False, time_steps=CLINICAL_TIME_STEPS, hlbm=True)
    t_axis = np.arange(1, len(m_std) + 1, dtype=float)
    b = float(BALANCE_CONSTANT)
    sig_hv_std = running_sigma_cumulative(m_std, "max_pair_dev_heaven")
    sig_hv_hl = running_sigma_cumulative(m_hl, "max_pair_dev_heaven")
    m_hv_std = np.array([m["max_pair_dev_heaven"] for m in m_std], dtype=float)
    m_hv_hl = np.array([m["max_pair_dev_heaven"] for m in m_hl], dtype=float)
    sig_step_std = np.exp(-m_hv_std / b)
    sig_step_hl = np.exp(-m_hv_hl / b)
    e2_std = [m["E_heaven"] for m in m_std]
    e2_hl = [m["E_heaven"] for m in m_hl]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True)
    axm, ax0, ax1 = axes
    axm.plot(t_axis, m_hv_std, "m--", label=r"Heaven $m(t)$: Standard", linewidth=1.3)
    axm.plot(t_axis, m_hv_hl, "c-", label=r"Heaven $m(t)$: HLBM", linewidth=1.4)
    axm.axvline(10, color="crimson", linestyle=":", alpha=0.75, label="Earth pulse t=10")
    axm.set_ylabel(r"$m(t)$"); axm.set_title("A: Imbalance on heaven-linked pairs")
    axm.legend(loc="upper right", fontsize=7); axm.grid(True, alpha=0.3); axm.set_yscale("symlog", linthresh=5.0)
    ax0.plot(t_axis, sig_hv_std, color="darkorange", linestyle="--", linewidth=1.2, label=r"Cumul. $\sigma$ Standard")
    ax0.plot(t_axis, sig_hv_hl, color="darkgreen", linestyle="-", linewidth=1.3, label=r"Cumul. $\sigma$ HLBM")
    ax0.plot(t_axis, sig_step_std, "m--", alpha=0.55, linewidth=0.9, label=r"Step $\sigma$ Standard")
    ax0.plot(t_axis, sig_step_hl, "c-", alpha=0.7, linewidth=0.9, label=r"Step $\sigma$ HLBM")
    ax0.axvline(10, color="crimson", linestyle=":", alpha=0.75)
    ax0.set_ylabel(r"$\sigma$"); ax0.set_title("B: Stability coefficient"); ax0.set_ylim(0.0, 1.02)
    ax0.legend(loc="lower left", fontsize=6); ax0.grid(True, alpha=0.3)
    ax1.plot(t_axis, e2_std, "m--", label="Heaven: Standard LBM", linewidth=1.4)
    ax1.plot(t_axis, e2_hl, "c-", label="Heaven: HLBM", linewidth=1.5)
    ax1.axvline(10, color="crimson", linestyle=":", alpha=0.75)
    ax1.set_xlabel("Time step t"); ax1.set_ylabel(r"Heaven $\sum E$"); ax1.set_title("C: Heaven layer energy")
    ax1.legend(loc="best", fontsize=7); ax1.grid(True, alpha=0.3)
    fig.suptitle("Clinical control: standard LBM vs 27G HLBM", y=0.998, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_CLINICAL_COMPARE, dpi=150, bbox_inches="tight")
    plt.close(fig)


def one_physics_substep(energy, velocity, current, next_coord, *, use_hetu=True, use_flying_star=True):
    direction = np.array(next_coord, dtype=float) - np.array(current, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        direction /= norm
    energy = energy.copy()
    current_k = (int(current[0]), int(current[1]), int(current[2]))
    vel = velocity.copy()
    n_cells = float(GRID_SIZE**3)
    if use_flying_star:
        energy[current_k] += 0.18
        vel[current_k] = 0.82 * vel[current_k] + 0.18 * direction
    else:
        energy += 0.18 / n_cells
        vel *= 0.95
    streamed = energy.copy()
    for coord in np.ndindex(energy.shape):
        neighbor_total = 0.0
        neighbor_count = 0
        x, y, z_ = coord
        for dx, dy, dz in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
            neighbor = (x+dx, y+dy, z_+dz)
            if all(0 <= v < GRID_SIZE for v in neighbor):
                neighbor_total += energy[neighbor]
                neighbor_count += 1
        if neighbor_count:
            streamed[coord] = 0.72 * energy[coord] + 0.28 * (neighbor_total / neighbor_count)
    pre_metrics = compute_hetu_metrics(streamed)
    center_t = tuple(int(c) for c in CENTER)
    pre_metrics = {**pre_metrics, "E_center_pre": float(streamed[center_t]), "max_pair_dev_heaven": float(max_pair_abs_dev_for_heaven_pairs(streamed))}
    constrained = apply_hetu_constraint(streamed) if use_hetu else streamed
    return constrained, vel, pre_metrics


def run_time_loops(path_coords, stress_pulse=None, verbose=True, time_steps=None, *, hlbm=True):
    sp = STRESS_PULSE if stress_pulse is None else stress_pulse
    n_steps = int(time_steps) if time_steps is not None else int(TIME_STEPS)
    energy = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), 1.0, dtype=float)
    if hlbm:
        energy = apply_hetu_constraint(energy)
    velocity = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), dtype=float)
    n = len(path_coords)
    energies, velocities, star_indices, metrics_history = [], [], [], []
    center_t = tuple(int(c) for c in CENTER)
    star_idx = 0
    for t in range(n_steps):
        if sp.get("enabled") and t == int(sp["at_step_0based"]):
            pc = sp["coord"]
            energy[pc[0], pc[1], pc[2]] += float(sp["amplitude"])
        current = path_coords[star_idx]
        d = star_step_direction(t)
        next_idx = (star_idx + d) % n
        next_coord = path_coords[next_idx]
        energy, velocity, m_pre = one_physics_substep(energy, velocity, current, next_coord, use_hetu=hlbm, use_flying_star=hlbm)
        m_post = compute_hetu_metrics(energy)
        e0, e1, e2 = layer_sums(energy)
        ht = m_post["H_tot"] or 1.0
        m = {"H_tot": m_post["H_tot"], "E_pair_mse": m_pre["E_pair_mse"], "max_pair_abs_dev": m_pre["max_pair_abs_dev"],
             "max_pair_dev_heaven": m_pre.get("max_pair_dev_heaven", 0.0), "E_pair_mse_post": m_post["E_pair_mse"],
             "E_earth": e0, "E_human": e1, "E_heaven": e2, "f_heaven": float(e2/ht),
             "E_center_pre": float(m_pre.get("E_center_pre", 5.0)), "E_center_post": float(energy[center_t])}
        energies.append(energy.copy()); velocities.append(velocity.copy())
        star_indices.append(star_idx); metrics_history.append(m)
        star_idx = next_idx
    return energies, velocities, star_indices, metrics_history


def build_vector_field(energy, velocity):
    x, y, z = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij")
    emax = float(np.max(energy)) or 1.0
    radial = np.stack((x,y,z), axis=-1).astype(float) - CENTER
    radial_norm = np.linalg.norm(radial, axis=-1, keepdims=True)
    radial = np.divide(radial, radial_norm, out=np.zeros_like(radial), where=radial_norm>0)
    magnitude = energy / emax
    vector = velocity + 0.38 * radial * magnitude[..., np.newaxis]
    center = tuple(int(c) for c in CENTER)
    vector[center] = np.array([0.0, 0.0, 0.72])
    return x, y, z, vector[...,0], vector[...,1], vector[...,2]


def path_energy_colors(energy, path_coords):
    return np.array([float(energy[c]) for c in path_coords], dtype=float)


def save_last_frame_png(palaces, path_coords, energy, velocity, star_idx):
    x, y, z, u, v, w = build_vector_field(energy, velocity)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(x, y, z, u, v, w, length=0.42, normalize=True, color="royalblue")
    path = np.array([p.coord for p in palaces], dtype=float)
    ax.plot(path[:,0], path[:,1], path[:,2], color="goldenrod", linewidth=2.2)
    cvals = path_energy_colors(energy, path_coords)
    ax.scatter(path[:,0], path[:,1], path[:,2], c=cvals, cmap="viridis", s=90)
    cx, cy, cz = path_coords[star_idx]
    ax.scatter([cx],[cy],[cz], color="crimson", s=200, label="飛星")
    center = tuple(int(c) for c in CENTER)
    ax.scatter([center[0]],[center[1]],[center[2]], color="k", s=30, label="河圖中五")
    ax.set_title("3D HLBM Genesis (最後一幀)"); ax.view_init(elev=26, azim=42)
    fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_stress_layer_plot(metrics_history):
    t = np.arange(1, len(metrics_history)+1)
    e0 = [m["E_earth"] for m in metrics_history]
    e1 = [m["E_human"] for m in metrics_history]
    e2 = [m["E_heaven"] for m in metrics_history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, e0, label="Earth (地)", color="saddlebrown", linewidth=1.5)
    ax.plot(t, e1, label="Human (人)", color="olive", linewidth=1.5)
    ax.plot(t, e2, label="Heaven (天)", color="mediumpurple", linewidth=1.5)
    if STRESS_PULSE.get("enabled"):
        ax.axvline(int(STRESS_PULSE["at_step_0based"])+1, color="crimson", linestyle="--", alpha=0.7, label="衝脈")
    ax.set_xlabel("t"); ax.set_ylabel("Layer sum"); ax.set_title("Pressure test: conduction to Heaven")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_STRESS_LAYERS, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_hetu_stability_plot(h_tot_trace, e_pair_trace):
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(1, len(h_tot_trace)+1)
    ax.plot(t, h_tot_trace, "b-", label=r"$H_{tot}$ 河圖總能量")
    ax2 = ax.twinx()
    ax2.plot(t, e_pair_trace, "r--", label=r"$E_{pair,pre}$ 約束前 MSE")
    ax.set_xlabel("t"); ax.set_ylabel(r"$H_{tot}$", color="b"); ax2.set_ylabel(r"$E_{pair,pre}$", color="r")
    ax.set_title("100 步內 河圖 動態有界性"); ax.grid(True, alpha=0.3)
    h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="best")
    fig.tight_layout(); fig.savefig(OUTPUT_HETU_PLOT, dpi=150, bbox_inches="tight"); plt.close(fig)


def draw_metrics_panel(ax2d, h_tot_trace, e_pair_trace, step_idx):
    ax2d.clear()
    t_axis = np.arange(1, len(h_tot_trace)+1)
    ax2d.plot(t_axis, h_tot_trace, "b-", label=r"$H_{tot}$", linewidth=1.2)
    ax2d_twin = ax2d.twinx()
    ax2d_twin.plot(t_axis, e_pair_trace, "r--", label=r"$E_{pair,pre}$", linewidth=1.0, alpha=0.9)
    ax2d.set_xlabel("時間步 t"); ax2d.set_ylabel(r"$H_{tot}$", color="b")
    ax2d_twin.set_ylabel(r"$E_{pair,pre}$", color="r")
    ax2d.set_title("動態穩定性"); ax2d.grid(True, alpha=0.3); ax2d.set_xlim(0, TIME_STEPS+0.5)
    h1,l1=ax2d.get_legend_handles_labels(); h2,l2=ax2d_twin.get_legend_handles_labels()
    ax2d.legend(h1+h2, l1+l2, loc="upper right", fontsize=7)


def run_animation(path_coords, energies, velocities, star_indices, metrics_history):
    def _make_fig():
        fig = plt.figure(figsize=(12, 5.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], left=0.05, right=0.98)
        ax3d = fig.add_subplot(gs[0,0], projection="3d")
        ax2d = fig.add_subplot(gs[0,1])
        return fig, ax3d, ax2d

    def _on_frame(frame, ax3d, ax2d):
        energy = energies[frame]; velocity = velocities[frame]; star_idx = star_indices[frame]
        cvals = path_energy_colors(energy, path_coords)
        h_tot_trace = [metrics_history[i]["H_tot"] for i in range(frame+1)]
        e_pair_trace = [metrics_history[i]["E_pair_mse"] for i in range(frame+1)]
        x, y, z, u, v, w = build_vector_field(energy, velocity)
        ax3d.clear()
        ax3d.quiver(x,y,z,u,v,w, length=0.4, normalize=True, color="royalblue", alpha=0.85)
        path = np.array(path_coords, dtype=float)
        ax3d.plot(path[:,0],path[:,1],path[:,2], color="goldenrod", linewidth=1.3, alpha=0.5)
        ax3d.scatter(path[:,0],path[:,1],path[:,2], c=cvals, cmap="viridis", s=45, alpha=0.55)
        cx,cy,cz = path_coords[star_idx]
        ax3d.scatter([cx],[cy],[cz], color="crimson", s=200, zorder=5, depthshade=True, label="飛星")
        ax3d.set_title(f"3D HLBM | t={frame+1}/{TIME_STEPS}")
        ax3d.set_zticks(range(GRID_SIZE)); ax3d.set_zticklabels(("Earth","Human","Heaven"))
        ax3d.view_init(elev=24, azim=42)
        draw_metrics_panel(ax2d, h_tot_trace, e_pair_trace, frame)

    fig1, a3, a2 = _make_fig()
    anim1 = FuncAnimation(fig1, lambda f: _on_frame(f,a3,a2), frames=TIME_STEPS, interval=200, blit=False)
    try:
        anim1.save(OUTPUT_GIF, writer=PillowWriter(fps=5), dpi=100)
    finally:
        plt.close(fig1)
    save_hetu_stability_plot([m["H_tot"] for m in metrics_history], [m["E_pair_mse"] for m in metrics_history])


def main():
    palaces = build_twenty_seven_palaces()
    path_coords = [p.coord for p in palaces]
    print(f"河圖平衡常數 B={BALANCE_CONSTANT}，時間步長 = {TIME_STEPS}")
    print("\n========== 暴力增壓測試 ==========")
    brutal_ok, brutal_records = run_brutal_sweep(path_coords)
    save_brutal_sweep_plot(brutal_records)
    run_clinical_control_experiment(path_coords)
    energies, velocities, star_indices, metrics_history = run_time_loops(path_coords)
    if energies:
        save_last_frame_png(palaces, path_coords, energies[-1], velocities[-1], star_indices[-1])
        save_stress_layer_plot(metrics_history)
        run_animation(path_coords, energies, velocities, star_indices, metrics_history)
        print(f"完成。輸出: {OUTPUT_FILE}, {OUTPUT_GIF}, {OUTPUT_HETU_PLOT}, {OUTPUT_STRESS_LAYERS}, {OUTPUT_BRUTAL_SWEEP}, {OUTPUT_CLINICAL_COMPARE}")


if __name__ == "__main__":
    main()
