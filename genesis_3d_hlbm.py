"""
Genesis 3D-HLBM: Symmetry-Constrained Discrete Lattice Model
Author: Yao-Kai Kao (高堯楷)
Institute of Brain Science, National Yang Ming Chiao Tung University
Date: April 2026

Reference:
    Kao, Y.-K. (2026). Genesis 3D-HLBM: A Symmetry-Constrained Discrete
    Lattice Model with Bounded Energy Dynamics.
    DOI: https://doi.org/10.5281/zenodo.19784150
"""

import numpy as np
import matplotlib.pyplot as plt

GRID  = 3
N     = GRID**3   # 27 nodes
B     = 1.0       # pairing constant: E_p + E_p' = B
ALPHA = B / 2     # center node (fixed)

def get_pair(p, c=(1,1,1)):
    return tuple(2*c[i] - p[i] for i in range(3))

def all_pairs():
    nodes = [(x,y,z) for x in range(3) for y in range(3) for z in range(3)]
    center = (1,1,1)
    visited, pairs = set(), []
    for p in nodes:
        if p == center: continue
        pp  = get_pair(p)
        key = tuple(sorted([p, pp]))
        if key not in visited:
            visited.add(key)
            pairs.append((p, pp))
    return pairs

PAIRS = all_pairs()   # 13 unique pairs

def idx(p):
    return p[0]*9 + p[1]*3 + p[2]

# ── Operators ──────────────────────────────────

def diffusion_step(E, D=0.1, dt=0.1):
    E3  = E.reshape(3,3,3)
    out = E3.copy()
    for x in range(3):
        for y in range(3):
            for z in range(3):
                nb = []
                for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx,ny,nz = x+dx,y+dy,z+dz
                    if 0<=nx<3 and 0<=ny<3 and 0<=nz<3:
                        nb.append(E3[nx,ny,nz])
                out[x,y,z] = E3[x,y,z] + D*dt*(sum(nb)/len(nb) - E3[x,y,z])
    return out.flatten()

def forcing_step(E, amplitude, rng):
    noise = rng.uniform(-amplitude, amplitude, size=N)
    return np.clip(E + noise*0.01, 0, B)

def projection_step(E):
    """Enforce E_p + E_p' = B for all 13 pairs. Center fixed at ALPHA."""
    E2 = E.copy()
    E2[idx((1,1,1))] = ALPHA
    for p, pp in PAIRS:
        i, j = idx(p), idx(pp)
        corr = 0.5 * (B - (E2[i] + E2[j]))
        E2[i] += corr
        E2[j] += corr
    return E2

def step(E, amplitude, rng):
    return projection_step(forcing_step(diffusion_step(E), amplitude, rng))

def init_E():
    E = np.full(N, B/2)
    E[idx((1,1,1))] = ALPHA
    return E

# ── Simulation ─────────────────────────────────

def run(amplitude=1.0, T=100, seed=42):
    rng = np.random.default_rng(seed)
    E   = init_E()
    rec = dict(total=[], var=[], maxgrad=[])
    for _ in range(T):
        E = step(E, amplitude, rng)
        rec["total"].append(E.sum())
        rec["var"].append(E.var())
        rec["maxgrad"].append(float(np.max(np.abs(np.diff(E)))))
    return rec

def run_pair(amplitude=100.0, T=100, seed=42):
    """Constrained vs unconstrained side-by-side."""
    rng_c = np.random.default_rng(seed)
    rng_u = np.random.default_rng(seed)
    Ec, Eu = init_E(), init_E()
    hc = dict(var=[], maxgrad=[])
    hu = dict(var=[], maxgrad=[])
    for _ in range(T):
        Ec = step(Ec, amplitude, rng_c)
        hc["var"].append(Ec.var())
        hc["maxgrad"].append(float(np.max(np.abs(np.diff(Ec)))))
        # unconstrained: skip projection
        Eu = forcing_step(diffusion_step(Eu), amplitude, rng_u)
        hu["var"].append(Eu.var())
        hu["maxgrad"].append(float(np.max(np.abs(np.diff(Eu)))))
    return hc, hu

# ── Theorem verification ────────────────────────

def verify_theorems(amplitude=500.0, T=200, seed=0):
    rng = np.random.default_rng(seed)
    E   = init_E()
    expected = 13*B + ALPHA
    tol = 1e-9
    ok  = [True, True, True]
    for _ in range(T):
        E = step(E, amplitude, rng)
        if abs(E.sum() - expected) > tol:      ok[0] = False
        if E.min() < -tol or E.max() > B+tol: ok[1] = False
        if E.std() > B/2 + tol:               ok[2] = False
    print("="*45)
    print(f"Theorem 1  Energy Invariance : {'PASS' if ok[0] else 'FAIL'}")
    print(f"Theorem 2  Boundedness       : {'PASS' if ok[1] else 'FAIL'}")
    print(f"Theorem 3  Variance Bound    : {'PASS' if ok[2] else 'FAIL'}")
    print("="*45)

# ── Plotting ───────────────────────────────────

def plot_results(save=True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Genesis 3D-HLBM — Key Results  (Kao, 2026)", fontsize=13, fontweight='bold')

    # A: Energy invariance
    ax = axes[0,0]
    h  = run(amplitude=100.0, T=100)
    ax.plot(h["total"], color='steelblue', lw=2, label='Σ E(t)')
    ax.axhline(13*B+ALPHA, color='red', ls='--', lw=1.5, label=f'13B+α = {13*B+ALPHA:.2f}')
    ax.set_title("Theorem 1: Total Energy Invariance")
    ax.set_xlabel("t"); ax.set_ylabel("Σ E"); ax.legend(); ax.grid(alpha=.3)

    # B: Variance bound
    ax = axes[0,1]
    ax.plot(h["var"], color='green', lw=2, label='σ²(t)')
    ax.axhline((B/2)**2, color='red', ls='--', lw=1.5, label=f'(B/2)²={( B/2)**2:.3f}')
    ax.set_title("Theorem 3: Variance Bounded by (B/2)²")
    ax.set_xlabel("t"); ax.set_ylabel("σ²"); ax.legend(); ax.grid(alpha=.3)

    # C: Constrained vs Unconstrained
    ax  = axes[1,0]
    hc, hu = run_pair(amplitude=100.0, T=100)
    ax.plot(hc["var"], color='steelblue', lw=2, label='Constrained')
    ax.plot(hu["var"], color='orange',    lw=2, ls='--', label='Unconstrained')
    ax.set_title("Constrained vs Unconstrained: Variance")
    ax.set_xlabel("t"); ax.set_ylabel("σ²"); ax.legend(); ax.grid(alpha=.3)

    # D: Brute-force sweep
    ax   = axes[1,1]
    amps = [10, 50, 100, 200, 500, 1000]
    fv   = [run(amplitude=a, T=100)["var"][-1] for a in amps]
    ax.plot(amps, fv, 'o-', color='purple', lw=2, label='Final σ²')
    ax.axhline((B/2)**2, color='red', ls='--', lw=1.5, label='Bound')
    ax.set_title("Brute-Force Sweep: Final σ² vs Amplitude")
    ax.set_xlabel("Forcing amplitude"); ax.set_ylabel("Final σ²"); ax.legend(); ax.grid(alpha=.3)

    plt.tight_layout()
    fname = "genesis_3d_results.png"
    if save:
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
    else:
        plt.show()
    plt.close()

# ── Entry point ────────────────────────────────

if __name__ == "__main__":
    print("Genesis 3D-HLBM — reproducing key results...\n")
    verify_theorems()
    print("\nGenerating figures...")
    plot_results(save=True)
    print("\nDone.  See genesis_3d_results.png")
