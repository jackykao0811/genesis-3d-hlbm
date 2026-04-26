# Genesis 3D-HLBM

**A Symmetry-Constrained Discrete Lattice Model with Bounded Energy Dynamics**

Author: Yao-Kai Kao (高堯楷)  
Institute of Brain Science, National Yang Ming Chiao Tung University  
Date: April 2026  
DOI: [10.5281/zenodo.19784150](https://doi.org/10.5281/zenodo.19784150)

---

## Quick Reproduction (One Command)

```bash
git clone https://github.com/jackykao0811/genesis-3d-hlbm.git
cd genesis-3d-hlbm
pip install -r requirements.txt
python run_demo.py
```

Expected output:
```
==================================================
  Genesis 3D-HLBM  —  Theorem Verification
==================================================
=============================================
Theorem 1  Energy Invariance : PASS
Theorem 2  Boundedness       : PASS
Theorem 3  Variance Bound    : PASS
=============================================

Generating 4-panel figure...
Saved: genesis_3d_results.png
```

---

## Model Summary

A finite-dimensional discrete dynamical system on a **3×3×3 lattice** (27 nodes) with:

| Component | Description |
|-----------|-------------|
| State space | Energy field `E: G → ℝ`, `G = {0,1,2}³` |
| Center node | `E_c = α` (fixed) |
| Pairing | `p' = 2c − p` (reflection through center) |
| Constraint | `E_p + E_p' = B` for all 13 pairs |
| Projection | `E_p ← E_p + ½(B − (E_p + E_p'))` |
| Update | Diffusion → Forcing → Projection |

---

## Theoretical Results

| Theorem | Statement | Status |
|---------|-----------|--------|
| **Theorem 1** | Total energy invariant: `Σ Eᵢ = 13B + α` | Proved + verified |
| **Theorem 2** | If `Eᵢ ≥ 0`, then `0 ≤ Eᵢ ≤ B` | Proved + verified |
| **Theorem 3** | Variance bounded: `σ_E ≤ B/2` | Proved + verified |

---

## File Structure

```
genesis-3d-hlbm/
├── genesis_3d_hlbm.py        # Core model (operators, simulation, plotting)
├── run_demo.py               # One-command reproduction script
├── requirements.txt          # numpy, matplotlib
├── Kao_2026_Genesis_3D_HLMB_v1.pdf   # Paper (v1)
└── *.png                     # Result figures
```

---

## Citation

```bibtex
@misc{kao2026hlbm,
  author = {Kao, Yao-Kai},
  title  = {Genesis 3D-HLBM: A Symmetry-Constrained Discrete Lattice Model
             with Bounded Energy Dynamics},
  year   = {2026},
  doi    = {10.5281/zenodo.19784150},
  url    = {https://doi.org/10.5281/zenodo.19784150}
}
```
