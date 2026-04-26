# Genesis 3D-HLBM

**A 27-Palace Symmetry-Constrained Discrete Lattice Model with Bounded Energy Dynamics**

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
python genesis_3d_hlbm.py
```

Expected outputs:
- `genesis_3d_result.png` — 3D lattice final frame
- `brute_sweep_stability.png` — Brute-force sweep (amplitude 88→1000)
- `hetu_stability.png` — HeTu dynamic stability curve
- `stress_test_layers.png` — Layer energy (Earth/Human/Heaven) under pulse
- `clinical_lbm_hlbm_sigma.png` — Standard LBM vs HLBM comparison
- `genesis_3d_time.gif` — Animated simulation

---

## Model Summary

A 3×3×3 lattice (27 nodes = 27 palaces) inspired by the I-Ching cube and HeTu (河圖) pairing principle.

| Component | Description |
|-----------|-------------|
| Grid | 3×3×3 = 27 nodes, 3 layers: Earth / Human / Heaven |
| Flying-star path | 27-palace spiral, alternating 順/逆 every 27 steps |
| HeTu constraint | Symmetric pairs through center sum to B=10 |
| Center node | Fixed at E_c = 5 |
| Stress test | Large pulse at Earth corner; tracks propagation upward |
| Brute sweep | Amplitudes 88→1000; verifies bounded σ_E under extreme forcing |

---

## Key Results

- Total energy bounded under all tested amplitudes (88–1000)
- Variance σ_E remains finite (no blow-up)
- HLBM shows stronger stability coefficient σ vs Standard LBM after pulse
- Heaven layer energy self-regulates (autoregulation-like behavior)

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
