"""
run_demo.py — One-command reproduction of all Genesis 3D-HLBM results.

Usage:
    python run_demo.py

Outputs:
    genesis_3d_results.png   (4-panel figure)
    Theorem verification printed to stdout
"""
from genesis_3d_hlbm import verify_theorems, plot_results

if __name__ == "__main__":
    print("=" * 50)
    print("  Genesis 3D-HLBM  —  Theorem Verification")
    print("=" * 50)
    verify_theorems(amplitude=500.0, T=200, seed=0)

    print("\nGenerating 4-panel figure...")
    plot_results(save=True)
    print("\nAll done.  Open genesis_3d_results.png to view results.")
