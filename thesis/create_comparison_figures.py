#!/usr/bin/env python3
"""
Create comparison figures with experimental data for thesis sections 4.1 and 4.2
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless environment
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path("../hirataken20251122-2/data")
OUTPUT_DIR = Path("figures")

def load_exp_sq():
    """Load experimental S(Q) data"""
    data = np.loadtxt(DATA_DIR / "sq_real_data.csv", delimiter=',')
    return data[:, 0], data[:, 1]

def load_exp_gr():
    """Load experimental g(r) data"""
    data = np.loadtxt(DATA_DIR / "g_exp_150C_cleaned.dat", comments='#')
    return data[:, 0], data[:, 1]

def load_sim_rdf(filepath):
    """Load simulation RDF from LAMMPS output"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                float(parts[0])
                data_lines.append(line)
            except ValueError:
                continue

    if not data_lines:
        return None, None

    data = np.loadtxt(data_lines, ndmin=2)
    return data[:, 1], data[:, 2]  # r, g(r)

def calc_sq_from_gr(r, g, rho=0.0522):
    """Calculate S(Q) from g(r) using Fourier transform"""
    Q = np.linspace(0.5, 15.0, 500)
    S = np.ones_like(Q)
    dr = r[1] - r[0] if len(r) > 1 else 0.1

    for i, q in enumerate(Q):
        if q < 1e-6:
            continue
        integrand = (g - 1.0) * r * np.sin(q * r)
        S[i] = 1.0 + 4.0 * np.pi * rho * np.sum(integrand) * dr / q

    return Q, S

def create_rdf_comparison():
    """Create g(r) comparison figure with experimental data"""
    # Load experimental g(r)
    r_exp, g_exp = load_exp_gr()

    # Load simulation g(r) at 423K (150C)
    sim_rdf_path = Path("../hirataken20251122-2/outputs/rdf_423K.dat")
    if not sim_rdf_path.exists():
        # Try alternative path
        sim_rdf_path = Path("../hirataken20251122-2/outputs/fine_search_shoulder/out_sig12_100_ga1_100.rdf")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot experimental data
    ax.scatter(r_exp, g_exp, color='black', s=15, alpha=0.7, label='Experiment (150°C)', zorder=5)

    # If we have simulation data, plot it
    if sim_rdf_path.exists():
        r_sim, g_sim = load_sim_rdf(sim_rdf_path)
        if r_sim is not None:
            ax.plot(r_sim, g_sim, 'b-', lw=2, label='LJ Simulation')

    ax.set_xlabel('r (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_title('Radial Distribution Function: Simulation vs Experiment', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rdf_comparison_with_exp.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'rdf_comparison_with_exp.png'}")

def create_sq_comparison():
    """Create S(Q) comparison figure with experimental data"""
    # Load experimental S(Q)
    q_exp, s_exp = load_exp_sq()

    # Load simulation g(r) and calculate S(Q)
    sim_rdf_path = Path("../hirataken20251122-2/outputs/rdf_423K.dat")

    # Use pure LJ simulation RDF (sigma=100%, epsilon=100% = standard LJ)
    possible_paths = [
        Path("../hirataken20251122-2/outputs/grid_search/out_s100_e100.rdf"),
    ]

    sim_rdf_path = None
    for p in possible_paths:
        if p.exists():
            sim_rdf_path = p
            break

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot experimental data
    ax.scatter(q_exp, s_exp, color='black', s=15, alpha=0.7, label='Experiment (150°C)', zorder=5)

    # Highlight shoulder region
    ax.axvspan(2.8, 3.5, color='orange', alpha=0.15, label='Shoulder region')

    # If we have simulation data, calculate and plot S(Q)
    if sim_rdf_path and sim_rdf_path.exists():
        r_sim, g_sim = load_sim_rdf(sim_rdf_path)
        if r_sim is not None:
            q_sim, s_sim = calc_sq_from_gr(r_sim, g_sim)
            ax.plot(q_sim, s_sim, 'b-', lw=2, label='LJ Simulation')

    ax.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax.set_ylabel('S(Q)', fontsize=12)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.legend(fontsize=10)
    ax.set_title('Structure Factor: Single-component LJ vs Experiment', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add annotation for shoulder
    ax.annotate('Shoulder\n(not reproduced)', xy=(3.1, 1.8), xytext=(4.5, 2.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sq_comparison_with_exp.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'sq_comparison_with_exp.png'}")

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Creating comparison figures...")
    create_rdf_comparison()
    create_sq_comparison()
    print("Done!")
