import numpy as np
import matplotlib.pyplot as plt

python_pe_atom_method_1 = np.loadtxt("energies.txt",skiprows=1)
python_pe_atom_method_1 = python_pe_atom_method_1[:,1]
lammps_pe_atom = np.loadtxt("pe_atom.trj",skiprows=9)
lammps_pe_atom = lammps_pe_atom[:,1]

# Plot and save figure
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
fig, ax = plt.subplots(figsize=(3.25, 3.25))

ax.scatter(python_pe_atom_method_1, lammps_pe_atom,
           s=8, alpha=0.6, edgecolor="none",
           label="Per-particle energies")

min_val = min(python_pe_atom_method_1.min(), lammps_pe_atom.min())
max_val = max(python_pe_atom_method_1.max(), lammps_pe_atom.max())
ax.plot([min_val, max_val], [min_val, max_val],
        ls="--", lw=1, color="0.2",
        label="y = x")

ax.set_xlabel("Per-particle energy (method 1)", fontsize=8)
ax.set_ylabel("Per-particle energy (LAMMPS)", fontsize=8)
ax.set_aspect('equal', adjustable='box')

ax.text(0.97, 0.50, "(b)",
        transform=ax.transAxes,
        fontsize=9,
        va="center", ha="right")

# Error 
total_energy_method1 = np.sum(python_pe_atom_method_1)
total_energy_lammps  = np.sum(lammps_pe_atom)

error_total_abs = abs(total_energy_method1 - total_energy_lammps)
err_per_atom    = python_pe_atom_method_1 - lammps_pe_atom
err_rms         = np.sqrt(np.mean(err_per_atom**2))
textstr = (
    f"Total energy (method 1):   {total_energy_method1:.15g}\n"
    f"Total energy (LAMMPS): {total_energy_lammps:.15g}\n"
    f"Abs. ΔE (total): {error_total_abs:.5e}\n"
    f"RMS  |Δe_i|: {err_rms:.5e}"
)

ax.text(0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=7,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

ax.legend(fontsize=8, loc="lower right", frameon=False)
fig.savefig("comparison_lammps.png", dpi=1600, bbox_inches="tight")
plt.close(fig)

