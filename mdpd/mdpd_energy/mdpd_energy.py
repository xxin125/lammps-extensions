import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters
N   = 500
box = np.array([5.0, 5.0, 5.0])
A   = -50.0
B   = 25.0
r_c = 1.0
r_d = 0.75

# Precompute constants for psi
C_c = math.pi * r_c**4 / 30.0 * A
C_d = math.pi * r_d**4 / 30.0 * B

# Generate random positions in a 5x5x5 box
np.random.seed(64613164)
positions = np.random.rand(N, 3) * box
positions = np.round(positions, decimals=6)

# Initialize local densities
bar_rho   = np.zeros(N)
tilde_rho = np.zeros(N)

# Define weight functions
def w_bar(r):
    return (15.0 / (2.0 * math.pi * r_c**3) * (1.0 - r / r_c)**2) if r < r_c else 0.0

def w_tilde(r):
    return (15.0 / (2.0 * math.pi * r_d**3) * (1.0 - r / r_d)**2) if r < r_d else 0.0

# Minimum-image convention for periodic boundaries
def min_image_distance(i, j):
    dx = positions[j] - positions[i]
    dx -= box * np.round(dx / box)
    return math.sqrt(np.dot(dx, dx))

# Compute local densities
for i in range(N):
    for j in range(N):
        if i == j: continue
        r = min_image_distance(i, j)
        bar_rho[i]   += w_bar(r)
        tilde_rho[i] += w_tilde(r)

# Method 1: per-particle free energy psi_i
psi = C_c * bar_rho + C_d * tilde_rho**2
total_energy_method1 = psi.sum()

# Method 2: per-particle energy from pairwise potentials
per_atom = np.zeros(N)
total_energy_method2 = 0.0

def tilde_omega_c(r):
    return max(0.0, 1.0 - r / r_c)

def tilde_omega_d(r):
    return max(0.0, 1.0 - r / r_d)

for i in range(N):
    for j in range(i + 1, N):
        r = min_image_distance(i, j)
        # Conservative term split half-half
        u_c = A * r_c / 2.0 * tilde_omega_c(r)**2
        # Density dependent term split proportional
        factor = B * r_d / 4.0 * tilde_omega_d(r)**2
        u_d_i  = factor * tilde_rho[i]
        u_d_j  = factor * tilde_rho[j]
        u_ij   = u_c + u_d_i + u_d_j
        
        total_energy_method2 += (u_c + u_d_i + u_d_j)
        per_atom[i] += (u_c / 2.0 + u_d_i)
        per_atom[j] += (u_c / 2.0 + u_d_j)

# Check totals
print(f"Total energy (method 1): {total_energy_method1:.15g}")
print(f"Total energy (method 2): {total_energy_method2:.15g}")

# Plot and save figure
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
fig, ax = plt.subplots(figsize=(3.25, 3.25))

ax.scatter(psi, per_atom,
           s=8, alpha=0.6, edgecolor="none",
           label="Per-particle energies")

min_val = min(psi.min(), per_atom.min())
max_val = max(psi.max(), per_atom.max())
ax.plot([min_val, max_val], [min_val, max_val],
        ls="--", lw=1, color="0.2",
        label="y = x")

ax.set_xlabel("Per-particle energy (method 1)", fontsize=8)
ax.set_ylabel("Per-particle energy (method 2)", fontsize=8)
ax.set_aspect('equal', adjustable='box')

ax.text(0.97, 0.50, "(a)",
        transform=ax.transAxes,
        fontsize=9,
        va="center", ha="right")

# Error 
error_total_abs = abs(total_energy_method1 - total_energy_method2)
err_per_atom    = per_atom - psi
err_rms         = np.sqrt(np.mean(err_per_atom**2))
textstr = (
    f"Total energy (method 1): {total_energy_method1:.15g}\n"
    f"Total energy (method 2): {total_energy_method2:.15g}\n"
    f"Abs. ΔE (total): {error_total_abs:.5e}\n"
    f"RMS  |Δe_i|: {err_rms:.5e}"
)

ax.text(0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=7,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
ax.legend(fontsize=8, loc="lower right", frameon=False)
fig.savefig("comparison.png", dpi=1600, bbox_inches="tight")
plt.close(fig)

# Save per-particle energies to file
data = np.vstack((np.arange(1, N+1), psi, per_atom)).T
np.savetxt('energies.txt', data, 
           header='id method1_energy method2_energy', fmt='%d %.15g %.15g')

# Write LAMMPS data file
with open('lammps.data', 'w') as f:
    f.write("LAMMPS data file via script\n\n")
    f.write(f"{N} atoms\n1 atom types\n\n")
    f.write(f"0.0 {box[0]} xlo xhi\n")
    f.write(f"0.0 {box[1]} ylo yhi\n")
    f.write(f"0.0 {box[2]} zlo zhi\n\n")
    f.write("Atoms\n\n")
    for idx, pos in enumerate(positions, start=1):
        f.write(f"{idx} 1 {pos[0]} {pos[1]} {pos[2]}\n")

print("Files written: comparison.png, energies.txt, lammps.data")
