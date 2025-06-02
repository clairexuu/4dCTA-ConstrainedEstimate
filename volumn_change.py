import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import ConvexHull

def estimate_volume(point_cloud):
    hull = ConvexHull(point_cloud)
    return hull.volume

# === Setup ===
phases = list(range(0, 100, 5))
pc_dir = "2ndRound/pointclouds"
cpd_dir = "2ndRound/cpd_results"
est_dir = "2ndRound/ce_results"

vol_stl, vol_cpd, vol_est = [], [], []

# === Load reference mesh ===
vertices_ref = np.load(os.path.join(pc_dir, "points_0.npy"))
print("Vertices in ref mesh:", vertices_ref.shape)

# === Loop over all phases ===
for phase in phases:
    # -- STL point cloud volume --
    pc = np.load(os.path.join(pc_dir, f"points_{phase}.npy"))
    vol_stl.append(estimate_volume(pc))

    # -- CPD-deformed volume --
    try:
        disp = np.load(os.path.join(cpd_dir, f"disp_{phase}.npy"))
        cpd_pc = vertices_ref + disp
        vol_cpd.append(estimate_volume(cpd_pc))
    except:
        vol_cpd.append(np.nan)

    # -- CE-deformed volume --
    try:
        est_disp = np.load(os.path.join(est_dir, f"estimated_disp_{phase}.npy"))
        est_pc = vertices_ref + est_disp
        vol_est.append(estimate_volume(est_pc))
    except:
        vol_est.append(np.nan)

# === Convert to NumPy arrays ===
vol_stl = np.array(vol_stl)
vol_cpd = np.array(vol_cpd)
vol_est = np.array(vol_est)

# === Compute % change from Phase 0 ===
v0 = vol_stl[0]
pct_stl = (vol_stl - v0) / v0 * 100
pct_cpd = (vol_cpd - v0) / v0 * 100
pct_est = (vol_est - v0) / v0 * 100

# === Plot 1: Absolute Volume ===
plt.figure(figsize=(10, 5))
plt.plot(phases, vol_stl, '-o', label="STL", linewidth=2)
plt.plot(phases, vol_cpd, '-s', label="CPD", linewidth=2)
plt.plot(phases, vol_est, '-^', label="Constrained Estimation", linewidth=2)
plt.xlabel("Cardiac Phase")
plt.ylabel("Volume (mm³)")
plt.title("Aneurysm Volume Over Cardiac Cycle")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig("volume_absolute_trimesh.png")
plt.show()

# === Plot 2: % Volume Change ===
plt.figure(figsize=(10, 5))
plt.plot(phases, pct_stl, '-o', label="STL", linewidth=2)
plt.plot(phases, pct_cpd, '-s', label="CPD", linewidth=2)
plt.plot(phases, pct_est, '-^', label="Constrained Estimation", linewidth=2)
plt.xlabel("Cardiac Phase")
plt.ylabel("Volume Change (%)")
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Relative Aneurysm Volume Change")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig("volume_percentage_trimesh.png")
plt.show()