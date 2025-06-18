import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import ConvexHull

def estimate_volume_from_mesh(path):
    mesh = trimesh.load_mesh(path)
    if not mesh.is_watertight:
        print(f"[!] Warning: mesh at {path} not watertight.")
    return abs(mesh.volume)

def estimate_volume_from_points(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return np.nan

# === Setup ===
phases = list(range(0, 100, 5))
mesh_dir = "5000CPD/meshesCleaned"           # STL files (ground truth)
pc_dir = "5000CPD/point_clouds"              # Fixed sampled points from phase 0
cpd_dir = "5000CPD/cpd_results"              # Displacement from CPD
est_dir = "5000CPD/ce_results"               # Displacement from CE

vol_stl, vol_cpd, vol_est, vol_pc= [], [], [], []

# === Load reference ===
ref_vertices = np.load(os.path.join(pc_dir, "points_0.npy"))
ref_faces = trimesh.load_mesh(os.path.join(mesh_dir, "0pct.stl")).faces

# === Loop over phases ===
for phase in phases:
    # -- STL volume --
    stl_path = os.path.join(mesh_dir, f"{phase}pct.stl")
    vol_stl.append(estimate_volume_from_mesh(stl_path))

    # -- PC volume --
    pc = np.load(os.path.join(pc_dir, f"points_{phase}.npy"))
    vol_pc.append(estimate_volume_from_points(pc))

    # -- CPD-deformed volume --
    try:
        disp = np.load(os.path.join(cpd_dir, f"disp_{phase}.npy"))
        deformed = ref_vertices + disp
        mesh = trimesh.Trimesh(vertices=deformed, faces=ref_faces)
        if not mesh.is_watertight:
            print(f"[!] Phase {phase} CPD mesh not watertight.")
        vol_cpd.append(abs(mesh.volume))
    except Exception as e:
        print(f"[!] CPD failed at phase {phase}: {e}")
        vol_cpd.append(np.nan)

    # -- CE-deformed volume --
    try:
        est_disp = np.load(os.path.join(est_dir, f"estimated_disp_{phase}.npy"))
        est_pc = ref_vertices + est_disp
        vol_est.append(estimate_volume_from_points(est_pc))
    except Exception as e:
        print(f"[!] CE failed at phase {phase}: {e}")
        vol_est.append(np.nan)

# === Convert to arrays
vol_stl = np.array(vol_stl)
vol_cpd = np.array(vol_cpd)
vol_est = np.array(vol_est)
vol_pc = np.array(vol_pc)

# === Compute % change from Phase 0
pct_stl = (vol_stl - vol_stl[0]) / vol_stl[0] * 100
pct_cpd = (vol_cpd - vol_cpd[0]) / vol_cpd[0] * 100
pct_est = (vol_est - vol_est[0]) / vol_est[0] * 100
pct_pc = (vol_pc - vol_pc[0]) / vol_pc[0] * 100

# === Plot: Absolute Volume
plt.figure(figsize=(10, 5))
plt.plot(phases, vol_stl, '-o', label="STL", linewidth=2)
plt.plot(phases, vol_cpd, '-s', label="CPD", linewidth=2)
plt.plot(phases, vol_est, '-^', label="Constrained Estimation", linewidth=2)
plt.plot(phases, vol_pc, '-*', label="PC", linewidth=2)
plt.xlabel("Cardiac Phase")
plt.ylabel("Volume (mm³)")
plt.title("Aneurysm Volume Over Cardiac Cycle")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Plot: Relative Change
plt.figure(figsize=(10, 5))
plt.plot(phases, pct_stl, '-o', label="STL", linewidth=2)
plt.plot(phases, pct_cpd, '-s', label="CPD", linewidth=2)
plt.plot(phases, pct_est, '-^', label="Constrained Estimation", linewidth=2)
plt.plot(phases, pct_pc, '-*', label="PC", linewidth=2)
plt.xlabel("Cardiac Phase")
plt.ylabel("Volume Change (%)")
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Relative Aneurysm Volume Change")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()