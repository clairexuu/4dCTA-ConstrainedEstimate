import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

def estimate_volume_from_mesh(path):
    mesh = trimesh.load_mesh(path)
    if not mesh.is_watertight:
        print(f"[!] Warning: mesh at {path} not watertight.")
    return abs(mesh.volume)


phases = list(range(0, 100, 5))
mesh_dir = "CPDCEoutput/meshesCleaned"
cpd_dir = "CPDCEoutput/cpd_output"
ce_dir = "CPDCEoutput/ce500_output"


vol_stl, vol_cpd, vol_ce = [], [], []


ref_mesh = trimesh.load_mesh(os.path.join(mesh_dir, "0pct.stl"))
ref_vertices = ref_mesh.vertices
ref_faces = ref_mesh.faces
print(f"[✓] Loaded Phase 0 reference mesh: {ref_vertices.shape[0]} vertices, {ref_faces.shape[0]} faces")


for phase in phases:
    # STL volume
    stl_path = os.path.join(mesh_dir, f"{phase}pct.stl")
    vol_stl.append(estimate_volume_from_mesh(stl_path))

    # use stl volume at phase 0 for CPD and CE
    if phase == 0:
        vol_cpd.append(vol_stl[0])
        vol_ce.append(vol_stl[0])
        continue

    # CPD volume
    cpd_path = os.path.join(cpd_dir, f"cpd_{phase}pct.stl")
    vol_cpd.append(estimate_volume_from_mesh(cpd_path))

    # CE volume
    ce_path = os.path.join(ce_dir, f"ce_{phase}pct.stl")
    vol_ce.append(estimate_volume_from_mesh(ce_path))

vol_stl = np.array(vol_stl)
vol_cpd = np.array(vol_cpd)
vol_ce = np.array(vol_ce)

pct_stl = (vol_stl - vol_stl[0]) / vol_stl[0] * 100
pct_cpd = (vol_cpd - vol_stl[0]) / vol_stl[0] * 100
pct_ce = (vol_ce - vol_stl[0]) / vol_stl[0] * 100

# Absolute Volume
plt.figure(figsize=(10, 5))
plt.plot(phases, vol_stl, '-o', label="STL", linewidth=2)
plt.plot(phases, vol_cpd, '-s', label="CPD", linewidth=2)
plt.plot(phases, vol_ce, '-^', label="CE", linewidth=2)
plt.xlabel("Cardiac Phase")
plt.ylabel("Volume (mm³)")
plt.title("Absolute Aneurysm Volume Change")
plt.grid(True)
plt.legend()
plt.show()

# Relative Volume Change in %
plt.figure(figsize=(10, 5))
plt.plot(phases, pct_stl, '-o', label="STL", linewidth=2)
plt.plot(phases, pct_cpd, '-s', label="CPD", linewidth=2)
plt.plot(phases, pct_ce, '-^', label="CE", linewidth=2)
plt.xlabel("Cardiac Phase")
plt.ylabel("Volume Change (%)")
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Relative Aneurysm Volume Change")
plt.grid(True)
plt.legend()
plt.show()