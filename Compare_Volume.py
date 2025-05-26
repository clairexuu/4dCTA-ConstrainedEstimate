from matplotlib.contour import ContourSet

import Constrained_Estimation
import measure_volumn
import os
import matplotlib.pyplot as plt
from glob import glob

nii_dir = "nnunet_outputs_pp2"
output_dir = "pipeline_outputs"
surface_meshes_dir = "pipeline_outputs/surface_meshes/"
cleaned_surface_meshes_dir = "pipeline_outputs/cleaned_surface_meshes/"
repair_meshes_dir = "pipeline_outputs/repaired_cleaned_meshes/"

# Step 1: Extract surface meshes from a directory of .nii.gz binary masks and save as .stl.
# Constrained_Estimation.extract_surface_meshes(nii_dir, surface_meshes_dir, level=0.5)

# Step 2: Clean up all stl files: remove small disconnected components by # faces
# for pct in range(0, 100, 5):
#     stl_path = os.path.join(surface_meshes_dir, f"{pct}pct.stl")
#     clean_path = Constrained_Estimation.clean_stl_keep_main_volume(stl_path, cleaned_surface_meshes_dir)

# stl_path = os.path.join(cleaned_surface_meshes_dir, "0pct.stl")
# Constrained_Estimation.load_and_repair_stl(stl_path, repair_meshes_dir)

# Repair by fill convex hull
# for pct in range(0, 100, 5):
#     stl_path = os.path.join(cleaned_surface_meshes_dir, f"{pct}pct_cleaned.stl")
#     clean_path = Constrained_Estimation.load_and_repair_stl(stl_path, repair_meshes_dir)

# Set reference phase, all remaining phases will be estimated
cleaned_stl = sorted(glob(os.path.join("pipeline_outputs/surface_meshes2", "*.stl")), key=measure_volumn.extract_pct)
reference_phase_dir = cleaned_stl[0]
target_phases_dir = cleaned_stl[1:]

# Compute volume change using mesh
phases, mesh_volumns = measure_volumn.compute_volume_change(nii_dir)

mesh_v0 = mesh_volumns[0]
mesh_delta_v = [v - mesh_v0 for v in mesh_volumns]
mesh_delta_pct = [(v - mesh_v0) / mesh_v0 * 100 for v in mesh_volumns]

# Constrained Estimation of Volume Change
successful_phases, estimated_volumes = Constrained_Estimation.constrained_estimate(reference_phase_dir, target_phases_dir, 100)

if not estimated_volumes:
    raise RuntimeError("❌ No estimated volumes were computed. Check for upstream errors.")

estimated_v0 = estimated_volumes[0]
estimated_delta_v = [v - estimated_v0 for v in estimated_volumes]
estimated_delta_pct = [(v - estimated_v0) / estimated_v0 * 100 for v in estimated_volumes]

plt.figure(figsize=(10, 5))
plt.plot(phases, mesh_delta_v, label="Mesh Volume Change (mm³)", marker='o')
plt.plot(successful_phases, estimated_delta_v, label="Constrained Estimated Volume Change (mm³)", marker='s')
plt.xticks(rotation=45)
plt.xlabel("Cardiac Phase")
plt.ylabel("Volume Change")
plt.title("Aneurysm Volume Change relative to 0pct Over Cardiac Cycle")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()