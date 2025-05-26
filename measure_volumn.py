import os
import nibabel as nib
import numpy as np
import trimesh
from skimage import measure
import matplotlib.pyplot as plt
from glob import glob
import re

# Helper to extract numeric phase value from filename
def extract_pct(filename):
    match = re.search(r'(\d+)pct', filename)
    return int(match.group(1)) if match else -1

def compute_volume_change(data_dir):
    nii_files = sorted(glob(os.path.join(data_dir, "*.nii.gz")), key=extract_pct)

    phases = []
    volumes = []

    for file_path in nii_files:
        phase_label = os.path.basename(file_path).split(".")[0]
        phases.append(phase_label)

        img = nib.load(file_path)
        data = img.get_fdata()
        voxel_spacing = img.header.get_zooms()

        binary_data = (data > 0.5).astype(np.uint8)

        verts, faces, _, _ = measure.marching_cubes(binary_data, level=0.5, spacing=voxel_spacing)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        volumes.append(mesh.volume)

    return phases, volumes

def plot_mesh_volume(phases, volumes):
    phases, volumes = compute_volume_change("nnunet_outputs_pp2")

    # volume change relative to 0pct
    v0 = volumes[0]
    delta_v = [v - v0 for v in volumes]
    delta_pct = [(v - v0) / v0 * 100 for v in volumes]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(phases, delta_v, label="Absolute Volume Change (mm³)", marker='o')
    plt.plot(phases, delta_pct, label="Percentage Volume Change (%)", marker='s')
    plt.xticks(rotation=45)
    plt.xlabel("Cardiac Phase")
    plt.ylabel("Volume Change")
    plt.title("Aneurysm Volume Change relative to 0pct Over Cardiac Cycle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()