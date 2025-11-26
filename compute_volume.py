import nibabel as nib
import numpy as np
from skimage import measure
import os
import matplotlib.pyplot as plt
import trimesh
import pandas as pd
from scipy.ndimage import label, binary_closing, binary_fill_holes

# Process a single NIfTI (.nii.gz) file to extract and save a cleaned STL mesh.
def extract_surface_mesh(nii_path, output_stl_path, threshold=0.5, closing_iterations=2, preview=True):
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()  # keep full zooms for compatibility
        shape_str = f"{data.shape[0]}*{data.shape[1]}*{data.shape[2]}"  # format shape like 35*35*10

        # threshold
        binary = (data > threshold).astype(np.uint8)

        # filter largest connected component
        labeled, num = label(binary)
        if num == 0:
            print(f"No components found in {os.path.basename(nii_path)}")
            return None  # return None to indicate failure
        sizes = np.bincount(labeled.ravel())
        largest_label = sizes[1:].argmax() + 1
        blob = (labeled == largest_label).astype(np.uint8)

        # Step 3: clean the mask
        blob = binary_closing(blob, iterations=closing_iterations)
        blob = binary_fill_holes(blob)

        if preview:
            mid = blob.shape[2] // 2
            plt.imshow(blob[:, :, mid], cmap="gray")
            plt.title(f"Middle slice: {os.path.basename(nii_path)}")
            plt.show()

        # surface extraction
        verts, faces, _, _ = measure.marching_cubes(blob, level=0.5, spacing=spacing)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # export
        os.makedirs(os.path.dirname(output_stl_path), exist_ok=True)
        mesh.export(output_stl_path)

        # Collect mesh info
        info = {
            "Watertight": mesh.is_watertight,
            "Euler": mesh.euler_number,
            "Volume_mm3": abs(mesh.volume),
            "Faces": len(mesh.faces),
            "Vertices": len(mesh.vertices),
            "Dimensions": shape_str  # new column
        }

        print(f"Saved mesh to {output_stl_path}")
        return info

    except Exception as e:
        print(f"Failed to process {nii_path}: {e}")
        return None

def estimate_volume_from_mesh(path):
    mesh = trimesh.load_mesh(path)
    if not mesh.is_watertight:
        print(f"[!] Warning: mesh at {path} not watertight.")
    return abs(mesh.volume)

if __name__ == "__main__":
    # Phases: 0%, 5%, ..., 95%
    phases = list(range(0, 100, 5))

    # Input/output paths
    nii_folder = "0596010_aneurysm/nnunet_outputs_pp"
    output_dir = "volume"
    mesh_dir = "volume/0596010_aneurysm"
    os.makedirs(mesh_dir, exist_ok=True)

    # Prepare CSV data
    csv_rows = []

    # Step 1: Generate STL meshes and record info
    for phase in phases:
        nii_path = os.path.join(nii_folder, f"{phase}pct.nii.gz")
        output_stl_path = os.path.join(mesh_dir, f"{phase}pct.stl")
        mesh_info = extract_surface_mesh(nii_path, output_stl_path, threshold=0.5, closing_iterations=2, preview=False)

        if mesh_info:
            csv_rows.append({
                "Phase": phase,
                **mesh_info  # includes Watertight, Euler, Volume_mm3, Faces, Vertices, Dimensions
            })
        else:
            # In case of failure, append NaNs
            csv_rows.append({
                "Phase": phase,
                "Watertight": None,
                "Euler": None,
                "Volume_mm3": None,
                "Faces": None,
                "Vertices": None,
                "Dimensions": None
            })

    # Convert to DataFrame and save
    case_name = os.path.basename(os.path.dirname(os.path.normpath(nii_folder)))
    output_csv = os.path.join(output_dir, f"{case_name}_vol.csv")
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved detailed volume data to {output_csv}")

    # Step 2: Plot volume curve
    # Only plot if we have valid volume data
    if df["Volume_mm3"].notnull().any():
        plt.figure(figsize=(10, 5))
        plt.plot(df["Phase"], df["Volume_mm3"], '-o', label="STL", linewidth=2)
        plt.xlabel("Cardiac Phase (%)")
        plt.ylabel("Volume (mm³)")
        plt.title(f"{case_name} - volume change")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_plot = os.path.join(output_dir, f"{case_name}_vol.png")
        plt.savefig(output_plot)
        print(f"Saved volume plot to {output_plot}")
        plt.show()
    else:
        print("No valid volume data to plot.")