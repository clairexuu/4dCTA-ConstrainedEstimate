import nibabel as nib
import numpy as np
from skimage import measure
import os
import matplotlib.pyplot as plt
import trimesh
from scipy.ndimage import label, binary_closing, binary_fill_holes


# Process a single NIfTI (.nii.gz) file to extract and save a cleaned STL mesh.
def extract_surface_mesh(nii_path, output_stl_path, threshold = 0.5, closing_iterations = 2, preview = True):
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()

        # threshold
        binary = (data > threshold).astype(np.uint8)

        # filter largest connected component
        labeled, num = label(binary)
        if num == 0:
            print(f"No components found in {os.path.basename(nii_path)}")
            return False
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

        print(f"Saved mesh to {output_stl_path}")
        print(f"   Watertight: {mesh.is_watertight}, Euler: {mesh.euler_number}, Volume: {mesh.volume:.2f}, Faces: {len(mesh.faces)}, Vertices: {len(mesh.vertices)}")
        return True

    except Exception as e:
        print(f"Failed to process {nii_path}: {e}")
        return False

for phase in range(0, 100, 5):
    nii_path = f"4007775_aneurysm/nnunet_outputs_pp/{phase}pct.nii.gz"
    output_stl_path = f"4007775_aneurysm/CPDCEoutput/meshesCleaned2/{phase}pct.stl"
    extract_surface_mesh(nii_path, output_stl_path, threshold=0.5, closing_iterations=2, preview=False)