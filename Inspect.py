import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
from scipy.ndimage import label, binary_closing, binary_fill_holes
import trimesh
import matplotlib.pyplot as plt
import os

# for pct in range(0, 100, 5):
#     img = nib.load(f"nnunet_outputs_pp2/{pct}pct.nii.gz")
#     data = img.get_fdata()
#
#     # plt.imshow(data[:, :, data.shape[2] // 2], cmap='gray')
#     # plt.title("Middle Slice")
#     # plt.show()
#
#     from scipy.ndimage import label
#
#     binary_data = (data > 0.5).astype(np.uint8)
#     labeled, num = label(binary_data)
#     print(f"Connected components: {num}")

def extract_surface_mesh(
    nii_path: str,
    output_stl_path: str,
    threshold: float = 0.5,
    closing_iterations: int = 2,
    verbose: bool = True,
    preview: bool = True,
) -> bool:
    """
    Process a single NIfTI (.nii.gz) file to extract and save a cleaned STL mesh.
    """

    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()

        # Step 1: threshold
        binary = (data > threshold).astype(np.uint8)

        # Step 2: filter largest connected component
        labeled, num = label(binary)
        if num == 0:
            if verbose:
                print(f"❌ No components found in {os.path.basename(nii_path)}")
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

        # Step 4: surface extraction
        verts, faces, _, _ = measure.marching_cubes(blob, level=0.5, spacing=spacing)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Step 5: export
        os.makedirs(os.path.dirname(output_stl_path), exist_ok=True)
        mesh.export(output_stl_path)

        if verbose:
            print(f"✅ Saved mesh to {output_stl_path}")
            print(f"   Watertight: {mesh.is_watertight}, Euler: {mesh.euler_number}, Volume: {mesh.volume:.2f}, Faces: {len(mesh.faces)}")

        return True
    except Exception as e:
        if verbose:
            print(f"❌ Failed to process {nii_path}: {e}")
        return False

for pct in range(0, 100, 5):
    nii_path = f"nnunet_outputs_pp2/{pct}pct.nii.gz"
    output_stl_path = f"meshesCleaned/{pct}pct.stl"
    extract_surface_mesh(nii_path, output_stl_path)

# for pct in range(0, 100, 5):
#     nii_path = f"nnunet_outputs_pp2/{pct}pct.nii.gz"
#     output_stl_path = f"pipeline_outputs/surface_meshes2/{pct}pct.stl"
#
#     img = nib.load(nii_path)
#     data = img.get_fdata()
#
#     plt.imshow(data[:, :, data.shape[2] // 2], cmap='gray')
#     plt.title("Middle Slice")
#     plt.show()
#
#     extract_surface_mesh(nii_path, output_stl_path)

# extract_surface_mesh("nnunet_outputs_pp2/65pct_filled.nii.gz", "pipeline_outputs/surface_meshes2/65pct_filled.stl")

def fix_non_genus0_mesh(mesh: trimesh.Trimesh, verbose=True) -> trimesh.Trimesh:
    """
    Attempts to repair a watertight mesh that is not genus-0 by filling holes and simplifying.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        verbose (bool): Whether to print status messages.

    Returns:
        trimesh.Trimesh: A repaired mesh that is (hopefully) genus-0 and watertight.
    """
    initial_genus = (2 - mesh.euler_number) // 2
    if verbose:
        print(f"🔧 Initial genus: {initial_genus}, Watertight: {mesh.is_watertight}, Faces: {len(mesh.faces)}")

    # Split into components and keep the largest watertight one
    components = mesh.split(only_watertight=True)
    if len(components) > 1:
        mesh = sorted(components, key=lambda m: len(m.faces), reverse=True)[0]
        if verbose:
            print(f"🔍 Found multiple watertight components, using largest with {len(mesh.faces)} faces")

    # Try hole-filling and re-check genus
    mesh.fill_holes()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())

    final_genus = (2 - mesh.euler_number) // 2
    if verbose:
        print(f"✅ Final genus: {final_genus}, Watertight: {mesh.is_watertight}, Volume: {mesh.volume:.2f} mm³")

    return mesh

# mesh = trimesh.load("pipeline_outputs/surface_meshes2/65pct.stl")
# mesh.show()
# fixed_mesh = fix_non_genus0_mesh(mesh)
# fixed_mesh.show()
# fixed_mesh.export("pipeline_outputs/surface_meshes2/65pct_fixed.stl")


# import nibabel as nib
# import numpy as np
# from scipy.ndimage import binary_closing, binary_fill_holes
#
# nii_path = "nnunet_outputs_pp2/65pct.nii.gz"
# img = nib.load(nii_path)
# data = img.get_fdata()
#
# # Step 1: Threshold to get binary mask
# binary = (data > 0.5)
#
# # Step 2: Apply morphological closing (to smooth narrow connections)
# closed = binary_closing(binary, iterations=3)
#
# # Step 3: Fill internal holes in 3D
# filled = binary_fill_holes(closed, structure=np.ones((3, 3, 3)))
#
# import matplotlib.pyplot as plt
#
# mid = filled.shape[2] // 2
# plt.imshow(filled[:, :, mid], cmap='gray')
# plt.title("Middle Slice After Repair")
# plt.show()
#
# fixed_img = nib.Nifti1Image(filled.astype(np.uint8), affine=img.affine, header=img.header)
# nib.save(fixed_img, "nnunet_outputs_pp2/65pct_filled.nii.gz")