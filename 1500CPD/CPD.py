import nibabel as nib
import numpy as np
from skimage import measure
import os
from pycpd import DeformableRegistration
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
        print(f"   Watertight: {mesh.is_watertight}, Euler: {mesh.euler_number}, Volume: {mesh.volume:.2f}, Faces: {len(mesh.faces)}")
        return True

    except Exception as e:
        print(f"Failed to process {nii_path}: {e}")
        return False


# Load a mesh, sample n_points uniformly from its surface, and save the result.
# number of vertices in each mesh need to be the same
def load_uniform_mesh_points(path, n_points, save_path):
    mesh = trimesh.load_mesh(path)
    if not mesh.is_empty and mesh.vertices.shape[0] > 0:
        points, _ = trimesh.sample.sample_surface_even(mesh, count=n_points)
        points = np.asarray(points, dtype=np.float64)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, points)
            print(f"[✓] Saved point cloud: {save_path}")

        return points
    else:
        raise ValueError(f"[!] Invalid or empty mesh at: {path}")


def run_cpd_registration(source, target, beta=2.0, lamb=3.0, max_iters=50):
    reg = DeformableRegistration(X=target, Y=source, beta=beta, lamb=lamb, max_iterations=max_iters)
    TY, _ = reg.register()
    return TY

# Displacement
def compute_displacement(reference_points, deformed_points):
    return deformed_points - reference_points

# CPD
def cpd(ref_mesh_path, input_dir, cpd_output_dir, pc_output_dir, n_points):
    os.makedirs(cpd_output_dir, exist_ok=True)
    os.makedirs(pc_output_dir, exist_ok=True)

    # Sample reference mesh
    ref_save_path = os.path.join(pc_output_dir, "points_0.npy")
    ref_points = load_uniform_mesh_points(ref_mesh_path, n_points, ref_save_path)

    cpd_results = {}

    # Register all other phases to Phase 0
    for i in range(0, 20):
        phase = i * 5
        target_path = os.path.join(input_dir, f"{phase}pct.stl")
        pc_path = os.path.join(pc_output_dir, f"points_{phase}.npy")

        target_points = load_uniform_mesh_points(target_path, n_points, pc_path)
        TY = run_cpd_registration(source=ref_points, target=target_points)
        disp = compute_displacement(ref_points, TY)

        cpd_results[phase] = {
            "registered_points": TY,
            "displacement": disp
        }

        print(f"[✓] Registered Phase {phase} → Phase 0")

    # Save
    for phase, result in cpd_results.items():
        np.save(os.path.join(cpd_output_dir, f"regpts_{phase}.npy"), result["registered_points"])
        np.save(os.path.join(cpd_output_dir, f"disp_{phase}.npy"), result["displacement"])


def load_mesh_points(path):
    mesh = trimesh.load_mesh(path)
    return mesh.vertices, mesh.faces

def main():
    # # Build mesh from nii.gz
    # for pct in range(0, 100, 5):
    #     nii_path = f"nnunet_outputs_pp2/{pct}pct.nii.gz"
    #     output_stl_path = f"meshesCleaned/{pct}pct.stl"
    #     extract_surface_mesh(nii_path, output_stl_path)

    input_dir = "../remeshCPD/meshesCleaned"
    for phase in range(0, 100, 5):
        vertices, faces = load_mesh_points(os.path.join(input_dir, f"{phase}pct.stl"))
        print(f"{phase}: {vertices.shape} vertices and {faces.shape} faces")

    input_dir = "5000CPD/meshesCleaned"
    cpd_output_dir = "5000CPD/cpd_results"
    pc_output_dir = "5000CPD/point_clouds"
    ref_mesh_path = "5000CPD/meshesCleaned/0pct.stl"

    # cpd(ref_mesh_path, input_dir, cpd_output_dir, pc_output_dir, n_points=5000)


if __name__ == "__main__":
    main()