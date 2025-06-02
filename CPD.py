import nibabel as nib
import numpy as np
from skimage import measure
import open3d as o3d
import os
from pycpd import DeformableRegistration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# build mesh -----------------------------------------
def nii_to_mesh(nii_path, output_path, level=0.5, smoothing_iterations=5):
    img = nib.load(nii_path)
    data = img.get_fdata()
    affine = img.affine

    # marching cubes
    verts, faces, normals, _ = measure.marching_cubes(data, level=level)

    # coordinates
    verts_h = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
    verts_world = (affine @ verts_h.T).T[:, :3]

    # Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Laplacian smoothing
    # mesh = mesh.filter_smooth_simple(number_of_iterations=smoothing_iterations)

    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"[✓] Saved mesh: {output_path}")


def sample_and_save_pointclouds(input_dir, output_dir, n_points=1500):
    os.makedirs(output_dir, exist_ok=True)
    for phase in range(0, 100, 5):
        mesh_path = os.path.join(input_dir, f"{phase}pct.stl")
        mesh = trimesh.load_mesh(mesh_path)
        pc = mesh.sample(n_points)
        np.save(os.path.join(output_dir, f"points_{phase}.npy"), pc)
        print(f"[✓] Phase {phase}: Saved {n_points} points")




# CPD --------------------------------------------

# number of vertices in each mesh range from 1690 to 1760
# thus need to downsample to the same size
def load_uniform_mesh_points(path, n_points=1500):
    mesh = trimesh.load_mesh(path)
    if not mesh.is_empty and mesh.vertices.shape[0] > 0:
        points, _ = trimesh.sample.sample_surface_even(mesh, count=n_points)
        return np.asarray(points, dtype=np.float64)  # <--- force plain ndarray
    else:
        raise ValueError(f"[!] Invalid or empty mesh at: {path}")


def run_cpd_registration(source, target, beta=2.0, lamb=3.0, max_iters=50):
    reg = DeformableRegistration(X=target, Y=source, beta=beta, lamb=lamb, max_iterations=max_iters)
    TY, _ = reg.register()
    return TY

# --- Displacement ---
def compute_displacement(reference_points, deformed_points):
    return deformed_points - reference_points

# --- Main CPD Workflow ---
def cpd(ref_mesh_path, input_dir, output_dir, n_points=1500):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Sample reference mesh
    ref_points = load_uniform_mesh_points(ref_mesh_path, n_points=n_points)

    cpd_results = {}

    # Step 2: Register all other phases to Phase 0
    for i in range(0, 20):
        phase = i * 5
        target_path = os.path.join(input_dir, f"mesh_{phase:02d}.stl")

        target_points = load_uniform_mesh_points(target_path, n_points=n_points)
        TY = run_cpd_registration(source=ref_points, target=target_points)
        disp = compute_displacement(ref_points, TY)

        cpd_results[phase] = {
            "registered_points": TY,
            "displacement": disp
        }

        print(f"[✓] Registered Phase {phase} → Phase 0")

    # Step 3: Save results
    for phase, result in cpd_results.items():
        np.save(os.path.join(output_dir, f"regpts_{phase}.npy"), result["registered_points"])
        np.save(os.path.join(output_dir, f"disp_{phase}.npy"), result["displacement"])

# Check -----------------------------------------------------------

def load_mesh_points(path):
    mesh = trimesh.load_mesh(path)
    return mesh.vertices

def main():
    input_dir = "meshes"
    output_dir = "cpd_results"

    # for pct in range(0, 100, 5):
    #     nii_path = f"nnunet_outputs_pp2/{pct}pct.nii.gz"
    #     output_path = f"mesh/mesh_{pct}.stl"
    #     nii_to_mesh(nii_path, output_path)

    # ref_points = load_mesh_points("meshes/mesh_0.stl")
    # target_points = load_mesh_points("meshes/mesh_5.stl")
    #
    # print(ref_points.shape, target_points.shape)  # ← must be identical!

    sample_and_save_pointclouds("2ndRound/meshesCleaned", "2ndRound/pointclouds", 1500)


if __name__ == "__main__":
    main()