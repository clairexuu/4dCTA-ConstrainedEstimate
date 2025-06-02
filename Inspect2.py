import nibabel as nib
import numpy as np
import os
import open3d as o3d
import scipy.io as sio
import trimesh

def check_segmentation_values(nii_path):
    img = nib.load(nii_path)
    data = img.get_fdata()
    unique_vals = np.unique(data)
    print(f"Unique values in {nii_path}: {unique_vals}")

    if set(unique_vals) <= {0, 1}:
        print("✅ This is a binary segmentation.")
    else:
        print("⚠️ This is a multi-label or non-binary segmentation.")


def inspect_mesh_resolution(path):
    mesh = o3d.io.read_triangle_mesh(path)
    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)

    print(f"Mesh: {path}")
    print(f"  Vertices : {num_vertices}")
    print(f"  Triangles: {num_triangles}")
    print(f"  Resolution: {num_vertices / 1000:.1f}k vertices, {num_triangles / 1000:.1f}k triangles")


def convert_msh_to_mat(path, phase, outdir):
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    outpath = os.path.join(outdir, f"mesh_{phase}.mat")

    sio.savemat(outpath, {"vertices": vertices, "faces": faces})
    print(f"[✓] Saved: {outpath}")


def convert_npy_to_mat(npy_dir, out_dir):
    for fname in os.listdir(npy_dir):
        if fname.endswith('.npy'):
            path = os.path.join(npy_dir, fname)
            data = np.load(path)

            if fname.startswith("regpts_"):
                varname = "regpts"
            elif fname.startswith("disp_"):
                varname = "displacement"
            else:
                varname = "data"  # fallback

            matname = fname.replace(".npy", ".mat")
            save_path = os.path.join(out_dir, matname)

            sio.savemat(save_path, {varname: data})
            print(f"[✓] {fname} → {matname} ({varname})")


def check_regpts_vs_vertices(mesh_path, regpts_path, index=0):
    # Load mesh vertices
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)

    # Load CPD-registered points
    regpts = np.load(regpts_path)

    # Basic checks
    assert vertices.shape == regpts.shape, f"Shape mismatch: {vertices.shape} vs {regpts.shape}"

    # Compute per-point displacement
    displacement = np.linalg.norm(regpts - vertices, axis=1)

    # Print a few samples
    print(f"Loaded: {mesh_path} and {regpts_path}")
    print(f"    Shape: {vertices.shape}")
    print(f"    Displacement at index {index}: {displacement[index]:.4f} mm")
    print(f"    Max displacement: {np.max(displacement):.4f} mm")
    print(f"    Mean displacement: {np.mean(displacement):.4f} mm")

def check_waterright(file):
    mesh = o3d.io.read_triangle_mesh(file)
    is_watertight = mesh.is_edge_manifold() and mesh.is_vertex_manifold() and mesh.is_watertight()

    print(f"[INFO] Mesh is watertight: {is_watertight}")
    print("[✓] Edge-manifold:", mesh.is_edge_manifold())
    print("[✓] Vertex-manifold:", mesh.is_vertex_manifold())
    print("[✓] Watertight:", mesh.is_watertight())

def main():

    meshcleaned_dir = "meshesCleaned"  # your mesh folder
    meshes_dir = "meshes"
    for phase in range(0, 100, 5):
        meshcleaned_path = os.path.join(meshcleaned_dir, f"{phase}pct.stl")
        meshes_path =  os.path.join(meshes_dir, f"mesh_{phase}.stl")

        meshcleaned = trimesh.load_mesh(meshcleaned_path)
        print(f"[Phase {phase}] meshcleaned Vertices: {meshcleaned.vertices.shape[0]}")

        mesh = trimesh.load_mesh(meshes_path)
        print(f"[Phase {phase}] msh Vertices: {mesh.vertices.shape[0]}")
        print("\n")

if __name__ == "__main__":
    main()