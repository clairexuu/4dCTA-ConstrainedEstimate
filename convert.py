import open3d as o3d
import numpy as np
import os
import scipy.io as sio

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

# input_dir = "CPDCEoutput/ce500_output"
# output_dir = "matlab_visualizations/ce"
# convert_npy_to_mat(input_dir, output_dir)

for phase in range(0, 100, 5):
    mesh_path = f"CPDCEoutput/meshesCleaned/{phase}pct.stl"
    out_dir = "matlab_visualizations/meshes"
    convert_msh_to_mat(mesh_path, phase, out_dir)