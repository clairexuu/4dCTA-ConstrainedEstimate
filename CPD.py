import os
import numpy as np
import trimesh
from pycpd import DeformableRegistration

def run_cpd_on_raw_phase0(ref_path, mesh_dir, output_dir, beta=1.0, lamb=3.0):
    os.makedirs(output_dir, exist_ok=True)

    # Load reference mesh from Phase 0
    ref_mesh = trimesh.load_mesh(ref_path)
    ref_vertices = ref_mesh.vertices
    ref_faces = ref_mesh.faces

    np.save(os.path.join(output_dir, "ref_vertices.npy"), ref_vertices)
    np.save(os.path.join(output_dir, "ref_faces.npy"), ref_faces)
    print(f"[✓] Loaded Phase 0 reference mesh: {ref_vertices.shape[0]} vertices, {ref_faces.shape[0]} faces")

    for phase in range(0, 100, 5):
        if phase == 0:
            continue

        mesh_path = os.path.join(mesh_dir, f"{phase}pct.stl")
        target_mesh = trimesh.load_mesh(mesh_path)

        # Match target points to ref_vertices using closest points
        target_points, _, _ = trimesh.proximity.closest_point(target_mesh, ref_vertices)

        ref_vertices = np.asarray(ref_vertices, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)

        print("target_points shape:", target_points.shape)
        print("ref_vertices shape:", ref_vertices.shape)

        # CPD Registration
        reg = DeformableRegistration(X=target_points, Y=ref_vertices, beta=beta, lamb=lamb, max_iterations=50)
        TY, _ = reg.register()
        disp = TY - ref_vertices

        # Save displacement and deformed mesh
        np.save(os.path.join(output_dir, f"disp_{phase}.npy"), disp)
        np.save(os.path.join(output_dir, f"regpts_{phase}.npy"), TY)

        deformed_mesh = trimesh.Trimesh(vertices=TY, faces=ref_faces)
        deformed_mesh.export(os.path.join(output_dir, f"cpd_{phase}pct.stl"))

        print(f"[✓] Phase {phase}: CPD completed and mesh saved.")

def main():
    ref_path = "4097359_17_vessel/CPDCEoutput/meshesCleaned/0pct.stl"
    mesh_dir = "4097359_17_vessel/CPDCEoutput/meshesCleaned"
    output_dir = "4097359_17_vessel/CPDCEoutput/cpd_output"
    run_cpd_on_raw_phase0(ref_path, mesh_dir, output_dir)

if __name__ == "__main__":
    main()