import argparse
import os
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix, identity
from scipy.sparse.linalg import inv
import trimesh

def build_surface_stiffness_matrix(vertices, faces, E=1.4e6):
    N = vertices.shape[0]
    I, J, V = [], [], []

    def cotangent(a, b, c):
        ba = b - a
        ca = c - a
        cosine = np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return 1.0 / np.tan(angle + 1e-6)

    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        cij = cotangent(vk, vi, vj)
        cjk = cotangent(vi, vj, vk)
        cki = cotangent(vj, vk, vi)
        for (a, b, w) in [(i, j, cij), (j, k, cjk), (k, i, cki)]:
            for d in range(3):
                I.extend([3*a+d, 3*a+d, 3*b+d, 3*b+d])
                J.extend([3*a+d, 3*b+d, 3*a+d, 3*b+d])
                V.extend([w, -w, -w, w])

    K = coo_matrix((V, (I, J)), shape=(3*N, 3*N)).tocsc()
    return E * K

def select_sparse_samples(displacement, n):
    N = displacement.shape[0]
    indices = np.random.choice(N, size=n, replace=False)
    H = lil_matrix((3*n, 3*N))
    y = np.zeros((3*n,))
    for i, idx in enumerate(indices):
        for d in range(3):
            H[3*i + d, 3*idx + d] = 1.0
            y[3*i + d] = displacement[idx, d]
    return H.tocsc(), y, indices

def constrained_estimation_solver(K, H, y, max_iter=10, tol=1e-6):
    print("[INFO] Solving K⁻¹...")
    epsilon = 1e-5
    K_reg = K + epsilon * identity(K.shape[0])
    K_inv = inv(K_reg)

    print("[INFO] Precomputing L matrix...")
    HKH_T = H @ K_inv @ H.T
    HKH_T_inv = np.linalg.inv(HKH_T.toarray())
    L = K_inv @ H.T @ HKH_T_inv

    U = np.zeros(K.shape[0])
    print("[INFO] Starting iterations...")
    for k in range(max_iter):
        r = y - H @ U
        delta = L @ r
        U_new = U + delta
        if np.linalg.norm(delta) < tol:
            print(f"[✓] Converged at iteration {k}")
            break
        U = U_new
    return U

def constrained_estimate(ref_mesh_path, disp_dir, output_dir, sparse_n, phases=range(5, 100, 5)):
    os.makedirs(output_dir, exist_ok=True)

    mesh = trimesh.load_mesh(ref_mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces
    N = vertices.shape[0]

    print(f"[STEP] Building stiffness matrix from reference mesh...")
    K = build_surface_stiffness_matrix(vertices, faces)

    for phase in phases:
        disp_path = os.path.join(disp_dir, f"disp_{phase}.npy")
        if not os.path.exists(disp_path):
            print(f"[!] Missing CPD displacement for phase {phase}")
            continue

        disp = np.load(disp_path)
        if disp.shape != (N, 3):
            print(f"[!] Shape mismatch for phase {phase}")
            continue

        print(f"\n[Phase {phase}] Selecting {sparse_n} sparse points...")
        H, y, indices = select_sparse_samples(disp, n=sparse_n)

        print(f"[Phase {phase}] Running constrained estimation...")
        U_est = constrained_estimation_solver(K, H, y)
        est_disp = U_est.reshape(N, 3)

        # Save estimated displacement
        est_disp_path = os.path.join(output_dir, f"estimated_disp_{phase}.npy")
        np.save(est_disp_path, est_disp)
        print(f"[✓] Saved: {est_disp_path}")

        # Save reconstructed mesh
        deformed_vertices = vertices + est_disp
        deformed_mesh = trimesh.Trimesh(vertices=deformed_vertices, faces=faces)
        out_mesh_path = os.path.join(output_dir, f"ce_{phase}pct.stl")
        deformed_mesh.export(out_mesh_path)
        print(f"[✓] Saved CE mesh: {out_mesh_path}")

def main():
    # for n_sparse in [10, 50, 100, 200, 300, 400, 500, 600]:
    #     constrained_estimate(
    #         ref_mesh_path="CPDCEoutput/meshesCleaned/0pct.stl",
    #         disp_dir="CPDCEoutput/cpd_output",
    #         output_dir=f"CPDCEoutput/ce{n_sparse}_output",
    #         sparse_n = n_sparse
    #     )

    constrained_estimate(
        ref_mesh_path="4097359_17_vessel/CPDCEoutput/meshesCleaned/0pct.stl",
        disp_dir="4097359_17_vessel/CPDCEoutput/cpd_output",
        output_dir=f"4097359_17_vessel/CPDCEoutput/ce500_output",
        sparse_n=500
    )

if __name__ == "__main__":
    main()