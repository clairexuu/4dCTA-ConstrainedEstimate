import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import inv
import os
from scipy.sparse import identity
from sklearn.neighbors import NearestNeighbors

def build_knn_stiffness_matrix(points, k=10, E=1.4e6):
    """
    Build a point-cloud-based stiffness matrix using a KNN graph Laplacian.

    Args:
        points (Nx3): point cloud
        k (int): number of nearest neighbors
        E (float): Young's modulus scaling

    Returns:
        K (3N x 3N sparse matrix): global stiffness matrix
    """
    N = points.shape[0]
    knn = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = knn.kneighbors(points)

    I, J, V = [], [], []

    for i in range(N):
        for j in indices[i]:
            if i == j:
                continue
            for d in range(3):
                I.extend([3 * i + d, 3 * i + d, 3 * j + d, 3 * j + d])
                J.extend([3 * i + d, 3 * j + d, 3 * i + d, 3 * j + d])
                V.extend([1, -1, -1, 1])

    K = coo_matrix((V, (I, J)), shape=(3 * N, 3 * N)).tocsc()
    return E * K

def build_surface_stiffness_matrix(vertices, faces, E=1.4e6, nu=0.47):
    """
    Build a simplified surface stiffness matrix using cotangent Laplacian.

    Args:
        vertices (Nx3): vertex positions
        faces (Mx3): triangle vertex indices
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Returns:
        K (3N x 3N sparse matrix): global stiffness matrix
    """
    N = vertices.shape[0]
    I = []
    J = []
    V = []

    def cotangent(a, b, c):
        # angle at vertex a in triangle (a, b, c)
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
                I.extend([3 * a + d, 3 * a + d, 3 * b + d, 3 * b + d])
                J.extend([3 * a + d, 3 * b + d, 3 * a + d, 3 * b + d])
                V.extend([w, -w, -w, w])

    K = coo_matrix((V, (I, J)), shape=(3 * N, 3 * N)).tocsc()

    # Scale stiffness matrix by E (material strength)
    return E * K



def select_sparse_samples(displacement, n=600):
    """
    Randomly sample n vertices and build observation vector and H matrix.

    Args:
        displacement (Nx3): full CPD displacement vectors
        n (int): number of sparse observations

    Returns:
        H (3n x 3N sparse matrix): sampling matrix
        y (3n x 1 array): observed sparse displacement
        indices (n array): vertex indices sampled
    """
    N = displacement.shape[0]
    indices = np.random.choice(N, size=n, replace=False)

    H = lil_matrix((3 * n, 3 * N))
    y = np.zeros((3 * n,))

    for i, idx in enumerate(indices):
        for d in range(3):
            H[3 * i + d, 3 * idx + d] = 1.0
            y[3 * i + d] = displacement[idx, d]

    return H.tocsc(), y, indices


def constrained_estimation_solver(K, H, y, max_iter=10, tol=1e-6):
    """
    Iterative constrained estimation solver for displacement field.

    Args:
        K: (3N x 3N sparse matrix) stiffness matrix
        H: (3n x 3N sparse matrix) sampling matrix
        y: (3n,) vector of sparse displacements
        max_iter: number of iterations
        tol: convergence threshold

    Returns:
        U: (3N,) estimated full displacement field
    """
    print("[INFO] Solving K⁻¹...")
    epsilon = 1e-5
    K_reg = K + epsilon * identity(K.shape[0])
    K_inv = inv(K_reg)

    print("[INFO] Precomputing L matrix...")
    HKH_T = H @ K_inv @ H.T
    HKH_T_inv = np.linalg.inv(HKH_T.toarray())
    L = K_inv @ H.T @ HKH_T_inv

    # Initial guess: zero displacement
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

def main():
    # Setup
    pointcloud_dir = "5000CPD/point_clouds"
    disp_dir = "5000CPD/cpd_results"
    output_dir = "5000CPD/ce_results"
    os.makedirs(output_dir, exist_ok=True)

    n_points = 5000  # Must match CPD input
    N = n_points

    # Load Phase 0 point cloud
    points_0 = np.load(os.path.join(pointcloud_dir, "points_0.npy"))

    print("[STEP] Building stiffness matrix from KNN graph...")
    K = build_knn_stiffness_matrix(points_0, k=10, E=1.4e6)

    # Loop over phases
    for i in range(20):
        phase = i * 5
        disp_path = os.path.join(disp_dir, f"disp_{phase}.npy")

        if not os.path.exists(disp_path):
            print(f"[!] Missing: {disp_path}")
            continue

        disp = np.load(disp_path)
        if disp.shape != (N, 3):
            print(f"[!] Shape mismatch in {disp_path}, expected {(N,3)}, got {disp.shape}")
            continue

        print(f"\n[Phase {phase}] Sampling points...")
        H, y, indices = select_sparse_samples(disp, n=600)

        print(f"[Phase {phase}] Running estimation...")
        U_est = constrained_estimation_solver(K, H, y)

        U_est_xyz = U_est.reshape(N, 3)
        out_path = os.path.join(output_dir, f"estimated_disp_{phase}.npy")
        np.save(out_path, U_est_xyz)
        print(f"[✓] Saved: {out_path}")

    # Compare with CPD ----------------------------------------------------

    # cpd_dir = "2ndRound/cpd_results"
    # est_dir = "2ndRound/ce_results"
    # phases = range(0, 100, 5)
    #
    # print(f"{'Phase':>6} | {'MSE':>10} | {'RMSE':>10} | {'MAE':>10}")
    # print("-" * 45)
    #
    # for phase in phases:
    #     try:
    #         cpd_disp = np.load(os.path.join(cpd_dir, f"disp_{phase}.npy"))  # (N, 3)
    #         est_disp = np.load(os.path.join(est_dir, f"estimated_disp_{phase}.npy"))  # (N, 3)
    #
    #         if cpd_disp.shape != est_disp.shape:
    #             raise ValueError("Shape mismatch")
    #
    #         diff = cpd_disp - est_disp
    #         mse = np.mean(diff ** 2)
    #         rmse = np.sqrt(mse)
    #         mae = np.mean(np.abs(diff))
    #
    #         print(f"{phase:6} | {mse:10.6f} | {rmse:10.6f} | {mae:10.6f}")
    #
    #     except Exception as e:
    #         print(f"{phase:6} | ERROR: {str(e)}")


if __name__ == "__main__":
    main()