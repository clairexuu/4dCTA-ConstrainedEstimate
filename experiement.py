import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from constrained_estimate import *

sparse_counts = [50, 100, 200, 300, 400, 500]
phases = range(0, 100, 5)

disp_dir = "2ndRound/cpd_results"
pc_dir = "2ndRound/pointclouds"

# Assume all point clouds have the same shape
ref_points = np.load(os.path.join(pc_dir, "points_0.npy"))
N = ref_points.shape[0]

# Run tests
results = defaultdict(list)

for n in sparse_counts:
    print(f"\n=== Testing with {n} sparse points ===")
    all_mse, all_rmse, all_mae = [], [], []

    for phase in tqdm(phases, desc=f"Sparse={n}"):
        try:
            # Load point cloud and displacement
            pc_path = os.path.join(pc_dir, f"points_{phase}.npy")
            disp_path = os.path.join(disp_dir, f"disp_{phase}.npy")

            if not os.path.exists(pc_path) or not os.path.exists(disp_path):
                print(f"[Phase {phase}] Missing data")
                continue

            points = np.load(pc_path)
            disp = np.load(disp_path)

            if points.shape != disp.shape:
                print(f"[Phase {phase}] Shape mismatch: points {points.shape}, disp {disp.shape}")
                continue

            # Build KNN stiffness matrix for this point cloud
            K = build_knn_stiffness_matrix(points, k=10, E=1.4e6)

            # Sample sparse points
            H, y, _ = select_sparse_samples(disp, n=n)

            # Regularize stiffness matrix
            from scipy.sparse import identity
            K_reg = K + 1e-5 * identity(K.shape[0])

            # Run estimation
            U_est = constrained_estimation_solver(K_reg, H, y, max_iter=10)

            # Compare to ground truth
            est_disp = U_est.reshape(N, 3)
            mse = np.mean((est_disp - disp) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(est_disp - disp))

            all_mse.append(mse)
            all_rmse.append(rmse)
            all_mae.append(mae)

        except Exception as e:
            print(f"[Phase {phase}] ERROR: {e}")

    # Store average metrics
    results["sparse"].append(n)
    results["mse"].append(np.mean(all_mse))
    results["rmse"].append(np.mean(all_rmse))
    results["mae"].append(np.mean(all_mae))

# === Summary ===
print("\n=== Summary ===")
print(f"{'Points':>6} | {'MSE':>10} | {'RMSE':>10} | {'MAE':>10}")
print("-" * 45)
for i in range(len(sparse_counts)):
    print(f"{results['sparse'][i]:6} | {results['mse'][i]:10.6f} | {results['rmse'][i]:10.6f} | {results['mae'][i]:10.6f}")