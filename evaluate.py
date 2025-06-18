import numpy as np
import os
import csv

def compute_error_metrics(true_disp, est_disp):
    diff = true_disp - est_disp
    mse = np.mean(np.sum(diff ** 2, axis=1))
    mae = np.mean(np.linalg.norm(diff, axis=1))
    rmse = np.sqrt(mse)
    return mse, mae, rmse

def evaluate_constrained_estimation_vs_cpd(
    cpd_dir,
    ce_dir,
    n_sparse,
    ref_disp_name_func,
    est_disp_name_func
):
    total_mse, total_mae, total_rmse = 0.0, 0.0, 0.0
    valid_phases = 0

    for phase in range(5, 100, 5):
        cpd_path = os.path.join(cpd_dir, ref_disp_name_func(phase))
        ce_path = os.path.join(ce_dir, est_disp_name_func(phase, n_sparse))

        if not os.path.exists(cpd_path) or not os.path.exists(ce_path):
            print(f"[!] Missing file at phase {phase}, n={n_sparse}")
            continue

        cpd_disp = np.load(cpd_path)
        ce_disp = np.load(ce_path)

        if cpd_disp.shape != ce_disp.shape:
            print(f"[!] Shape mismatch at phase {phase}, n={n_sparse}")
            continue

        mse, mae, rmse = compute_error_metrics(cpd_disp, ce_disp)
        total_mse += mse
        total_mae += mae
        total_rmse += rmse
        valid_phases += 1

    if valid_phases > 0:
        avg_mse = total_mse / valid_phases
        avg_mae = total_mae / valid_phases
        avg_rmse = total_rmse / valid_phases
        return (n_sparse, avg_mse, avg_mae, avg_rmse)
    else:
        return (n_sparse, np.nan, np.nan, np.nan)

def main():
    cpd_dir = "CPDCEoutput/cpd_output"
    sparse_points_list = [10, 50, 100, 200, 300, 400, 500, 600]

    results = [("N_sprs", "Mean Square Error", "Mean Absolute Error", "Root Mean Square Deviation")]

    def ref_disp_name(phase): return f"disp_{phase}.npy"
    def est_disp_name(phase, n): return f"estimated_disp_{phase}.npy"

    for n_sparse in sparse_points_list:
        ce_dir = f"CPDCEoutput/ce{n_sparse}_output"
        result = evaluate_constrained_estimation_vs_cpd(
            cpd_dir,
            ce_dir,
            n_sparse,
            ref_disp_name_func=ref_disp_name,
            est_disp_name_func=est_disp_name
        )
        results.append(result)

    # === Save to CSV ===
    with open("CPDCEoutput/ce_vs_cpd_error_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)
    print("[✓] Saved metrics to ce_vs_cpd_error_metrics.csv")

if __name__ == "__main__":
    main()