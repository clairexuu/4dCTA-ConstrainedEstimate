import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from pycpd import DeformableRegistration
import csv

# === SETTINGS ===
phases = list(range(0, 100, 5))
mesh_dir = "CPDCEoutput/meshesCleaned"
output_dir = "CPDCEoutput"

# === Load reference mesh ===
ref_mesh = trimesh.load_mesh(os.path.join(mesh_dir, "0pct.stl"))
ref_vertices = ref_mesh.vertices
ref_faces = ref_mesh.faces

def estimate_volume_from_mesh(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return abs(mesh.volume)

def run_cpd(Y, X, beta, lamb):
    reg = DeformableRegistration(X=X, Y=Y, beta=beta, lamb=lamb, max_iterations=50)
    TY, _ = reg.register()
    return TY

# === Ground truth volumes ===
vol_stl = []
for phase in phases:
    mesh = trimesh.load_mesh(os.path.join(mesh_dir, f"{phase}pct.stl"))
    vol_stl.append(abs(mesh.volume))
vol_stl = np.array(vol_stl)

# === Sweep parameters ===
beta_values = [1.0, 2.0, 3.0]
lambda_values = [1.0, 3.0, 5.0]

results = {}

for beta in beta_values:
    for lamb in lambda_values:
        label = f"β={beta}, λ={lamb}"
        print(f"[RUNNING] {label}")
        volumes = []

        for phase in phases:
            if phase == 0:
                volumes.append(abs(ref_mesh.volume))
                continue

            tgt_mesh = trimesh.load_mesh(os.path.join(mesh_dir, f"{phase}pct.stl"))
            target_vertices = tgt_mesh.vertices

            ref_vertices = np.asarray(ref_vertices, dtype=np.float64)
            target_vertices = np.asarray(target_vertices, dtype=np.float64)

            TY = run_cpd(Y=ref_vertices, X=target_vertices, beta=beta, lamb=lamb)
            vol = estimate_volume_from_mesh(vertices=TY, faces=ref_faces)
            volumes.append(vol)

        results[label] = np.array(volumes)

# === Compute error metrics and save to CSV ===
def compute_metrics(gt, pred):
    diff = pred - gt
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)
    return mse, mae, rmse

metrics_csv = os.path.join(output_dir, "cpd_volume_error_metrics.csv")
with open(metrics_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Beta", "Lambda", "MSE", "MAE", "RMSE"])

    for label, volumes in results.items():
        beta_val = float(label.split("β=")[1].split(",")[0])
        lamb_val = float(label.split("λ=")[1])
        mse, mae, rmse = compute_metrics(vol_stl, volumes)
        writer.writerow([beta_val, lamb_val, mse, mae, rmse])
        print(f"[✓] {label}: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

print(f"\n[✓] Saved volume error metrics to {metrics_csv}")

# === Plot volume comparisons ===
plt.figure(figsize=(10, 6))
plt.plot(phases, vol_stl, 'k-o', label='STL (Ground Truth)', linewidth=3)

for label, volumes in results.items():
    plt.plot(phases, volumes, '--', label=label)

plt.xlabel("Cardiac Phase")
plt.ylabel("Volume (mm³)")
plt.title("Volume Comparison for Different CPD Parameters")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()