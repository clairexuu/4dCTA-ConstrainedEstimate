# 4D-CTA Constrained Estimation
**Constrained Estimation of Intracranial Aneurysm Surface Deformation using 4D-CTA**

[![Paper](https://img.shields.io/badge/Paper-PubMed-blue)](https://pubmed.ncbi.nlm.nih.gov/38128464/)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/)

---

## Project Motivation

Intracranial aneurysms are life-threatening vascular abnormalities, and their irregular surface pulsation patterns may be strongly associated with rupture risk. Accurately estimating aneurysm surface deformation from medical imaging data is crucial for clinical diagnosis and risk assessment.

This project implements the algorithm described in **[Xie et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38128464/)** published in *Computer Methods and Programs in Biomedicine*. The method transforms surface deformation estimation into a **constrained optimization problem**, minimizing the error between model-estimated displacement and sparse data points extracted from 4D-CT angiography (4D-CTA) imaging.

---

## Technical Highlights

### Core Algorithm
The method solves a constrained optimization problem to estimate dense surface deformation from sparse 4D-CTA measurements:

```
min ||K^(-1) * f||² + λ||H*f - y||²
```

**Where:**
- `K`: Surface stiffness matrix (cotangent Laplacian-based)
- `f`: Unknown force field driving deformation
- `H`: Sparse sampling operator
- `y`: Observed displacement at sparse points
- `λ`: Regularization weight balancing smoothness and data fidelity

### Key Components

1. **CPD-based Point Tracking** ([CPD.py](CPD.py))
   - Uses Coherent Point Drift (CPD) algorithm for deformable registration
   - Extracts sparse displacement measurements from multi-phase 4D-CTA meshes
   - Handles topology preservation across cardiac phases

2. **Constrained Estimation Solver** ([Constrained_Estimate.py](Constrained_Estimate.py))
   - Builds surface stiffness matrix using cotangent weights
   - Solves sparse linear system with regularization
   - Reconstructs dense displacement field from sparse samples

3. **Volumetric Analysis** ([Volume_Change.py](Volume_Change.py))
   - Computes aneurysm volume changes across cardiac cycle
   - Validates deformation estimation accuracy

4. **Mesh Processing Pipeline** ([toMesh.py](toMesh.py), [convert.py](convert.py))
   - DICOM to NIfTI conversion
   - 3D mesh extraction from segmented 4D-CTA volumes
   - Mesh cleaning and preprocessing

---

## Project Structure

```
4dCTA-ConstrainedEstimate/
├── README.md
├── .gitignore
│
├── CPD.py                      # Coherent Point Drift registration
├── Constrained_Estimate.py     # Core constrained optimization solver
├── Volume_Change.py            # Volumetric analysis tools
│
├── toMesh.py                   # DICOM → mesh extraction
├── convert.py                  # Format conversion utilities
├── fix_dicom.py                # DICOM preprocessing
├── compute_volume.py           # Volume computation from meshes
│
├── find_params.py              # Hyperparameter tuning
├── evaluate.py                 # Evaluation metrics
│
└── visualize_disp.mlx          # MATLAB visualization script
```

---

## Installation

### Prerequisites
- Python 3.7+

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/clairexuu/4dCTA-ConstrainedEstimate.git
cd 4dCTA-ConstrainedEstimate

# Install dependencies
pip install numpy scipy trimesh pycpd nibabel pydicom
```

---

## Usage

### Step 1: Prepare 4D-CTA Data

Your input should be multi-phase 4D-CTA DICOM or segmented mesh files:

```bash
patient_data/
├── phase_0/   # Systole or reference phase
├── phase_5/
├── phase_10/
...
└── phase_95/
```

### Step 2: Extract Meshes from DICOM (if needed)

```bash
python toMesh.py --input patient_data/ --output meshes/
```

### Step 3: Run CPD Registration

Extract sparse displacement measurements:

```bash
python CPD.py
```

Edit the `main()` function in [CPD.py](CPD.py) to specify:
- `ref_path`: Reference mesh path (usually phase 0)
- `mesh_dir`: Directory containing all phase meshes
- `output_dir`: Output directory for displacement data
- `beta`, `lamb`: CPD smoothness parameters (default: 1.0, 3.0)

### Step 4: Constrained Estimation

Reconstruct dense displacement field:

```bash
python Constrained_Estimate.py
```

Edit the `main()` function to configure:
- `mesh_path`: Reference mesh for stiffness matrix construction
- `disp_path`: Sparse displacement from CPD
- `n_samples`: Number of sparse samples to use
- `E`: Young's modulus for stiffness (default: 1.4e6 Pa)

### Step 5: Evaluate and Visualize

```bash
# Compute volume changes
python Volume_Change.py

# Evaluate accuracy
python evaluate.py
```

For visualization, open `visualize_disp.mlx` in MATLAB.

---

## Citation

```bibtex
@article{xie2024constrained,
  title={Constrained estimation of intracranial aneurysm surface deformation using 4D-CTA},
  author={Xie, Hujin and others},
  journal={Computer Methods and Programs in Biomedicine},
  volume={245},
  pages={108031},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.cmpb.2024.108031}
}
```

**Original Paper:** [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/38128464/)