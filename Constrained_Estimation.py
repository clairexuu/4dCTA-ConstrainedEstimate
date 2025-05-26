import os
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh
from glob import glob
import gmsh
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pycpd import DeformableRegistration
from trimesh import repair

def extract_surface_meshes(nii_dir, out_dir, level=0.5):
    """
    Extract surface meshes from a directory of .nii.gz binary masks and save as .stl.
    """
    os.makedirs(out_dir, exist_ok=True)
    nii_files = sorted(glob(os.path.join(nii_dir, "*.nii.gz")),
                       key=lambda f: int(os.path.basename(f).split("pct")[0]))

    for file_path in nii_files:
        name = os.path.basename(file_path).replace(".nii.gz", "")
        img = nib.load(file_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()

        binary = (data > 0.5).astype(np.uint8)
        verts, faces, _, _ = measure.marching_cubes(binary, level=level, spacing=spacing)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        mesh.export(os.path.join(out_dir, f"{name}.stl"))
        print(f"Exported mesh: {name}.stl")


def generate_gmsh_volume(
    stl_path,
    mesh_save_path="output.msh",
    max_element_size=1.0,
    reclassify=True,
    verbose=True,):
    """
    Generate a tetrahedral volume mesh from an STL surface mesh using GMSH.
    """
    # Optional: Fix STL winding if volume is negative
    mesh = trimesh.load(stl_path)

    print(f"🔍 Watertight: {mesh.is_watertight}")
    print(f"🔍 Euler number: {mesh.euler_number}")
    print(f"🔍 Non-manifold edges: {mesh.edges_unique.shape[0] - mesh.edges.shape[0]}")
    print(f"🔍 Volume: {mesh.volume:.2f}")

    gmsh.initialize()
    gmsh.model.add("volume_mesh")

    try:
        gmsh.merge(stl_path)
    except Exception as e:
        gmsh.finalize()
        raise RuntimeError(f"❌ Failed to load STL: {e}")

    gmsh.option.setNumber("Mesh.MeshSizeMax", max_element_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_element_size)

    if reclassify:
        gmsh.model.mesh.classifySurfaces(40, True, 1, True)
        gmsh.model.mesh.createGeometry()

    surface_tags = [s[1] for s in gmsh.model.getEntities(2)]
    if not surface_tags:
        gmsh.finalize()
        raise RuntimeError("❌ No surfaces found in STL.")

    gmsh.model.geo.addSurfaceLoop(surface_tags, tag=1)
    gmsh.model.geo.addVolume([1], tag=1)
    gmsh.model.geo.synchronize()

    if verbose:
        print(f"🧪 Surface has {len(surface_tags)} discrete surfaces")
        volumes = gmsh.model.getEntities(3)
        print(f"✅ GMSH found {len(volumes)} volume(s)")

    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_save_path)
    if verbose:
        print(f"✅ GMSH wrote volume mesh to {mesh_save_path}")

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)

    element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()

    elements = None
    surface_faces = None
    for etype, enodes in zip(element_types, element_nodes):
        if etype == 4:  # Tetrahedra
            elem_array = np.array(enodes, dtype=np.int64)
            if elem_array.size % 4 != 0:
                gmsh.finalize()
                raise ValueError(f"❌ Invalid tetrahedral element size: {elem_array.size}")
            elements = elem_array.reshape(-1, 4) - 1
            if verbose:
                print(f"✅ Extracted {elements.shape[0]} tetrahedra.")
        elif etype == 2:  # Triangle surface faces
            face_array = np.array(enodes, dtype=np.int64)
            if face_array.size % 3 != 0:
                gmsh.finalize()
                raise ValueError(f"❌ Invalid triangle face size: {face_array.size}")
            surface_faces = face_array.reshape(-1, 3) - 1

    gmsh.finalize()

    if elements is None:
        raise ValueError("❌ No tetrahedral elements found in the mesh.")

    return nodes, elements, surface_faces


def compute_D_matrix(E, nu):
    """Construct the elasticity (constitutive) matrix D for isotropic material."""
    coeff = E / ((1 + nu) * (1 - 2 * nu))
    D = coeff * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ])
    return D


def tet_volume(a, b, c, d):
    return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0


def assemble_stiffness_matrix(nodes, elements, E=1.4e6, nu=0.47):
    """
    Assemble global FEM stiffness matrix K for linear elasticity.
    Using Young’s modulus and Poisson’s ratio from the Paper

    Args:
        nodes: (N, 3) array of node positions
        elements: (M, 4) array of tetrahedral node indices
        E: Young's modulus
        nu: Poisson's ratio

    Returns:
        scipy.sparse.csr_matrix: Global stiffness matrix K of shape (3N, 3N)
    """
    N = nodes.shape[0]
    D = compute_D_matrix(E, nu)

    K = sp.lil_matrix((3 * N, 3 * N))

    for tet in elements:
        tet_nodes = nodes[tet]  # shape (4, 3)
        v0, v1, v2, v3 = tet_nodes

        # Construct B matrix
        A = np.array([
            [1, *v0],
            [1, *v1],
            [1, *v2],
            [1, *v3]
        ])
        volume = np.abs(np.linalg.det(A)) / 6.0
        if volume < 1e-10:
            continue  # skip degenerate tets

        grads = np.linalg.inv(A)[1:, :]  # shape (3, 4)

        B = np.zeros((6, 12))  # 6 strains, 12 dofs (3 per node)
        for i in range(4):
            bi, ci, di = grads[:, i]
            B[:, 3 * i:3 * i + 3] = [
                [bi, 0, 0],
                [0, ci, 0],
                [0, 0, di],
                [ci, bi, 0],
                [0, di, ci],
                [di, 0, bi]
            ]

        Ke = B.T @ D @ B * volume

        dof = np.array([3 * tet[i] + j for i in range(4) for j in range(3)])
        for i in range(12):
            for j in range(12):
                K[dof[i], dof[j]] += Ke[i, j]

    return K.tocsr()

def build_H_y(sparse_node_indices, reference_positions, target_positions, total_nodes):
    """
    Constructs H matrix and y vector for constrained estimation.

    Args:
        sparse_node_indices: indices into the full mesh (length n)
        reference_positions: (n, 3) original positions (subset of full)
        target_positions: (n, 3) deformed positions from CPD
        total_nodes: total number of nodes in the full mesh

    Returns:
        H (scipy.sparse.csr_matrix): shape (3n, 3N)
        y (np.array): shape (3n,)
    """
    n = len(sparse_node_indices)
    rows, cols, data = [], [], []
    y = np.zeros(3 * n)

    for i in range(n):
        global_idx = sparse_node_indices[i]
        for j in range(3):  # x, y, z
            rows.append(3 * i + j)
            cols.append(3 * global_idx + j)
            data.append(1.0)

        displacement = target_positions[i] - reference_positions[i]
        y[3 * i:3 * i + 3] = displacement

    H = sp.csr_matrix((data, (rows, cols)), shape=(3 * n, 3 * total_nodes))
    return H, y



def constrained_estimation(K, H, y, max_iter=10, tol=1e-6):
    """
    Iteratively estimate the full displacement vector U using constrained estimation.

    Args:
        K: sparse stiffness matrix (3N x 3N)
        H: sampling matrix (3n x 3N)
        y: observed displacements at sparse points (3n,)
        max_iter: max iterations
        tol: convergence tolerance

    Returns:
        U_est: estimated displacement vector (3N,)
    """
    N = K.shape[0]
    U = np.zeros(N)

    # Precompute K⁻¹ via a linear solver
    K_solver = spla.factorized(K.tocsc())

    # Compute L = K⁻¹ Hᵀ (H K⁻¹ Hᵀ)⁻¹
    HKinvT = H @ K_solver(np.eye(N))
    M = HKinvT @ H.T
    Minv = np.linalg.pinv(M)
    L = K_solver(H.T @ Minv)

    for it in range(max_iter):
        residual = y - H @ U
        step = L @ residual
        U += step

        if np.linalg.norm(step) < tol:
            print(f"✅ Converged in {it+1} iterations.")
            break
    else:
        print(f"⚠️ Reached max iterations ({max_iter}) without full convergence.")

    return U

def deform_mesh(nodes, U_est):
    """Apply estimated displacements to original mesh nodes."""
    return nodes + U_est.reshape(-1, 3)


def compute_mesh_volume(vertices, faces):
    """
    Compute volume of a triangular mesh using trimesh
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh.volume

def clean_stl_keep_main_volume(stl_path, clean_dir, apply_hole_fill=False, verbose=True):
    """
    Clean STL while preserving geometric fidelity. No convex hull unless manually enabled.

    Args:
        stl_path (str): Path to original STL
        clean_dir (str): Directory to save cleaned STL
        apply_hole_fill (bool): Whether to attempt hole filling
        verbose (bool): Whether to log status

    Returns:
        str: Path to cleaned STL
    """
    mesh = trimesh.load(stl_path)
    components = mesh.split(only_watertight=False)

    if not components:
        raise ValueError(f"❌ No components found in mesh: {stl_path}")

    if len(components) > 1:
        if verbose:
            print(f"⚠️ Found {len(components)} components in {os.path.basename(stl_path)}")
        # Instead of blindly picking largest, show each?
        main_mesh = sorted(components, key=lambda m: m.volume if m.is_volume else len(m.faces), reverse=True)[0]
    else:
        main_mesh = components[0]

    if main_mesh.volume < 0:
        main_mesh.invert()

    if verbose:
        print(f"🧼 Cleaning {os.path.basename(stl_path)}")
        print(f"   - Initial watertight: {main_mesh.is_watertight}")
        print(f"   - Faces: {len(main_mesh.faces)}, Volume: {main_mesh.volume:.2f} mm³")

    # Only apply lightweight cleanup
    main_mesh.remove_unreferenced_vertices()
    if apply_hole_fill:
        main_mesh.fill_holes()

    # Save
    file_name = os.path.basename(stl_path).replace(".stl", "_cleaned.stl")
    save_path = os.path.join(clean_dir, file_name)
    os.makedirs(clean_dir, exist_ok=True)
    main_mesh.export(save_path)

    if verbose:
        print(f"✅ Cleaned STL saved: {save_path}")
        print(f"    Watertight: {main_mesh.is_watertight}, Faces: {len(main_mesh.faces)}, Volume: {main_mesh.volume:.2f} mm³")

    return save_path

def load_and_repair_stl(stl_path, repair_dir, verbose=True):
    mesh = trimesh.load(stl_path)

    if verbose:
        print(f"Before repair:")
        print(f"  Watertight: {mesh.is_watertight}")
        print(f"  Volume: {mesh.volume:.2f}")
        print(f"  Euler number: {mesh.euler_number}")

    # mesh.remove_duplicate_faces()
    # mesh.remove_unreferenced_vertices()
    # mesh.update_faces(mesh.nondegenerate_faces())
    # mesh.fill_holes()
    # repair.fix_normals(mesh)
    # repair.fix_inversion(mesh)

    if not mesh.is_watertight:
        print("⚠️ Still not watertight. Using convex hull.")
        mesh = mesh.convex_hull

    if verbose:
        print(f"After repair:")
        print(f"  Watertight: {mesh.is_watertight}")
        print(f"  Volume: {mesh.volume:.2f}")
        print(f"  Euler number: {mesh.euler_number}")

    file_name = os.path.basename(stl_path).replace(".stl", "_repaired.stl")
    save_path = os.path.join(repair_dir, file_name)
    os.makedirs(repair_dir, exist_ok=True)
    mesh.export(save_path)

    return mesh


def inspect_stl_components(stl_path, verbose=True):
    """
    Load an STL file and report specs of each connected component.

    Args:
        stl_path (str): Path to the STL file.
        verbose (bool): Print detailed component info.

    Returns:
        List[Dict]: List of dictionaries with component stats.
    """
    mesh = trimesh.load(stl_path)
    components = mesh.split(only_watertight=False)

    if not components:
        raise ValueError(f"❌ No components found in mesh: {stl_path}")

    print(f"📦 {len(components)} components found in {os.path.basename(stl_path)}:")

    report = []
    for i, comp in enumerate(components):
        data = {
            "index": i,
            "num_faces": len(comp.faces),
            "volume": float(comp.volume),
            "watertight": comp.is_watertight,
            "bbox_size": comp.bounding_box.extents,
        }
        if verbose:
            print(f"  • Component {i}: {data['num_faces']} faces | "
                  f"Vol: {data['volume']:.2f} mm³ | "
                  f"Watertight: {data['watertight']} | "
                  f"BBox: {data['bbox_size']}")
        report.append(data)

    return report

def select_sparse_correspondences(ref_verts, tgt_verts, n_sparse, seed=0, verbose=True):
    """
    Randomly select sparse points from both meshes, register with CPD, and return displacements.

    Args:
        ref_verts (np.ndarray): (N, 3) array of reference mesh vertices (e.g., phase 0)
        tgt_verts (np.ndarray): (M, 3) array of target mesh vertices (e.g., phase N)
        n_sparse (int): Number of sparse points to use
        seed (int): Random seed for reproducibility
        verbose (bool): Whether to print progress info

    Returns:
        ref_sparse (np.ndarray): (n_sparse, 3) sampled reference points
        TY (np.ndarray): (n_sparse, 3) registered target points
        indices (np.ndarray): sampled indices into the ref_verts
    """
    np.random.seed(seed)
    n_points = len(ref_verts)
    if n_sparse > n_points:
        raise ValueError(f"Requested {n_sparse} points, but reference mesh has only {n_points} vertices.")

    indices = np.random.choice(n_points, size=n_sparse, replace=False)

    ref_sparse = np.asarray(ref_verts[indices])
    tgt_sparse = np.asarray(tgt_verts[np.random.choice(len(tgt_verts), size=n_sparse, replace=False)])

    if ref_sparse.ndim != 2 or ref_sparse.shape[1] != 3:
        raise ValueError(f"❌ ref_sparse has invalid shape: {ref_sparse.shape}")
    if tgt_sparse.ndim != 2 or tgt_sparse.shape[1] != 3:
        raise ValueError(f"❌ tgt_sparse has invalid shape: {tgt_sparse.shape}")

    # Register target to reference using CPD
    if verbose:
        print(f"📌 Running CPD on {n_sparse} points...")
    reg = DeformableRegistration(X=ref_sparse, Y=tgt_sparse, max_iterations=30)
    TY, _ = reg.register()

    return ref_sparse, TY, indices

from scipy.spatial import cKDTree

def map_surface_to_volume_nodes(sparse_surface_points, volume_nodes):
    """
    For each sparse point from surface mesh, find the nearest node in volume mesh.

    Args:
        sparse_surface_points: (n, 3) subset of reference STL vertices
        volume_nodes: (N, 3) all volume mesh node positions

    Returns:
        np.ndarray: (n,) array of indices into volume_nodes
    """
    tree = cKDTree(volume_nodes)
    dists, indices = tree.query(sparse_surface_points)
    return indices

# TODO: Why n_sparse cannot be more than 200?
def constrained_estimate(reference_phase_path, target_phase_paths, n_sparse):
    """
    Estimate volume changes using constrained FEM estimation with CPD alignment.

    Args:
        reference_phase_path (str): Path to the reference STL.
        target_phase_paths (List[str]): List of STL paths for target phases.
        n_sparse (int): Number of sparse correspondence points.

    Returns:
        Tuple[List[str], List[float]]: (list of successful phase names, estimated volumes)
    """
    nodes, elements, ref_faces = generate_gmsh_volume(reference_phase_path)
    K = assemble_stiffness_matrix(nodes, elements)

    ref_mesh = trimesh.load(reference_phase_path)
    ref_verts = np.asarray(ref_mesh.vertices)

    estimated_volumes = []
    successful_phase_names = []

    for path in target_phase_paths:
        phase_name = os.path.basename(path).replace(".stl", "")

        try:
            tgt_mesh = trimesh.load(path)
            tgt_verts = np.asarray(tgt_mesh.vertices)

            ref_sparse, TY, _ = select_sparse_correspondences(ref_verts, tgt_verts, n_sparse)

            # ⚠️ Map sparse surface points to volume mesh node indices
            sparse_node_indices = map_surface_to_volume_nodes(ref_sparse, nodes)

            H, y = build_H_y(sparse_node_indices, ref_sparse, TY, total_nodes=len(nodes))

            U_est = constrained_estimation(K, H, y)
            deformed_nodes = deform_mesh(nodes, U_est)
            volume_est = compute_mesh_volume(deformed_nodes, ref_faces)

            estimated_volumes.append(volume_est)
            successful_phase_names.append(phase_name)

        except Exception as e:
            print(f"❌ Skipping {phase_name}: {e}")

    return successful_phase_names, estimated_volumes