"""
Piecewise constant field of effective elastic tensors, one 6x6 matrix C_K per macro element K, and its checks
(Stebel, DP Siddall - schema homogenizace, ch. 2.4).

All 6x6 matrices in this module are KELVIN matrices (in the text with a hat: C_K = W C*_K W, W = KELVIN scaling
diag(1, 1, 1, sqrt2, sqrt2, sqrt2) of the Voigt matrix C*_K with engineering shear strains). In the Kelvin basis the
Frobenius norm and the eigenvalues are those of the 4th order tensor.

Input is the field upscale_m writes for Flow123d (macro_elastic_tensor.msh2): per macro element six 3x3 tensors
stiffness_tensor_0 .. stiffness_tensor_5, the j-th being the j-th column of the 6x6 Kelvin tensor put into a
symmetric 3x3 (upscale_m macro_mechanics_model.kelvin_columns_3x3), ordering [xx, yy, zz, yz, xz, xy].

With C_K the Kelvin matrix of a macro element (as read, not symmetrized) and C_K_sym = (C_K + C_K^T) / 2:
    symmetry                delta_sym_K = ||C_K - C_K^T||_F / (2 ||C_K||_F)                           (12)
    positive definiteness   real eigenvalues lambda_1 <= .. <= lambda_6 of C_K_sym (sec:diag-pd):
                            lambda_1 > 0 <=> positive definite => regular;  regular <=> min_i |lambda_i| > 0
    isotropy                delta_iso_K = ||C_K_sym - C_iso_K||_F / ||C_K_sym||_F                       (13)-(15)
                            (M. Moakher, A. N. Norris: v = 1/sqrt(3) (1, 1, 1, 0, 0, 0)^T, J = v v^T, L = I_6 - J,
                            C_iso_K = 3 kappa_K J + 2 mu_K L, kappa_K = tr(J C_K_sym) / 3, mu_K = tr(L C_K_sym) / 10)
    Reuss-Voigt bounds      C_R = <C^-1>^-1  <=  C_K_sym  <=  C_V = <C>                                (16)-(17)
                            C_V = (1 - phi) C_m + phi C_f,  C_R = [(1 - phi) C_m^-1 + phi C_f^-1]^-1

<.> is the volume average over the micro elements captured by the macro element (Omega_K), the same selection
and the same measures as in the homogenization (upscale_m macro_element_averages): a micro element belongs to the
macro element that holds its barycenter g_b, g_f, a bulk element b has its volume vol_b, a fracture element f its
area times its cross_section, vol_f = A_f delta_f. phi = V_f / (V_b + V_f) is the fracture volume fraction.
Rock (matrix) and fracture are isotropic, C_m, C_f = 3 kappa J + 2 mu L.
The bounds are checked in the Loewner order: lambda_min(C_V - C_K_sym) >= 0 and lambda_min(C_K_sym - C_R) >= 0,
relative to ||C_K_sym||_F.

KUBC vs SUBC, the tensors as read (not symmetrized), the same macro element K:
    rel_diff_C_fields_kubc_subc (per K) = ||C_K^KUBC - C_K^SUBC||_F / ||C_K^KUBC||_F
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv
from bgem.gmsh.gmsh_io import GmshIO

# puts the upscale_m sources on sys.path
import common
from bgem.upscale import tn_to_voigt
from endorse.common import File
from endorse.mesh_class import _load_mesh
from postprocess import strain_stress_fields


v = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / np.sqrt(3.0)
J = np.outer(v, v)
L = np.eye(6) - J

# Kelvin scaling of the shear rows/columns of a Voigt tensor with engineering shear strains, C_hat = W C W
# (as upscale_m macro_mechanics_model.kelvin_columns_3x3)
W = np.diag([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])


def distance_from_isotropy(C_K: np.ndarray) -> float:
    # one effective tensor (6, 6), Kelvin -> delta_iso_K, (13)-(15)
    # symmetrized, the distance is defined for the symmetric tensor
    C_K_sym = 0.5 * (C_K + C_K.T)
    kappa_K = np.trace(J @ C_K_sym) / 3.0
    mu_K = np.trace(L @ C_K_sym) / 10.0
    C_iso_K = 3.0 * kappa_K * J + 2.0 * mu_K * L
    delta_iso_K = np.linalg.norm(C_K_sym - C_iso_K, "fro") / np.linalg.norm(C_K_sym, "fro")
    return delta_iso_K


def isotropic_kelvin(young: float, poisson: float) -> np.ndarray:
    # C = 3 kappa J + 2 mu L (C_m, C_f), as C_iso in (15); kappa, mu the bulk and shear modulus of the phase
    kappa = young / (3.0 * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return 3.0 * kappa * J + 2.0 * mu * L


def kelvin_from_columns(cols: np.ndarray) -> np.ndarray:
    # (n, 6, 3, 3) stiffness_tensor_j -> (n, 6, 6) Kelvin C_K; inverse of kelvin_columns_3x3:
    # column j = [T_xx, T_yy, T_zz, sqrt2 T_yz, sqrt2 T_xz, sqrt2 T_xy] of the j-th 3x3 T
    s = np.sqrt(2.0)
    T = cols
    columns = np.stack([T[..., 0, 0], T[..., 1, 1], T[..., 2, 2],
                        s * T[..., 1, 2], s * T[..., 0, 2], s * T[..., 0, 1]], axis=-1)
    return np.swapaxes(columns, -1, -2)


def gmsh_grid(gmsh_io: GmshIO, el_ids: np.ndarray, cell_type: pv.CellType) -> pv.UnstructuredGrid:
    # the gmsh elements el_ids (all of one type) as a pyvista grid, cells in the el_ids order
    # only the nodes of these elements
    el_nodes = np.array([gmsh_io.elements[i][2] for i in el_ids])
    node_ids = np.unique(el_nodes)
    points = np.array([gmsh_io.nodes[i] for i in node_ids], dtype=float)
    # gmsh node id -> row of points
    el_nodes = np.searchsorted(node_ids, el_nodes)
    cells = np.hstack([np.full((len(el_ids), 1), el_nodes.shape[1]), el_nodes]).ravel()
    return pv.UnstructuredGrid(cells, np.full(len(el_ids), cell_type, dtype=np.uint8), points)


def fracture_grid(micro_msh: Path, region_prefix: str) -> pv.UnstructuredGrid:
    # all fracture triangles of the micro mesh (regions <region_prefix>*, as in the upscale_m config
    # geometry.fracture_region_prefix), of all fractures, to be shown in ParaView together with the tensor field;
    # region_id the gmsh region, fracture_id the number of the fracture = the number at the end of the region name
    # (fam_0_001 -> 1, the index in fractures.fixed, as in upscale_m MicroMesh.fracture_radius_by_region_id)
    gmsh_io = GmshIO(str(micro_msh))
    fr_regions = {tag: name for name, (tag, dim) in gmsh_io.physical.items()
                  if dim == 2 and name.startswith(region_prefix)}
    # gmsh element type of a linear triangle, GMSH_TRIANGLE = 2
    el_ids = np.array([i for i in sorted(gmsh_io.elements)
                       if gmsh_io.elements[i][0] == 2 and gmsh_io.elements[i][1][0] in fr_regions])
    grid = gmsh_grid(gmsh_io, el_ids, pv.CellType.TRIANGLE)
    region_id = np.array([gmsh_io.elements[i][1][0] for i in el_ids])
    grid.cell_data["region_id"] = region_id
    grid.cell_data["fracture_id"] = np.array([int(fr_regions[r].rsplit("_", 1)[1]) for r in region_id])
    return grid


class TensorField:
    """
    The macro tetrahedra K of one macro_elastic_tensor.msh2 (the boundary triangles carry zero tensors and are
    dropped) with their Kelvin matrices C_K (n, 6, 6). The support tetrahedra (used by SUBC only) are not
    homogenized (they carry the rock tensor), is_support_tetra marks them.
    """

    def __init__(self, msh2: Path, prefix: str, support_region: str):
        gmsh_io = GmshIO(str(msh2))
        el_ids = np.array(sorted(gmsh_io.elements))
        # gmsh element type of a linear tetrahedron, GMSH_TETRA = 4
        tetra = np.array([gmsh_io.elements[i][0] == 4 for i in el_ids])
        cols = []
        for j in range(6):
            item = gmsh_io.element_data[f"{prefix}{j}"][0]
            assert np.array_equal(np.asarray(item.tags), el_ids), f"{prefix}{j}: element tags differ from the mesh"
            cols.append(np.asarray(item.values, dtype=float).reshape(-1, 3, 3))
        self.msh2 = Path(msh2)
        self.macro_el_ids = el_ids[tetra]
        self.C_K = kelvin_from_columns(np.stack(cols, axis=1)[tetra])
        self.C_K_sym = 0.5 * (self.C_K + np.swapaxes(self.C_K, 1, 2))
        self.grid = gmsh_grid(gmsh_io, self.macro_el_ids, pv.CellType.TETRA)
        support_id = gmsh_io.physical.get(support_region, (None, None))[0]
        self.is_support_tetra = np.array([gmsh_io.elements[i][1][0] == support_id for i in self.macro_el_ids])

    def same_mesh(self, other: "TensorField") -> bool:
        return (np.array_equal(self.macro_el_ids, other.macro_el_ids)
                and np.array_equal(self.grid.cell_connectivity, other.grid.cell_connectivity)
                and np.array_equal(self.grid.points, other.grid.points))

    def symmetry(self) -> np.ndarray:
        # delta_sym_K, (12)
        norm_C_K = np.linalg.norm(self.C_K, "fro", axis=(1, 2))
        return np.linalg.norm(self.C_K - np.swapaxes(self.C_K, 1, 2), "fro", axis=(1, 2)) / (2.0 * norm_C_K)

    def eigenvalues(self) -> np.ndarray:
        # (n, 6) eigenvalues lambda_1 <= .. <= lambda_6 of C_K_sym
        return np.linalg.eigvalsh(self.C_K_sym)

    def isotropy(self) -> np.ndarray:
        # delta_iso_K, (13)-(15)
        return np.array([distance_from_isotropy(C_K) for C_K in self.C_K])

    def reuss_voigt_margins(self, phi_macro_el: np.ndarray, C_m: np.ndarray, C_f: np.ndarray):
        return reuss_voigt_margins(self.C_K_sym, phi_macro_el, C_m, C_f)


def reuss_voigt_margins(C_sym: np.ndarray, phi: np.ndarray, C_m: np.ndarray, C_f: np.ndarray):
    # (16)-(17) for symmetric C_sym (n, 6, 6) (C_K_sym of the macro elements) with the fracture volume
    # fractions phi (n,): lambda_min(C_V - C_sym) / ||C_sym||_F and lambda_min(C_sym - C_R) / ||C_sym||_F
    phi = phi[:, None, None]
    C_V = (1.0 - phi) * C_m + phi * C_f
    C_R = np.linalg.inv((1.0 - phi) * np.linalg.inv(C_m) + phi * np.linalg.inv(C_f))
    norm_C_sym = np.linalg.norm(C_sym, "fro", axis=(1, 2))
    voigt_margin = np.linalg.eigvalsh(C_V - C_sym)[:, 0] / norm_C_sym
    reuss_margin = np.linalg.eigvalsh(C_sym - C_R)[:, 0] / norm_C_sym
    return voigt_margin, reuss_margin


def micro_measures(micro_msh: Path, cross_section_msh: Path, bulk_region: str, fracture_prefix: str):
    # barycenters and measures of the micro elements the homogenization averages over:
    # bulk tetrahedra b (g_b, vol_b = volume) and fracture triangles f (g_f, vol_f = area * cross_section)
    gmsh_io = GmshIO(str(micro_msh))
    bulk_id = gmsh_io.physical[bulk_region][0]
    fr_ids = {tag for name, (tag, dim) in gmsh_io.physical.items() if dim == 2 and name.startswith(fracture_prefix)}
    cs_item = GmshIO(str(cross_section_msh)).element_data["cross_section"][0]
    # cross_section = delta_f, the thickness of the fracture element
    cross_section = dict(zip(np.asarray(cs_item.tags), np.asarray(cs_item.values, dtype=float).ravel()))

    # gmsh element types: GMSH_TETRA = 4, GMSH_TRIANGLE = 2
    bulk = [i for i, e in gmsh_io.elements.items() if e[0] == 4 and e[1][0] == bulk_id]
    frac = [i for i, e in gmsh_io.elements.items() if e[0] == 2 and e[1][0] in fr_ids]
    xyz = lambda ids: np.array([[gmsh_io.nodes[n] for n in gmsh_io.elements[i][2]] for i in ids], dtype=float)
    P_b, P_f = xyz(bulk), xyz(frac)
    vol_b = np.abs(np.linalg.det(P_b[:, 1:] - P_b[:, :1])) / 6.0
    area_f = 0.5 * np.linalg.norm(np.cross(P_f[:, 1] - P_f[:, 0], P_f[:, 2] - P_f[:, 0]), axis=1)
    vol_f = area_f * np.array([cross_section[i] for i in frac])
    g_b, g_f = P_b.mean(axis=1), P_f.mean(axis=1)
    return g_b, vol_b, g_f, vol_f


def containing_tetra(grid: pv.UnstructuredGrid, points: np.ndarray) -> np.ndarray:
    # the macro tetrahedron K of each point by the rule of endorse MacroTetra(rel_radius=1.0): all barycentric
    # coordinates > 0 (-1 if none). vtk's find_containing_cell is only the first guess, it has a tolerance
    tets = grid.points[grid.cell_connectivity.reshape(-1, 4)]
    T_K_inv = np.linalg.inv(np.swapaxes(tets[:, 1:] - tets[:, :1], 1, 2))

    def min_barycentric(cells, pts):
        local = np.einsum("nij,nj->ni", T_K_inv[cells], pts - tets[cells, 0])
        return np.minimum(1.0 - local.sum(axis=1), local.min(axis=1))

    K_of_point = grid.find_containing_cell(points)
    wrong = np.flatnonzero((K_of_point < 0) | (min_barycentric(np.maximum(K_of_point, 0), points) <= 0.0))
    # a point outside the bounding box of the macro mesh is in no tetrahedron, no search needed
    lo, hi = tets.reshape(-1, 3).min(axis=0), tets.reshape(-1, 3).max(axis=0)
    outside = np.any((points[wrong] <= lo) | (points[wrong] >= hi), axis=1)
    K_of_point[wrong[outside]] = -1
    for i in wrong[~outside]:
        all_cells = np.arange(len(tets))
        inside = np.flatnonzero(min_barycentric(all_cells, np.broadcast_to(points[i], (len(tets), 3))) > 0.0)
        K_of_point[i] = inside[0] if len(inside) else -1
    return K_of_point


def fracture_fraction(grid: pv.UnstructuredGrid, micro) -> np.ndarray:
    # phi_macro_el = V_f / (V_b + V_f) per macro element K, the micro elements assigned by containing_tetra
    g_b, vol_b, g_f, vol_f = micro
    K_of_b, K_of_f = containing_tetra(grid, g_b), containing_tetra(grid, g_f)
    V_b = np.bincount(K_of_b[K_of_b >= 0], weights=vol_b[K_of_b >= 0], minlength=grid.n_cells)
    V_f = np.bincount(K_of_f[K_of_f >= 0], weights=vol_f[K_of_f >= 0], minlength=grid.n_cells)
    return V_f / (V_b + V_f)


def case_averages(case_files: List[Path], grid: pv.UnstructuredGrid, bulk_id: int,
                  aperture_by_region_id: Dict[int, float], mats: dict, stress_field: str,
                  region_id_field: str) -> np.ndarray:
    """
    The average strains <eps> of the six micro solutions of one BC type over each macro element K of grid, by the
    averaging of upscale_m (macro_element_averages): <.> = sum(vol * field) / V_b, V_b the bulk volume, a bulk element
    measured by its volume, a fracture element by its area times its aperture, the strain from the stress by the
    compliance of the element's phase (upscale_m postprocess.strain_stress_fields), the micro elements by barycenter
    (containing_tetra). Voigt vectors with engineering shear strains.
    Returns E_K (n, 6, 6), columns = load cases, nan for a macro element without bulk micro elements; it gives the
    2-norm condition number cond2_strain_matrix_E (the conditioning of the fit of C*_K in upscale_m).
    """
    # engineering shear on the Voigt strain shear rows: gamma = 2 eps
    eng_shear = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    fracture_ids = list(aperture_by_region_id)
    n_K = grid.n_cells
    E_K_cols = []
    g_ref = K_of_el = None
    for case_file in case_files:
        mesh = _load_mesh(File(str(case_file)), None)
        rid, eps9, _ = strain_stress_fields(
            mesh, bulk_id, fracture_ids,
            float(mats["young_modulus_rock"]), float(mats["poisson_ratio_rock"]),
            float(mats["young_modulus_fracture"]), float(mats["poisson_ratio_fracture"]),
            stress_field=stress_field, region_id_field=region_id_field)
        is_bulk = rid == bulk_id
        is_frac = np.isin(rid, fracture_ids)
        aperture = np.array([aperture_by_region_id.get(r, 1.0) for r in rid])
        # the bulk and fracture elements only (not the SUBC support tetrahedra)
        vol = np.where(is_bulk | is_frac, mesh.el_volumes * np.where(is_frac, aperture, 1.0), 0.0)

        # the assignment by barycenter once, the six outputs share the mesh
        g = mesh.el_barycenters()
        if g_ref is None:
            g_ref = g
            K_of_el = np.where(is_bulk | is_frac, containing_tetra(grid, g), -1)
        assert np.array_equal(g, g_ref), f"{case_file}: the micro mesh differs from the other load cases"
        inK = K_of_el >= 0
        V_b_K = np.bincount(K_of_el[inK & is_bulk], weights=vol[inK & is_bulk], minlength=n_K)
        sum_eps = np.zeros((n_K, 9))
        np.add.at(sum_eps, K_of_el[inK], vol[inK, None] * eps9[inK])
        with np.errstate(invalid="ignore", divide="ignore"):
            eps_K = sum_eps / V_b_K[:, None]
        E_K_cols.append(tn_to_voigt(eps_K.reshape(-1, 3, 3)) * eng_shear)
    return np.stack(E_K_cols, axis=-1)


def condition_numbers(M: np.ndarray) -> np.ndarray:
    # 2-norm condition number sigma_max / sigma_min of each (6, 6) matrix of M (n, 6, 6); nan stays nan
    out = np.full(len(M), np.nan)
    ok = np.all(np.isfinite(M), axis=(1, 2))
    out[ok] = np.linalg.cond(M[ok])
    return out
