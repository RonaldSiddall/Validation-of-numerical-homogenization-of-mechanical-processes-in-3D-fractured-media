"""
Analysis of the effective elastic tensor field of one run of scripts/main.py (Stebel, DP Siddall - schema
homogenizace, ch. 2.4):
- per macro element K and BC type: the condition number of the strain matrix E_K of the fit, symmetry, positive
  definiteness and regularity, isotropy and the
  Reuss-Voigt (HRV) bounds of its tensor C_K (fields with the suffix _macro_el);
- KUBC vs SUBC (when the run has both): the relative difference of the tensor fields, per macro element, and
  over the whole Omega_E delta C_L2 (the L2 norm of the difference of the fields, dC_L2).
The quantities and their formulas are in tensor_field.py. All tensors are Kelvin matrices.

Symmetry first: delta_sym_macro_el of the raw tensors C_K (as upscale_m fitted them, in general not symmetric);
the field is symmetric up to the numerics if delta_sym_macro_el <= tol_rel_delta_sym on all macro elements
(summary.txt, section SYMMETRY: the count above the tolerance, the maximum and its macro element). The symmetry is
then enforced in any case, C_K_sym = 1/2 (C_K + C_K^T), and all further checks (eigenvalues, isotropy, HRV bounds)
use C_K_sym. upscale_m writes this symmetrized field next to the raw one (tensor_field_file_sym); when it exists,
it is checked to be 1/2 (C_K + C_K^T) of the raw field. The KUBC vs SUBC difference uses the raw fields.

Everything else the script needs comes from the copy of the config that scripts/main.py saved in the run dir
(<run_dir>/config.yaml): the file names, the region names, the materials, the fractures (fractures.fixed) and
the BC types. The criteria of the flags is_macro_el_sym, is_cond2_of_E_good, is_macro_el_iso and
is_macro_el_within_hrv (CRITERIA: tol_rel_delta_sym, tol_cond2_strain_matrix_E, tol_rel_delta_iso,
tol_rel_hrv_margin) are in the analysis config; is_C_eff_pd (lambda_min > 0) and is_C_eff_regular
(|lambda_minabs| > 0) have no tolerance (sec:diag-pd). The eigenvalues of C_K_sym reported: lambda_max, lambda_min
and lambda_minabs (the one with the smallest absolute value, signed).

Output (config key out, by default <run_dir>/micro_macro/analysis):
    <bc>_analysis/tensor_field_analysis_<bc>/tensor_field_<bc>.vtu    all fields of the macro elements and the
                                                                      tensors tensor_C_field_<bc> (raw) and
                                                                      tensor_C_field_<bc>_sym (symmetrized)
    <bc>_analysis/tensor_field_analysis_<bc>/non_isotropic_<bc>.vtu   the macro elements with is_macro_el_iso = 0
    <bc>_analysis/tensor_field_analysis_<bc>/all_fracture_elements_<bc>.vtu  all fracture triangles of the micro
                                                                      mesh (region_id, fracture_id)
    <bc>_analysis/tensor_field_analysis_<bc>.vtm                      the three above as one ParaView multiblock
    <bc>_analysis/per_element_<bc>.txt                                all fields, one row per macro element
    diff_kubc_subc.vtu                                                (runs with both BCs) tensor_C_field_kubc,
                                                                      tensor_C_field_subc, rel_diff_C_fields_kubc_subc
    diff_kubc_subc.txt                                                rel_diff_C_fields_kubc_subc, one row per macro
                                                                      element
    summary.txt                                                       input files, parameters, statistics
tensor_C_field_<bc> (and tensor_C_field_kubc, tensor_C_field_subc) are the tensors C_K as read (not symmetrized),
tensor_C_field_<bc>_sym the symmetrized C_K_sym, each the 6x6 Kelvin matrix as 36 components, row-major, named
C_11 .. C_66; they are not in the txt tables.
The support tetrahedra (in the mesh of both BC types, used by SUBC only) are not homogenized (they carry the
rock tensor): they stay in the vtu files and in the tables (is_support_tetra = 1, no bounds: nan), but not in the
statistics and not in non_isotropic_<bc>.vtu.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from bgem.gmsh.gmsh_io import GmshIO

from tensor_field import (TensorField, case_averages, condition_numbers, fracture_fraction, fracture_grid,
                          isotropic_kelvin, micro_measures)

THESIS_DIR = Path(__file__).resolve().parents[1]
RULE = "=" * 64

# the config keys of the criteria of the checks per macro element (comments in the config)
CRITERIA = ("tol_rel_delta_sym", "tol_rel_delta_iso", "tol_rel_hrv_margin", "tol_cond2_strain_matrix_E")
# the symmetrized field upscale_m writes must be 1/2 (C_K + C_K^T) of the raw one up to the rounding
TOL_REL_SYM_FILE = 1e-12


def bc_fields(field: TensorField, phi_macro_el: np.ndarray, C_m: np.ndarray, C_f: np.ndarray,
              cond2_E: np.ndarray, crit: dict) -> dict:
    # real eigenvalues lambda_1 <= .. <= lambda_6 of the symmetric C_K_sym
    lambdas = field.eigenvalues()
    delta_iso = field.isotropy()
    voigt_margin, reuss_margin = field.reuss_voigt_margins(phi_macro_el, C_m, C_f)
    voigt_margin[field.is_support_tetra] = np.nan
    reuss_margin[field.is_support_tetra] = np.nan
    cond2_E = np.where(field.is_support_tetra, np.nan, cond2_E)
    # the eigenvalue with the smallest absolute value, WITH its sign: lambda_k, k = argmin_i |lambda_i(C_K_sym)|
    lambda_minabs = np.take_along_axis(lambdas, np.argmin(np.abs(lambdas), axis=1)[:, None], axis=1)[:, 0]
    tol_hrv = crit["tol_rel_hrv_margin"]
    # of the raw tensors C_K; everything below of C_K_sym
    delta_sym = field.symmetry()
    return {
        # 2-norm condition number of the matrix E_K of the averaged strains (columns = the six load cases), from
        # which C_K is fitted; NOT the condition number of the tensor C_K
        "cond2_strain_matrix_E": cond2_E,
        "is_cond2_of_E_good": (cond2_E <= crit["tol_cond2_strain_matrix_E"]).astype(np.uint8),
        "delta_sym_macro_el": delta_sym,
        "is_macro_el_sym": (delta_sym <= crit["tol_rel_delta_sym"]).astype(np.uint8),
        # three eigenvalues of C_K_sym show a pathological tensor: (i) the largest, (ii) the smallest, (iii) the one
        # with the smallest absolute value (signed); lambda_min > 0: positive definite, lambda_min < 0 < lambda_max:
        # indefinite, lambda_max < 0: negative definite, lambda_minabs ~ 0: (nearly) singular
        "lambda_max_macro_el": lambdas[:, -1],
        # (sec:diag-pd) lambda_min(C_K_sym) > 0  <=>  C_K_sym is positive definite  =>  C_K_sym is regular
        #              lambda_min(C_K_sym) <= 0 <=>  C_K_sym is not positive definite
        "lambda_min_macro_el": lambdas[:, 0],
        "is_C_eff_pd": (lambdas[:, 0] > 0.0).astype(np.uint8),
        # (sec:diag-pd) C_K_sym is regular  <=>  min_i |lambda_i(C_K_sym)| = |lambda_minabs| > 0
        "lambda_minabs_macro_el": lambda_minabs,
        "is_C_eff_regular": (np.abs(lambda_minabs) > 0.0).astype(np.uint8),
        "delta_iso_macro_el": delta_iso,
        "is_macro_el_iso": (delta_iso <= crit["tol_rel_delta_iso"]).astype(np.uint8),
        "voigt_margin_macro_el": voigt_margin,
        "reuss_margin_macro_el": reuss_margin,
        "is_macro_el_within_hrv": ((voigt_margin >= -tol_hrv) & (reuss_margin >= -tol_hrv)).astype(np.uint8),
    }


def diff_kubc_subc(C_kubc: np.ndarray, C_subc: np.ndarray) -> np.ndarray:
    # ||C^KUBC - C^SUBC||_F / ||C^KUBC||_F of (..., 6, 6), the tensors as read (not symmetrized)
    return np.linalg.norm(C_kubc - C_subc, "fro", axis=(-2, -1)) / np.linalg.norm(C_kubc, "fro", axis=(-2, -1))


def dC_L2(C_hat_K_kubc: np.ndarray, C_hat_K_subc: np.ndarray, V_K: np.ndarray) -> float:
    """
    Relative difference of the fields C*_KUBC and C*_SUBC in the L2 norm (notes week 6, "Definice L2 normy pro
    rozdil KUBC a SUBC"):
        (eq:dC_l2)      delta C_L2 [%] = 100 * ||C*_KUBC - C*_SUBC||_2 / ||C*_KUBC||_2
                                       = 100 * sqrt( int_Omega_E |C*_KUBC - C*_SUBC|^2 dV / int_Omega_E |C*_KUBC|^2 dV )
    both fields are constant on each macro element K, so by (eq:int-sum-const) int_Omega_E g dV = sum_K V_K g_K:
        (eq:dC_l2_sum)  delta C_L2 [%] = 100 * sqrt( sum_K V_K |C*_K,KUBC - C*_K,SUBC|^2 / sum_K V_K |C*_K,KUBC|^2 )
    K in T_H^hom (the homogenized macro elements, without the support tetrahedra), V_K = int_K dV the volume of K,
    |C*_K| = ||C_hat_K||_F the Frobenius norm of the Kelvin matrix C_hat_K = W C*_K W, W = diag(1, 1, 1, s2, s2, s2).
    C_hat_K_kubc, C_hat_K_subc: (M_hom, 6, 6) Kelvin matrices of the K in T_H^hom, as read (not symmetrized);
    V_K: (M_hom,)
    """
    # |C*_K,KUBC - C*_K,SUBC|^2 and |C*_K,KUBC|^2, per K
    norm2_diff_K = np.linalg.norm(C_hat_K_kubc - C_hat_K_subc, "fro", axis=(1, 2)) ** 2
    norm2_kubc_K = np.linalg.norm(C_hat_K_kubc, "fro", axis=(1, 2)) ** 2
    # (eq:dC_l2_sum)
    return 100.0 * float(np.sqrt(np.sum(V_K * norm2_diff_K) / np.sum(V_K * norm2_kubc_K)))


def add_tensor_field(grid, name: str, C: np.ndarray):
    # (n, 6, 6) Kelvin matrices as the cell data name, 36 components row-major, named C_11 .. C_66
    grid.cell_data[name] = C.reshape(-1, 36)
    for i in range(6):
        for j in range(6):
            grid.GetCellData().GetArray(name).SetComponentName(6 * i + j, f"C_{i + 1}{j + 1}")


def write_table(path: Path, fields: dict, header: list):
    # one row per macro element, integer columns as integers, the rest as %.6e
    names = list(fields)
    width = max(14, max(len(n) for n in names)) + 2
    is_int = [np.issubdtype(fields[n].dtype, np.integer) for n in names]
    lines = [f"# {h}" for h in header] + ["".join(f"{n:>{width}}" for n in names)]
    for row in zip(*fields.values()):
        lines.append("".join(f"{int(x):>{width}d}" if i else f"{x:>{width}.6e}" for x, i in zip(row, is_int)))
    path.write_text("\n".join(lines) + "\n")


def write_vtm(path: Path, blocks: dict):
    # a ParaView multiblock referencing existing vtu files, {block name: path relative to the vtm}
    datasets = "\n".join(f'    <DataSet index="{i}" name="{name}" file="{file}"/>'
                         for i, (name, file) in enumerate(blocks.items()))
    path.write_text('<?xml version="1.0"?>\n'
                    '<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">\n'
                    f'  <vtkMultiBlockDataSet>\n{datasets}\n  </vtkMultiBlockDataSet>\n</VTKFile>\n')


def row(label: str, cells: list, width: int = 22, label_width: int = 44) -> str:
    return (f"{label:<{label_width}}" + "".join(f"{c:<{width}}" for c in cells)).rstrip()


def stats(label: str, columns: list, masks: list) -> list:
    # min, mean, median, max of each column over its mask
    out = []
    for s, fn in (("min", np.min), ("mean", np.mean), ("median", np.median), ("max", np.max)):
        cells = []
        for c, m in zip(columns, masks):
            x = c[m]
            cells.append(f"{fn(x):.6e}" if len(x) else "-")
        out.append(row(f"{label if s == 'min' else '':<34}{s:>8}", cells))
    return out


def rel(p: Path) -> str:
    return os.path.relpath(p, THESIS_DIR).replace("\\", "/")


def section(title: str) -> list:
    return [RULE, title, RULE]


def summary(cfg_file: Path, run_cfg: dict, files: dict, bcs: list, fields: dict, diff: dict, crit: dict,
            dCL2: float, V_Omega_E: float) -> list:
    geo, mat = run_cfg["geometry"], run_cfg["materials"]
    # the homogenized macro elements (h): all, the matrix ones (no fracture element captured, phi = 0) and the
    # fractured ones (phi > 0)
    h = fields["is_support_tetra"] == 0
    groups = {"all": h, "matrix": h & (fields["phi_macro_el"] == 0.0), "frac": h & (fields["phi_macro_el"] > 0.0)}
    BC = [bc.upper() for bc in bcs]

    lines = [
        *section("Analysis of the effective elastic tensor field"),
        row("date", [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]),
        row("config", [rel(cfg_file)]),
        "",
        "INPUT FILES",
        *[row(label, [rel(p)]) for label, p in files.items()],
        "",
        "PARAMETERS (from the run config, as they enter the script)",
        row("L_inner [m]", [f"{float(geo['L_inner']):g}"]),
        row("L_ext_factor [-]", [f"{float(geo['L_ext_factor']):g}"]),
        row("young_modulus_rock [Pa]", [f"{float(mat['young_modulus_rock']):g}"]),
        row("poisson_ratio_rock [-]", [f"{float(mat['poisson_ratio_rock']):g}"]),
        row("young_modulus_fracture [Pa]", [f"{float(mat['young_modulus_fracture']):g}"]),
        row("poisson_ratio_fracture [-]", [f"{float(mat['poisson_ratio_fracture']):g}"]),
        row("aperture_per_r [-]", [f"{float(mat['aperture_per_r']):g}"]),
        *[row(f"fracture {i}: radius [m], center, normal",
              [f"{float(fr['radius']):g}, {list(fr['center'])}, {list(fr['normal'])}"])
          for i, fr in enumerate(run_cfg["fractures"]["fixed"])],
        "",
        "CRITERIA (from the analysis config)",
        *[row(f"{key} [-]", [f"{crit[key]:g}"]) for key in CRITERIA],
        "",
    ]

    # columns: for each group (all, matrix, frac) each BC type
    col = lambda name: [fields[name][bc] for _ in groups for bc in bcs]
    masks = [m for m in groups.values() for _ in bcs]
    count = lambda c, m: f"{int(c[m].sum())} ({100.0 * c[m].sum() / m.sum():.1f} %)" if m.sum() else "-"

    # symmetry of the raw fields, before the symmetry is enforced
    tol_sym = crit["tol_rel_delta_sym"]
    over, d_max, id_max = [], [], []
    for bc in bcs:
        d = fields["delta_sym_macro_el"][bc]
        i_max = np.flatnonzero(h)[np.argmax(d[h])]
        over.append(d > tol_sym)
        d_max.append(f"{d[i_max]:.6e}")
        id_max.append(f"{int(fields['macro_el_id'][i_max])}")
    sym_row = lambda label, cells: row(label, cells, label_width=56)
    lines += [
        *section("SYMMETRY OF THE RAW TENSOR FIELDS (homogenized macro elements only)"),
        f"delta_sym_macro_el of the raw C_K vs tol_rel_delta_sym = {tol_sym:g}",
        "the symmetry is then enforced in any case, C_K_sym = 1/2 (C_K + C_K^T),",
        "and all further checks use C_K_sym",
        "",
        sym_row("", BC),
        *[sym_row(f"delta_sym_macro_el > tol_rel_delta_sym, {g} elems", [count(o, m) for o in over])
          for g, m in groups.items()],
        sym_row("max delta_sym_macro_el [-]", d_max),
        sym_row("macro_el_id of the max delta_sym_macro_el", id_max),
        "",
    ]

    lines += [
        *section("RESULTS PER MACRO ELEMENT (homogenized macro elements only)"),
        row("", [f"{b}, {g} elems" for g in groups for b in BC]),
        row("macro elements", [f"{int(m.sum())}" for m in masks]),
        "",
        row("is_support_tetra = 1 (not counted)",
            [f"{int((~h).sum())} (always in the mesh, shared by KUBC and SUBC; used by SUBC only)"]),
        "",
        row("is_macro_el_sym = 1", [count(c, m) for c, m in zip(col("is_macro_el_sym"), masks)]),
        "",
        row("is_cond2_of_E_good = 1",
            [count(c, m) for c, m in zip(col("is_cond2_of_E_good"), masks)]),
        "",
        row("is_C_eff_pd = 1", [count(c, m) for c, m in zip(col("is_C_eff_pd"), masks)]),
        "",
        row("is_C_eff_regular = 1", [count(c, m) for c, m in zip(col("is_C_eff_regular"), masks)]),
        "",
        row("is_macro_el_iso = 1", [count(c, m) for c, m in zip(col("is_macro_el_iso"), masks)]),
        "",
        row("is_macro_el_within_hrv = 1", [count(c, m) for c, m in zip(col("is_macro_el_within_hrv"), masks)]),
        "",
    ]
    for name, unit in (("cond2_strain_matrix_E", "-"), ("delta_sym_macro_el", "-"), ("lambda_max_macro_el", "Pa"),
                       ("lambda_min_macro_el", "Pa"), ("lambda_minabs_macro_el", "Pa"), ("delta_iso_macro_el", "-"),
                       ("voigt_margin_macro_el", "-"), ("reuss_margin_macro_el", "-")):
        lines += [*stats(f"{name} [{unit}]", col(name), masks), ""]
    lines += [*stats("phi_macro_el [-]", [fields["phi_macro_el"]] * len(masks), masks)]

    if diff is not None:
        lines += ["", RULE,
                  "Difference between KUBC vs SUBC tensor fields",
                  "computed as ||C^KUBC - C^SUBC||_F / ||C^KUBC||_F,",
                  "which equals rel_diff_C_fields_kubc_subc",
                  "",
                  "The tensors in the fields are NOT symmetrized",
                  RULE,
                  row("", [f"{g} elems" for g in groups]),
                  row("macro elements", [f"{int(m.sum())}" for m in groups.values()]),
                  "",
                  *stats("rel_diff_C_fields_kubc_subc [-]", [diff] * len(groups), list(groups.values())),
                  "",
                  "Relative difference of the fields C*_KUBC and C*_SUBC in the L2 norm on Omega_E,",
                  "the fields are constant on each macro element K:",
                  "delta C_L2 [%] = 100 * ||C*_KUBC - C*_SUBC||_2 / ||C*_KUBC||_2",
                  "               = 100 * sqrt( sum_K V_K |C*_K,KUBC - C*_K,SUBC|^2 / sum_K V_K |C*_K,KUBC|^2 )",
                  "K in T_H^hom (the homogenized macro elements, without the support tetrahedra),",
                  "V_K = int_K dV the volume of K,",
                  "|C*_K| = ||W C*_K W||_F the Frobenius norm of the Kelvin matrix",
                  "",
                  row("M_hom = |T_H^hom|", [f"{int(h.sum())}"]),
                  row("sum_K V_K [m^3]", [f"{V_Omega_E:.10g}"]),
                  row("delta C_L2 [%]", [f"{dCL2:.7f}"])]
    return lines


def run_config(cfg_file: Path):
    # the paths in the config are relative to Masters_thesis
    cfg = yaml.safe_load(Path(cfg_file).read_text(encoding="utf-8"))
    run_dir = THESIS_DIR / cfg["run_dir"]
    out = THESIS_DIR / cfg["out"]
    crit = {key: float(cfg[key]) for key in CRITERIA}
    # the copy of the config that scripts/main.py saved in the run dir
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    geo, mat, macro, loads = (run_cfg["geometry"], run_cfg["materials"], run_cfg["mechanics_macroscale"],
                              run_cfg["loads"])
    assert run_cfg["fractures"].get("fixed"), "the apertures need the radii of the fractures of fractures.fixed"
    mesh_name = run_cfg["mesh"]["mesh_name"]
    bcs = [str(bc) for bc in run_cfg["run"]["bc_types"]]

    # scripts/run_macro.py: the micro problem in micro_macro/micro/, the tensor field in micro_macro/macro/<bc>/
    micro_dir, macro_dir = run_dir / "micro_macro" / "micro", run_dir / "micro_macro" / "macro"
    files = {"run config": run_dir / "config.yaml"}
    for bc in bcs:
        files[f"tensor field, {bc}"] = macro_dir / bc / macro["tensor_field_file"]
        # the symmetrized field of upscale_m (runs before 2026-11-08 have none)
        if macro.get("tensor_field_file_sym") and (macro_dir / bc / macro["tensor_field_file_sym"]).exists():
            files[f"tensor field sym, {bc}"] = macro_dir / bc / macro["tensor_field_file_sym"]
    files["micro mesh"] = micro_dir / f"{mesh_name}_healed.msh"
    files["cross section"] = micro_dir / f"{mesh_name}_cross_section.msh"
    case_files = {bc: [micro_dir / bc / d / loads["output_fields_file"] for d in loads[f"case_dir_names_{bc}"]]
                  for bc in bcs}
    for bc in bcs:
        files.update({f"micro solution, {bc}, {f.parts[-3]}": f for f in case_files[bc]})

    # --- per macro element
    tensor_fields = {bc: TensorField(files[f"tensor field, {bc}"], macro["tensor_field_name_prefix"],
                                     geo["support_region"]) for bc in bcs}
    first = tensor_fields[bcs[0]]
    for bc in bcs[1:]:
        assert first.same_mesh(tensor_fields[bc]), f"{bcs[0]} and {bc} have different macro meshes"
    # the symmetrized field of upscale_m must be the C_K_sym the checks use
    for bc in bcs:
        if f"tensor field sym, {bc}" in files:
            C_sym_file = TensorField(files[f"tensor field sym, {bc}"], macro["tensor_field_name_prefix"],
                                     geo["support_region"]).C_K
            C_K_sym = tensor_fields[bc].C_K_sym
            err = np.max(np.linalg.norm(C_sym_file - C_K_sym, axis=(1, 2)) / np.linalg.norm(C_K_sym, axis=(1, 2)))
            assert err <= TOL_REL_SYM_FILE, f"{bc}: the symmetrized field differs from 1/2 (C_K + C_K^T) by {err:.3e}"
            print(f"symmetrized field of upscale_m, {bc}: 1/2 (C_K + C_K^T) up to {err:.3e}")
    # the micro mesh is common to all BC types
    micro = micro_measures(files["micro mesh"], files["cross section"], geo["bulk_region"],
                           geo["fracture_region_prefix"])
    phi_macro_el = fracture_fraction(first.grid, micro)
    C_m = isotropic_kelvin(float(mat["young_modulus_rock"]), float(mat["poisson_ratio_rock"]))
    C_f = isotropic_kelvin(float(mat["young_modulus_fracture"]), float(mat["poisson_ratio_fracture"]))

    # --- averages of the six micro solutions per BC type over each macro element (the conditioning of the fit)
    fixed = run_cfg["fractures"]["fixed"]
    regions = dict(GmshIO(str(files["micro mesh"])).physical)
    # aperture of the region of fracture i (name ending _<i>): aperture_per_r * its radius (as upscale_m
    # MicroMesh.aperture_by_region_id)
    aperture_by_region_id = {tag: float(mat["aperture_per_r"]) * float(fixed[int(name.rsplit("_", 1)[1])]["radius"])
                             for name, (tag, dim) in regions.items()
                             if dim == 2 and name.startswith(geo["fracture_region_prefix"])}
    E_K = {}
    for bc in bcs:
        print(f"averages of the micro solutions, {bc}")
        E_K[bc] = case_averages(case_files[bc], first.grid, regions[geo["bulk_region"]][0], aperture_by_region_id,
                                mat, loads["stress_field_name"], loads["region_id_field_name"])

    common_fields = {"macro_el_id": first.macro_el_ids,
                     "is_support_tetra": first.is_support_tetra.astype(np.uint8),
                     "phi_macro_el": np.where(first.is_support_tetra, np.nan, phi_macro_el)}
    per_bc = {bc: bc_fields(tensor_fields[bc], phi_macro_el, C_m, C_f, condition_numbers(E_K[bc]), crit)
              for bc in bcs}

    # --- KUBC vs SUBC, both on the same macro mesh (same_mesh above); the support tetrahedra are not homogenized
    diff = None
    if "kubc" in bcs and "subc" in bcs:
        diff = np.where(first.is_support_tetra, np.nan,
                        diff_kubc_subc(tensor_fields["kubc"].C_K, tensor_fields["subc"].C_K))

    # --- output
    out.mkdir(parents=True, exist_ok=True)
    fracture = fracture_grid(files["micro mesh"], geo["fracture_region_prefix"])
    for bc in bcs:
        fields = {**common_fields, **per_bc[bc]}
        bc_dir = out / f"{bc}_analysis"
        vtu_dir = bc_dir / f"tensor_field_analysis_{bc}"
        vtu_dir.mkdir(parents=True, exist_ok=True)
        grid = first.grid.copy()
        for name, values in fields.items():
            grid.cell_data[name] = values
        add_tensor_field(grid, f"tensor_C_field_{bc}", tensor_fields[bc].C_K)
        add_tensor_field(grid, f"tensor_C_field_{bc}_sym", tensor_fields[bc].C_K_sym)
        grid.save(vtu_dir / f"tensor_field_{bc}.vtu")
        non_iso = (fields["is_macro_el_iso"] == 0) & (fields["is_support_tetra"] == 0)
        grid.extract_cells(np.flatnonzero(non_iso)).save(vtu_dir / f"non_isotropic_{bc}.vtu")
        fracture.save(vtu_dir / f"all_fracture_elements_{bc}.vtu")
        write_vtm(bc_dir / f"tensor_field_analysis_{bc}.vtm",
                  {name: f"{vtu_dir.name}/{name}_{bc}.vtu"
                   for name in ("tensor_field", "non_isotropic", "all_fracture_elements")})
        write_table(bc_dir / f"per_element_{bc}.txt", fields,
                    [f"{bc}, run_dir {cfg['run_dir']}, " + ", ".join(f"{key} {crit[key]:g}" for key in CRITERIA),
                     "is_support_tetra = 1: support tetrahedron (used by SUBC only), not homogenized"])

    if diff is not None:
        ids = {"macro_el_id": common_fields["macro_el_id"],
               "is_support_tetra": common_fields["is_support_tetra"]}
        diff_fields = {**ids, "rel_diff_C_fields_kubc_subc": diff}
        # the tensors C_K as read, (n, 6, 6) -> 36 components row-major
        C_kubc, C_subc = tensor_fields["kubc"].C_K, tensor_fields["subc"].C_K
        tensors = {"tensor_C_field_kubc": C_kubc, "tensor_C_field_subc": C_subc}
        grid = first.grid.copy()
        for name, values in diff_fields.items():
            grid.cell_data[name] = values
        for name, C in tensors.items():
            add_tensor_field(grid, name, C)
        grid.save(out / "diff_kubc_subc.vtu")
        write_table(out / "diff_kubc_subc.txt", diff_fields,
                    [f"||C_K^KUBC - C_K^SUBC||_F / ||C_K^KUBC||_F, tensors as read, run_dir {cfg['run_dir']}",
                     "is_support_tetra = 1: support tetrahedron (used by SUBC only), not homogenized (nan)"])

    # the common fields as arrays, the per BC ones as {bc: array}
    fields_by_name = {**common_fields, **{name: {bc: per_bc[bc][name] for bc in bcs} for name in per_bc[bcs[0]]}}
    # --- delta C_L2 over T_H^hom (the support tetrahedra are not homogenized), (eq:dC_l2_sum) in dC_L2
    dCL2 = V_Omega_E = None
    if diff is not None:
        hom = ~first.is_support_tetra
        # V_K = int_K dV, the volume of the macro element K
        V_K = np.abs(first.grid.compute_cell_sizes(length=False, area=False, volume=True,
                                                   vertex_count=False).cell_data["Volume"])[hom]
        dCL2 = dC_L2(tensor_fields["kubc"].C_K[hom], tensor_fields["subc"].C_K[hom], V_K)
        V_Omega_E = float(np.sum(V_K))
    lines = summary(cfg_file, run_cfg, files, bcs, fields_by_name, diff, crit, dCL2, V_Omega_E)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="configs/analyse_tensor_field_single_fracture.yaml")
    run_config(ap.parse_args(argv).config)


if __name__ == "__main__":
    sys.exit(main())
