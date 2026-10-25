"""
Relative distance delta_iso of every effective elastic tensor of the macro field from the class of
isotropic tensors.

Theory:
article from M. Moakher, A. N. Norris.
With C the symmetric 6x6 Kelvin matrix of an element:
        v = 1/sqrt(3) (1, 1, 1, 0, 0, 0)^T,    J = v v^T,    L = I_6 - J
        C_iso = 3 kappa J + 2 mu L,    kappa = tr(J C) / 3,    mu = tr(L C) / 10
        delta_iso = ||C - C_iso||_F / ||A||_F

Input is the anisotropic field that upscale_m writes for Flow123d (macro_elastic_tensor.msh2), i.e. per
macro element six 3x3 tensors stiffness_tensor_0 .. stiffness_tensor_5, the k-th being the k-th column of
the 6x6 Kelvin tensor, which is put into a symmetric 3x3 (upscale_m macro_mechanics_model.kelvin_columns_3x3),
ordering [xx, yy, zz, yz, xz, xy].

Output per field: delta_iso_per_element.txt (delta_iso of every macro element) and
summary_all_delta_iso.txt (the whole field)

"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from bgem.gmsh.gmsh_io import GmshIO


v = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / np.sqrt(3.0)
J = np.outer(v, v)
L = np.eye(6) - J


def distance_from_isotropy(C_K: np.ndarray) -> float:
    # one effective tensor (6, 6), Kelvin -> delta_iso
    # symmetrized, we need the tensor C to be symmetric
    C = 0.5 * (C_K + C_K.T)
    kappa = np.trace(J @ C) / 3.0
    mu = np.trace(L @ C) / 10.0
    C_iso = 3.0 * kappa * J + 2.0 * mu * L
    delta_iso = np.linalg.norm(C - C_iso, "fro") / np.linalg.norm(C, "fro")
    return delta_iso



def kelvin_from_columns(cols: np.ndarray) -> np.ndarray:
    # (n, 6, 3, 3) stiffness_tensor_k -> (n, 6, 6) Kelvin; inverse of kelvin_columns_3x3:
    # column k = [A_xx, A_yy, A_zz, sqrt2 A_yz, sqrt2 A_xz, sqrt2 A_xy] of the k-th 3x3
    s = np.sqrt(2.0)
    A = cols
    columns = np.stack([A[..., 0, 0], A[..., 1, 1], A[..., 2, 2],
                        s * A[..., 1, 2], s * A[..., 0, 2], s * A[..., 0, 1]], axis=-1)
    return np.swapaxes(columns, -1, -2)


def read_part_constant_tensor_field(msh2: Path, prefix: str):
    # the tetrahedra of the macro mesh (the boundary triangles carry zero tensors and are dropped)
    gmsh_io = GmshIO(str(msh2))
    el_ids = np.array(sorted(gmsh_io.elements))
    # gmsh element type of a linear tetrahedron, GMSH_TETRA = 4
    tetra = np.array([gmsh_io.elements[i][0] == 4 for i in el_ids])
    cols = []
    for k in range(6):
        item = gmsh_io.element_data[f"{prefix}{k}"][0]
        assert np.array_equal(np.asarray(item.tags), el_ids), f"{prefix}{k}: element tags differ from the mesh"
        cols.append(np.asarray(item.values, dtype=float).reshape(-1, 3, 3))
    return el_ids[tetra], np.stack(cols, axis=1)[tetra]


def analyse(msh2: Path, prefix: str, tol_rel: float, out_dir: Path):
    el_ids, cols = read_part_constant_tensor_field(msh2, prefix)
    C_kelvin = kelvin_from_columns(cols)

    delta_iso = np.array([distance_from_isotropy(C_K) for C_K in C_kelvin])
    # isotropic: delta_iso <= tol_rel
    is_iso = delta_iso <= tol_rel

    out_dir.mkdir(parents=True, exist_ok=True)
    # from 0 to the highest delta_iso, equal values stay in the element_id order
    order = np.argsort(delta_iso, kind="stable")
    np.savetxt(out_dir / "delta_iso_per_element.txt", np.column_stack([el_ids[order], delta_iso[order]]),
               fmt=["%d", "%.10e"], header=f"{msh2}\nelement_id  delta_iso [-]")

    n, n_iso = len(el_ids), int(is_iso.sum())
    aniso = delta_iso[~is_iso]
    stats = lambda x: f"min {x.min():.6e}   median {np.median(x):.6e}   max {x.max():.6e}" if len(x) else "-"
    lines = [
        "Relative distance from isotropy = delta_iso",
        f"file                           {msh2}",
        "",
        f"isotropy_tol                   {tol_rel}",
        f"macro elements                 {n}",
        f"isotropic (delta_iso <= isotropy_tol)                     {n_iso}   ({100.0 * n_iso / n:.1f} %) ",
        f"not isotropic                  {n - n_iso}   ({100.0 * (n - n_iso) / n:.1f} %)",
        "",
        f"delta_iso [-], all             {stats(delta_iso)}",
        f"delta_iso [-], not isotropic   {stats(aniso)}",
    ]
    (out_dir / "summary_all_delta_iso.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run_config(config_file: Path):
    # the paths in the config are relative to Masters_thesis
    cfg = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    thesis_dir = Path(__file__).resolve().parents[1]
    for label, msh2 in cfg["fields"].items():
        print(f"\n{label}")
        analyse(thesis_dir / msh2, str(cfg["field_prefix"]), float(cfg["tol_rel"]), thesis_dir / cfg["out"] / label)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="configs/distance_isotropy_of_tensor_field.yaml")
    run_config(ap.parse_args(argv).config)


if __name__ == "__main__":
    sys.exit(main())
