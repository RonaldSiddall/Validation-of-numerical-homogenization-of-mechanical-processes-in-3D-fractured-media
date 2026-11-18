"""
Error of the homogenized macro solution against the DNS reference, PyVista.

Each field is compared where Flow123d computes it, by a 1:1 copy of the ParaView pipeline used so far.

Output per load case: eff_vs_detailed_error.vtu (the DNS mesh with stress_error, displacement_error and the
other error fields) and eff_vs_detailed_error.txt (the errors and statistics); all load cases of a config also
give summary.csv.

Usage:
    python scripts/macro_dns_error.py configs/compare_summer.yaml       (all load cases, see the config)
    python scripts/macro_dns_error.py DNS.vtu MACRO.vtu --out DIR        (one DNS / macro pair)
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pyvista as pv
import yaml

THRESHOLDS = (1.0, 2.0, 5.0, 10.0)
# (i, j) of the six independent components of the row-major 3x3 stress
STRESS_IJ = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def stress_norm2(s9: np.ndarray) -> np.ndarray:
    # s_xx^2 + s_yy^2 + s_zz^2 + s_yz^2 + s_xz^2 + s_xy^2, as in the ParaView expressions
    s = np.asarray(s9).reshape(-1, 3, 3)
    return sum(s[:, i, j] ** 2 for i, j in STRESS_IJ)


# ============================================================================ readers, common filters

def xml_unstructured_grid_reader(path: Path, cell_arrays: Dict[str, str],
                                 point_arrays: Dict[str, str] = None) -> pv.UnstructuredGrid:
    # CellArrayStatus / PointArrayStatus: only the selected arrays, {name in file: name in pipeline}
    grid = pv.read(path)
    out = grid.copy(deep=False)
    out.clear_data()
    for name, new_name in cell_arrays.items():
        out.cell_data[new_name] = grid.cell_data[name]
    for name, new_name in (point_arrays or {}).items():
        out.point_data[new_name] = grid.point_data[name]
    return out


def threshold_vtk_valid(data) -> pv.UnstructuredGrid:
    # Threshold: Scalars = vtkValidPointMask, Lower = Upper = 1, All Scalars = 1 (ParaView default)
    return data.threshold((1.0, 1.0), scalars="vtkValidPointMask", preference="point", all_scalars=True)


def compute_quartiles(values: np.ndarray) -> dict:
    # as vtkComputeQuartiles (vtkOrderStatistics, InverseCDFAveragedSteps): the k-th quartile is the
    # mean of the sorted values of (1-based) rank round(k n / 4) and floor(k n / 4 + 1)
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    q = [x[0]]
    for k in (1, 2, 3):
        h = k * n / 4.0
        i1, i2 = max(int(np.floor(h + 0.5)), 1), min(int(np.floor(h + 1.0)), n)
        q.append(0.5 * (x[i1 - 1] + x[i2 - 1]))
    q.append(x[-1])
    return dict(zip(("min", "q1", "median", "q3", "max"), map(float, q)))


def descriptive_statistics(values: np.ndarray) -> dict:
    # as vtkDescriptiveStatistics with the ParaView default Task 'Detailed model of input data' (all
    # the samples): M2..M4 the central sums, the unbiased sample variance, skewness and (excess)
    # kurtosis the sample estimators G1 and G2
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    m2, m3, m4 = (float(np.sum((x - mean) ** k)) for k in (2, 3, 4))
    variance = m2 / (n - 1)
    g1, g2 = np.sqrt(n) * m3 / m2 ** 1.5, n * m4 / m2 ** 2 - 3.0
    return {"Cardinality": n, "Minimum": float(x.min()), "Maximum": float(x.max()), "Mean": mean,
            "M2": m2, "M3": m3, "M4": m4, "Standard Deviation": variance ** 0.5, "Variance": variance,
            "Skewness": g1 * np.sqrt(n * (n - 1)) / (n - 2),
            "Kurtosis": ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3)), "Sum": float(x.sum())}


# ============================================================================ stress: method A

def clean_to_grid(macro: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    # Flow123d writes every element with its own nodes; merging them lets the probe find the cell
    # of a point lying on a face between two macro cells
    return macro.clean(tolerance=0.0)


def cell_size(dns: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    # only Volume (ComputeVertexCount = ComputeLength = ComputeArea = 0): the triangles get Volume 0
    return dns.compute_cell_sizes(length=False, area=False, volume=True, vertex_count=False)


def cell_centers(dns: pv.UnstructuredGrid) -> pv.PolyData:
    # VertexCells = 1, CopyArrays = 1; the cell arrays become point arrays
    return dns.cell_centers(vertex=True)


def resample_with_dataset(source: pv.UnstructuredGrid, destination: pv.PolyData) -> pv.PolyData:
    # Source Data Arrays = CleantoGrid1, Destination Mesh = CellCenters1, Pass Point Arrays = 1,
    # Compute Tolerance = 1, Static Cell Locator; adds vtkValidPointMask
    return destination.sample(source, pass_point_data=True, pass_cell_data=False, pass_field_data=False,
                              locator="static_cell")


def weights(p) -> np.ndarray:
    # abs(Volume): the fracture triangles have Volume 0, so they weigh 0
    return np.abs(p["Volume"])


def volume_total(p) -> float:
    return float(np.sum(p["weights"]))


def ds2(p) -> np.ndarray:
    return stress_norm2(p["stress"] - p["stress_macro"])


def s2(p) -> np.ndarray:
    return stress_norm2(p["stress"])


def ds_percent(p) -> np.ndarray:
    return 100.0 * np.sqrt(p["ds2"]) / np.max(np.sqrt(p["s2"]))


def ds_l2(p) -> float:
    return 100.0 * float(np.sqrt(np.sum(p["weights"] * p["ds2"]) / np.sum(p["weights"] * p["s2"])))


def n_below(p, upper: float) -> int:
    # Threshold N<upper>: ds_percent between its minimum and upper, both inclusive
    return int(np.sum((p["ds_percent"] >= np.min(p["ds_percent"])) & (p["ds_percent"] <= upper)))


def v_below(p, upper: float, total: float) -> float:
    # Threshold N<upper> followed by the Python Calculator V<upper> = 100 * sum(weights) / Volume_total
    n = (p["ds_percent"] >= np.min(p["ds_percent"])) & (p["ds_percent"] <= upper)
    return 100.0 * float(np.sum(p["weights"][n])) / total


def stress_pipeline(dns: pv.UnstructuredGrid, macro: pv.UnstructuredGrid):
    # The method A state top to bottom. Returns the valid DNS centroids (point arrays as in ParaView)
    # and the scalar results.
    points = threshold_vtk_valid(resample_with_dataset(clean_to_grid(macro), cell_centers(cell_size(dns))))
    p = points.point_data
    p["weights"] = weights(p)
    total = volume_total(p)
    p["ds2"] = ds2(p)
    p["s2"] = s2(p)
    p["ds_percent"] = ds_percent(p)
    results = dict(points_valid=points.n_points, points_invalid=dns.n_cells - points.n_points,
                   Volume_total=total, dsL2=ds_l2(p))
    results.update({f"N{t:g}": n_below(p, t) for t in THRESHOLDS})
    results.update({f"V{t:g}": v_below(p, t, total) for t in THRESHOLDS})
    results["quartiles"] = compute_quartiles(p["ds_percent"])
    results["descriptive_statistics"] = descriptive_statistics(p["ds_percent"])
    return points, results


# ============================================================================ displacement: week 9 (pvsm)

def resample(source: pv.UnstructuredGrid, destination: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    # Resample: Source Data Arrays = Macro, Destination Mesh = DNS, Pass Point Arrays = Pass Cell
    # Arrays = Pass Field Arrays = 1, Compute Tolerance = 1, Mark Blank Points And Cells = 0,
    # Static Cell Locator; the macro mesh is not merged (as in the pvsm)
    return destination.sample(source, pass_point_data=True, pass_cell_data=True, pass_field_data=True,
                              mark_blank=False, locator="static_cell")


def du_percent(p) -> np.ndarray:
    # DisplacementDifference: 100*mag(displacement - displacement_macro)/max(mag(displacement))
    return 100.0 * np.linalg.norm(p["displacement"] - p["displacement_macro"], axis=1) \
        / np.max(np.linalg.norm(p["displacement"], axis=1))


def du2(p) -> np.ndarray:
    # DisplacementSquared: mag(displacement - displacement_macro)**2
    return np.linalg.norm(p["displacement"] - p["displacement_macro"], axis=1) ** 2


def u2(p) -> np.ndarray:
    # ReferenceDisplacementSquared: mag(displacement)**2
    return np.linalg.norm(p["displacement"], axis=1) ** 2


def integrate_variables(grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    # Integrate (Integrate Variables, Divide Cell Data By Volume = 0): point data integrated by the
    # linear interpolant over the cells of the highest dimension (tetrahedra); the triangles do not count
    return grid.integrate_data()


def du_l2(integrated) -> float:
    # L2Displacement: 100*sqrt(du2/u2)
    return 100.0 * float(np.sqrt(integrated.point_data["du2"][0] / integrated.point_data["u2"][0]))


def m_below(values: np.ndarray, upper: float) -> int:
    # number of the nodes with du_percent <= upper
    return int(np.sum(values <= upper))


def p_below(values: np.ndarray, upper: float) -> float:
    # share of the nodes with du_percent <= upper (nodes, not cells: no Threshold, which keeps cells)
    return 100.0 * float(np.mean(values <= upper))


def displacement_pipeline(dns: pv.UnstructuredGrid, macro: pv.UnstructuredGrid):
    # The pvsm top to bottom. Returns the valid DNS nodes (ValidPoints) and the scalar results.
    resampled = resample(macro, dns)
    valid = threshold_vtk_valid(resampled)
    p = valid.point_data
    p["du_percent"] = du_percent(p)
    p["du2"] = du2(p)
    p["u2"] = u2(p)
    integrated = integrate_variables(valid)
    results = dict(nodes=dns.n_points,
                   # nodes found in the macro mesh, vtkValidPointMask = 1
                   nodes_mask_valid=int(np.sum(np.asarray(resampled.point_data["vtkValidPointMask"]) == 1)),
                   # nodes of the cells ValidPoints keeps (all their nodes valid); the statistics use these
                   nodes_valid=valid.n_points, nodes_dropped=dns.n_points - valid.n_points,
                   u_max_dns=float(np.max(np.linalg.norm(p["displacement"], axis=1))),
                   int_du2=float(integrated.point_data["du2"][0]), int_u2=float(integrated.point_data["u2"][0]),
                   duL2=du_l2(integrated))
    results.update({f"M{t:g}": m_below(p["du_percent"], t) for t in THRESHOLDS})
    results.update({f"P{t:g}": p_below(p["du_percent"], t) for t in THRESHOLDS})
    results["quartiles"] = compute_quartiles(p["du_percent"])
    results["descriptive_statistics"] = descriptive_statistics(p["du_percent"])
    return valid, results


# ============================================================================ one DNS / macro pair

def compare(dns_vtu: Path, macro_vtu: Path, out_dir: Path, dns_stress="stress", macro_stress="stress",
            dns_disp="displacement", macro_disp="displacement") -> dict:
    dns = xml_unstructured_grid_reader(dns_vtu, {dns_stress: "stress"}, {dns_disp: "displacement"})
    macro = xml_unstructured_grid_reader(macro_vtu, {macro_stress: "stress_macro"},
                                         {macro_disp: "displacement_macro"})
    # not in ParaView: the cell and node indices, carried along to map the results back to the DNS mesh
    dns.cell_data["dns_cell"] = np.arange(dns.n_cells)
    dns.point_data["dns_point"] = np.arange(dns.n_points)

    # ---- stress, method A (centroids)
    points, stress = stress_pipeline(dns, macro)
    p = points.point_data

    # ---- displacement, week 9 (nodes)
    nodes, displacement = displacement_pipeline(dns, macro)
    q = nodes.point_data

    results = dict(dns=str(dns_vtu), macro=str(macro_vtu), cells=dns.n_cells, stress=stress,
                   displacement=displacement)

    # ---- output: the DNS mesh with stress, displacement and the error fields, NaN where not compared
    out = dns.copy()
    del out.cell_data["dns_cell"], out.point_data["dns_point"]
    dns_cell = np.asarray(p["dns_cell"]).astype(int)
    for name, values in dict(weights=p["weights"], stress_macro=p["stress_macro"],
                             stress_error=np.asarray(p["stress"]) - np.asarray(p["stress_macro"]),
                             ds_percent=p["ds_percent"]).items():
        values = np.asarray(values, dtype=float)
        out.cell_data[name] = np.full((dns.n_cells,) + values.shape[1:], np.nan)
        out.cell_data[name][dns_cell] = values
    dns_point = np.asarray(q["dns_point"]).astype(int)
    for name, values in dict(displacement_macro=q["displacement_macro"],
                             displacement_error=np.asarray(q["displacement"]) - np.asarray(q["displacement_macro"]),
                             du_percent=q["du_percent"]).items():
        values = np.asarray(values, dtype=float)
        out.point_data[name] = np.full((dns.n_points,) + values.shape[1:], np.nan)
        out.point_data[name][dns_point] = values

    out_dir.mkdir(parents=True, exist_ok=True)
    out.save(out_dir / "eff_vs_detailed_error.vtu")
    (out_dir / "eff_vs_detailed_error.txt").write_text(report(results))
    return results



RULE = "=" * 64
THESIS_DIR = Path(__file__).resolve().parents[1]


LABEL = 72


def rel_path(p) -> str:
    # a path relative to Masters_thesis (as given if outside it)
    p = Path(p).resolve()
    return p.relative_to(THESIS_DIR).as_posix() if p.is_relative_to(THESIS_DIR) else str(p)


def distribution(q: dict, d: dict) -> list:
    # the distribution of an error over the cells or nodes (each counts once, not weighted by its size)
    rows = [("n", f"{d['Cardinality']}"), ("min", q["min"]), ("q1", q["q1"]), ("median", q["median"]),
            ("q3", q["q3"]), ("max", q["max"]), ("mean", d["Mean"]), ("standard deviation", d["Standard Deviation"]),
            ("skewness", d["Skewness"]), ("kurtosis", d["Kurtosis"])]
    return [f"  {k:<{LABEL - 2}}{v if isinstance(v, str) else f'{v:.6g}'}" for k, v in rows]


def report(r: dict) -> str:
    # the effective model = the former macro problem, the detailed model = the former DNS (Stebel's terminology)
    s, u = r["stress"], r["displacement"]
    row = lambda label, value: f"{label:<{LABEL}}{value}"
    lines = [
        RULE, "MODEL COMPARISON: Effective vs. Detailed (reference: Detailed model)",
        "METRIC: Stress and Displacement Error", RULE,
        row("Detailed model input", rel_path(r["dns"])),
        row("Effective model input", rel_path(r["macro"])),
        "",
        RULE, "STRESS COMPARISON: Done in element centroids",
        "-> Data evaluated strictly as CellData",
        "-> No interpolation to PointData (no nodal averaging)", RULE,
        row("Detailed model cells compared (centroid in the effective model mesh)",
            f"{s['points_valid']} of {r['cells']}"),
        row("volume of the compared Detailed model cells [m^3]", f"{s['Volume_total']:.10g}"),
        row("dsL2 [%]", f"{s['dsL2']:.7f}"),
        "",
        *(row(f"cells with ds_percent <= {t:g} %", f"{s[f'N{t:g}']:>8d} cells, {s[f'V{t:g}']:8.4f} % of the volume")
          for t in THRESHOLDS),
        "",
        "ds_percent [%] over the Detailed model cells (each cell counts once)",
        *distribution(s["quartiles"], s["descriptive_statistics"]),
        "",
        RULE, "DISPLACEMENT COMPARISON: Done in element nodes",
        "-> Data evaluated strictly as PointData (the nodes of the Detailed model elements)",
        "-> Effective model displacement interpolated linearly inside its elements to these nodes", RULE,
        row("Detailed model nodes compared (in the effective model mesh)", f"{u['nodes_valid']} of {u['nodes']}"),
        row("duL2 [%]", f"{u['duL2']:.7f}"),
        "",
        *(row(f"nodes with du_percent <= {t:g} %", f"{u[f'M{t:g}']:>8d} nodes, {u[f'P{t:g}']:8.4f} % of the nodes")
          for t in THRESHOLDS),
        "",
        "du_percent [%] over the Detailed model nodes (each node counts once)",
        *distribution(u["quartiles"], u["descriptive_statistics"]),
    ]
    return "\n".join(lines) + "\n"


# ============================================================================ all load cases

def run_batch(dns_run: Path, macro_run: Path, out_root: Path, macro_case_dir: str = "{bc}", **kw):
    # macro_case_dir: the folder of the macro load cases of one BC type inside macro_run, {bc} = kubc / subc
    rows = []
    for bc in ("kubc", "subc"):
        for case_dir in sorted(p for p in (dns_run / bc).glob("*") if p.is_dir()):
            dns_vtu = case_dir / "dns_full.vtu"
            macro_vtu = macro_run / macro_case_dir.format(bc=bc) / case_dir.name / "macro_fields.vtu"
            if not (dns_vtu.is_file() and macro_vtu.is_file()):
                print(f"skip {bc}/{case_dir.name}: missing {dns_vtu if not dns_vtu.is_file() else macro_vtu}")
                continue
            r = compare(dns_vtu, macro_vtu, out_root / bc / case_dir.name, **kw)
            s, u = r["stress"], r["displacement"]
            print(f"{bc}/{case_dir.name}: dsL2 {s['dsL2']:.4f} %, duL2 {u['duL2']:.4f} %")
            rows.append(dict(case=f"{bc}/{case_dir.name}", dsL2=s["dsL2"], duL2=u["duL2"],
                             **{f"V{t:g}": s[f"V{t:g}"] for t in THRESHOLDS},
                             **{f"P{t:g}": u[f"P{t:g}"] for t in THRESHOLDS},
                             ds_median=s["quartiles"]["median"], ds_q3=s["quartiles"]["q3"],
                             du_median=u["quartiles"]["median"], du_q3=u["quartiles"]["q3"]))
    if rows:
        with open(out_root / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsummary of {len(rows)} case(s): {out_root / 'summary.csv'}")


def run_config(config_file: Path):
    # all load cases of the DNS and macro results named in the config (configs/compare_*.yaml);
    # the paths in the config are relative to Masters_thesis
    cfg = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    thesis_dir = Path(__file__).resolve().parents[1]
    fields = cfg["fields"]
    run_batch(thesis_dir / cfg["dns_run"], thesis_dir / cfg["macro_run"], thesis_dir / cfg["out"],
              macro_case_dir=str(cfg.get("macro_case_dir", "{bc}")),
              dns_stress=fields["dns_stress"], macro_stress=fields["macro_stress"],
              dns_disp=fields["dns_displacement"], macro_disp=fields["macro_displacement"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path, nargs="+", help="configs/compare_*.yaml, or DNS.vtu MACRO.vtu")
    ap.add_argument("--out", type=Path, help="output folder, only for a DNS / macro pair")
    a = ap.parse_args(argv)
    if len(a.inputs) == 1:
        run_config(a.inputs[0])
    elif len(a.inputs) == 2 and a.out is not None:
        print(report(compare(a.inputs[0], a.inputs[1], a.out)), end="")
    else:
        ap.error("give a config, or DNS.vtu MACRO.vtu --out DIR")


if __name__ == "__main__":
    sys.exit(main())
