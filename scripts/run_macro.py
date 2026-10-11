"""
Homogenized (macro) model from Masters_thesis: the micro problem on the buffered fractured mesh, the
homogenization (effective elastic tensor per macro element) and the macro problem (upscale_m micro_mesh,
bc, macro_mechanics_model), for the BC types and load cases chosen in the section run of the config.

    python scripts/run_macro.py configs/config_test.yaml

The micro problem always solves all 6 load cases of a BC type, the 6x6 tensor fit needs them; run.cases
chooses the load cases of the macro problem.

Results: <run.output_dir>/<run.name>/macro/, the micro and macro meshes, <bc_type>/<case>/ (micro problem),
<bc_type>/macro_elastic_tensor.msh2 and macro_<bc_type>/<case>/ with the Flow123d output and macro_fields.vtu.
"""
import sys
from pathlib import Path

from common import bc_types, load_config, prepare_run_dir, selected_cases
from endorse.common import dotdict, workdir
from bc import run_bc
from export_vtu import export_macro_case
from macro_mechanics_model import macro_elastic_tensor, macro_problem, make_macro_mesh
from micro_mesh import make_micro_mesh


def run_macro(cfg: dotdict, run_dir: Path):
    types = bc_types(cfg)
    # SUBC needs the rigid-body support tetrahedra built into both meshes
    subc_support = "subc" in types
    with workdir(run_dir / "macro"):
        print("=== micro and macro mesh ===", flush=True)
        micro = make_micro_mesh(cfg.geometry, cfg.mesh, cfg.fractures, cfg.materials.aperture_per_r,
                                work_dir=".", subc_support=subc_support)
        macro_mesh = make_macro_mesh(cfg.geometry, cfg.mechanics_macroscale, cfg.mesh,
                                     work_dir=".", subc_support=subc_support)
        macro_mesh_path = str(cfg.mechanics_macroscale.mesh_file)
        for bc_type in types:
            print(f"=== micro problem: {bc_type.upper()}, all 6 load cases ===", flush=True)
            micro_results = run_bc(cfg, micro, bc_type)
            tensor_file = macro_elastic_tensor(cfg, micro, macro_mesh, micro_results, bc_type)
            for name, dir_name, load in selected_cases(cfg, bc_type):
                print(f"=== macro problem: {bc_type.upper()} case {name} ===", flush=True)
                spatial_file = macro_problem(cfg, f"macro_{bc_type}/{dir_name}", load, macro_mesh,
                                             macro_mesh_path, tensor_file, bc_type)
                # macro_<bc_type>/<case>/output/mechanics_fields.msh -> macro_<bc_type>/<case>/macro_fields.vtu
                export_macro_case(cfg, spatial_file, str(Path(spatial_file).parents[1]))


def main(config_file: str):
    config_file = Path(config_file)
    cfg = load_config(config_file)
    run_dir = prepare_run_dir(cfg, config_file)
    print(f"[run_macro] config {config_file}, results {run_dir / 'macro'}", flush=True)
    run_macro(cfg, run_dir)
    print("[run_macro] done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
