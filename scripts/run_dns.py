"""
DNS from Masters_thesis: the cube solved directly on the explicitly fractured mesh (upscale_m
dns_mechanics_model), for the BC types and load cases chosen in the section run of the config.

    python scripts/run_dns.py configs/config_test.yaml

Results: <run.output_dir>/<run.name>/dns/, the DNS mesh and <bc_type>/<case>/ with the Flow123d output
and dns_full.vtu.
"""
import sys
from pathlib import Path

from common import bc_types, load_config, prepare_run_dir, selected_cases
from endorse.common import dotdict, workdir
from bc import micro_problem
from dns_mechanics_model import make_dns_mesh
from export_vtu import export_dns_case


def run_dns(cfg: dotdict, run_dir: Path):
    types = bc_types(cfg)
    with workdir(run_dir / "dns"):
        print("=== DNS: mesh ===", flush=True)
        # SUBC needs the rigid-body support tetrahedra built into the mesh
        mesh = make_dns_mesh(cfg, work_dir=".", subc_support="subc" in types)
        for bc_type in types:
            for name, dir_name, load in selected_cases(cfg, bc_type):
                print(f"=== DNS: {bc_type.upper()} case {name} ===", flush=True)
                spatial_file = micro_problem(cfg, f"{bc_type}/{dir_name}", load, mesh, bc_type)
                # <bc_type>/<case>/output/mechanics_fields.msh -> <bc_type>/<case>/dns_full.vtu
                export_dns_case(cfg, spatial_file, str(Path(spatial_file).parents[1]))


def main(config_file: str):
    config_file = Path(config_file)
    cfg = load_config(config_file)
    run_dir = prepare_run_dir(cfg, config_file)
    print(f"[run_dns] config {config_file}, results {run_dir / 'dns'}", flush=True)
    run_dns(cfg, run_dir)
    print("[run_dns] done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
