"""
Runs the simulations chosen in the section run of the config: the DNS (run.dns) and the homogenized
model, i.e. the micro problem, the homogenization and the macro problem (run.macro).

    python scripts/main.py configs/config_test.yaml

Results: <run.output_dir>/<run.name>/ with a copy of the config, dns/ and macro/.
"""
import sys
from pathlib import Path

from common import load_config, prepare_run_dir
from run_dns import run_dns
from run_macro import run_macro


def main(config_file: str):
    config_file = Path(config_file)
    cfg = load_config(config_file)
    run_dir = prepare_run_dir(cfg, config_file)
    print(f"[main] config {config_file}, results {run_dir}", flush=True)
    if cfg.run.dns:
        run_dns(cfg, run_dir)
    if cfg.run.macro:
        run_macro(cfg, run_dir)
    print("[main] done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
