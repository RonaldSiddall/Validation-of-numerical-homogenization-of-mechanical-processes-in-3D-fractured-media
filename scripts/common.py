"""
Shared helpers of the scripts that run the simulations from Masters_thesis (run_dns.py, run_macro.py).

The computational core is upscale_m in the endorse submodule (code/endorse, branch thesis_branch); these
scripts only read a config from configs/, choose what to run (its section run) and call the upscale_m
functions. The results go to <run.output_dir>/<run.name>/, together with a copy of the config.
"""
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

# paths given by the layout of the repository, not settings
THESIS_DIR = Path(__file__).resolve().parents[1]
UPSCALE_M_SRC = THESIS_DIR / "code" / "endorse" / "apps" / "upscale_m" / "src"
# the upscale_m modules import each other as top-level modules (bc, micro_mesh, ...)
sys.path.insert(0, str(UPSCALE_M_SRC))

from endorse import common
from endorse.common import dotdict
from bc import load_cases


def load_config(config_file: Path) -> dotdict:
    return common.load_config(Path(config_file).resolve())


def known_bc_types(cfg: dotdict) -> List[str]:
    # the BC types the config defines load cases for (loads.case_names_<type>)
    return [key[len("case_names_"):] for key in cfg.loads if key.startswith("case_names_")]


def bc_types(cfg: dotdict) -> List[str]:
    known = known_bc_types(cfg)
    types = [str(t).strip().lower() for t in cfg.run.bc_types]
    assert types and set(types) <= set(known), f"run.bc_types must be a subset of {known} (got {types})"
    return [t for t in known if t in types]


def selected_cases(cfg: dotdict, bc_type: str) -> List[Tuple[str, str, np.ndarray]]:
    # the load cases of one BC type that run.cases asks for: all, or a list of case names (E_11, S_12, ...)
    cases = load_cases(cfg, bc_type)
    if str(cfg.run.cases).strip().lower() == "all":
        return cases
    wanted = {str(c) for c in cfg.run.cases}
    known = {name for t in known_bc_types(cfg) for name, _dir, _load in load_cases(cfg, t)}
    assert wanted <= known, f"unknown case(s) in run.cases: {sorted(wanted - known)}"
    return [c for c in cases if c[0] in wanted]


def _run_identity(config_file: Path) -> dict:
    # the config as plain data without what does not change the results already in a run dir: the run
    # keys that only choose what to run (dns, macro, cases) and machine_config (how Flow123d is started)
    data = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    data["run"] = {k: v for k, v in data["run"].items() if k not in ("dns", "macro", "cases")}
    data.pop("machine_config", None)
    return data


def prepare_run_dir(cfg: dotdict, config_file: Path) -> Path:
    # <run.output_dir>/<run.name>/ (output_dir relative to Masters_thesis) with a copy of the config.
    # upscale_m reuses meshes and Flow123d results it finds in the run dir without checking the config,
    # so a run dir made with different parameters is refused.
    run_dir = THESIS_DIR / str(cfg.run.output_dir) / str(cfg.run.name)
    saved = run_dir / "config.yaml"
    if saved.is_file() and _run_identity(saved) != _run_identity(config_file):
        raise SystemExit(f"{run_dir} was created with different parameters than {config_file}:\n"
                         f"change run.name, or delete {run_dir} to recompute it")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_file, saved)
    return run_dir
