# Validation of numerical homogenization of mechanical processes in 3D fractured media

Master's thesis, Ronald Ch. Siddall. Numerical homogenization of the elastic properties of a 3D rock with a
discrete fracture network (DFN), validated against a direct numerical simulation (DNS) of the fractured rock.

The computational core is the application `upscale_m` of the [endorse](https://github.com/GeoMop/endorse)
library (fork [RonaldSiddall/endorse](https://github.com/RonaldSiddall/endorse), branch `thesis_branch`).
This repository only configures and runs it; the mechanics is solved by
[Flow123d](https://flow123d.github.io/).

## Repository layout

| path | content |
|---|---|
| `code/endorse` | submodule, endorse with `apps/upscale_m` (meshes, boundary conditions, homogenization, macro problem) |
| `code/bgem` | submodule, bgem (DFN sampling, meshing) |
| `configs/` | configs of the simulations: `config_test.yaml` (quick test task), `config_full.yaml` (full task) |
| `scripts/` | scripts that run the simulations from this repository |
| `runs/` | results of the simulations, `runs/<run.name>/` (not in git) |
| `requirements.txt` | exact versions of the Python packages |

## Setup

Clone with the submodules:

```
git clone --recurse-submodules https://github.com/RonaldSiddall/Validation-of-numerical-homogenization-of-mechanical-processes-in-3D-fractured-media.git
```

Python 3.11, virtual environment with the packages and the two submodules installed in editable mode:

```
python -m venv .venv
.venv/Scripts/activate          # Windows (Git Bash); Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e code/endorse -e code/bgem
```

Flow123d runs in Docker, image `flow123d/ci-gnu:4.0.3dev_9e91fe` (Flow123d 4.0.3dev, branch JS_constraints,
commit 9e91fe8). This version is needed by the macro problem: it reads the full anisotropic stiffness tensor
(`elasticity_type: general`, fields `stiffness_tensor_0` .. `stiffness_tensor_5`). The command that starts
Flow123d is set per computer in `machine_config` of the config, under the computer's hostname.

## Configs

Each config in `configs/` has a section `run` (what to run) and all the parameters of the simulations:

| key | meaning |
|---|---|
| `run.name`, `run.output_dir` | the results go to `<output_dir>/<name>/`, with a copy of the config |
| `run.dns` | run the DNS (reference solution) |
| `run.macro` | run the micro problem, the homogenization and the macro problem |
| `run.bc_types` | `[kubc, subc]`, `[kubc]` or `[subc]` |
| `run.cases` | `all`, or a list of load cases, e.g. `[E_11, S_12]` |

## Running the simulations

Work in progress: `scripts/run_dns.py`, `scripts/run_macro.py` and `scripts/main.py`.
