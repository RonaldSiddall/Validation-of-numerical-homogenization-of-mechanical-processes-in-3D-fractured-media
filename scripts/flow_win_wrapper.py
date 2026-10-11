import os
import re
import subprocess
import sys

DOCKER = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
# now using branch JS_constraints, commit 9e91fe8, which adds support for full anisotropic Cauchy tensor
# i.e. adds elasticity_type: general with stiffness_tensor_0..5
# (6 columns of the 6x6 tensor, each a symmetric 3x3 in Kelvin notation, readable via !FieldFE).
IMAGE = "flow123d/ci-gnu:4.0.3dev_9e91fe"

ENV = {"MALLOC_PERTURB_": "255"}

_WIN_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def to_container(path: str) -> str:
    m = _WIN_PATH.match(path)
    if not m:
        return path
    drive, rest = m.group(1), m.group(2).replace("\\", "/")
    return f"/{drive}/{rest}"


def main() -> int:
    cwd = os.getcwd()
    drive = cwd[0]
    container_cwd = to_container(cwd)
    args = [to_container(a) for a in sys.argv[1:]]

    env_args = [a for k, v in ENV.items() for a in ("-e", f"{k}={v}")]
    cmd = [
        DOCKER, "run", "--rm",
        *env_args,
        "-v", f"{drive.upper()}:\\:/{drive.upper()}/",
        "-v", f"{drive.lower()}:\\:/{drive.lower()}/",
        "-w", container_cwd,
        IMAGE, "flow123d", *args,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
