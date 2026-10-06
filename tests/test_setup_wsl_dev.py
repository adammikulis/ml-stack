from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="setup provisions Python 3.12")
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_setup_wsl_dev_persists_cuda_compiler_in_environment(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    venv = tmp_path / "venv"
    home.mkdir()
    fake_bin.mkdir()

    uname = fake_bin / "uname"
    uname.write_text("#!/bin/sh\necho 6.6.0-microsoft-standard-WSL2\n")
    uname.chmod(0o755)

    uv_log = tmp_path / "uv.log"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = venv ]; then\n"
        "  exec \"$PYTHON_FOR_TEST\" -m venv \"$4\"\n"
        "fi\n"
        "printf '%s\\n' \"$*\" >> \"$UV_LOG\"\n"
    )
    uv.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "ML_STACK_WSL_VENV": str(venv),
            "BASHRC": str(home / ".bashrc"),
            "UV": str(uv),
            "UV_LOG": str(uv_log),
            "PYTHON_FOR_TEST": sys.executable,
            "ML_STACK_WINDOWS_USER": "no-such-user",
        }
    )
    subprocess.run(
        ["bash", str(repo / "scripts/setup-wsl-dev")],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    nvcc = venv / "lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n")
    nvcc.chmod(0o755)

    config = home / ".config/ml-stack/wsl-dev.sh"
    sourced = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; printf "%s\\n%s\\n%s\\n" "$CUDA_HOME" "$CUDA_PATH" "$(command -v nvcc)"; nvcc --version',
            "test-shell",
            str(config),
        ],
        env={**env, "PATH": "/usr/bin:/bin", "PYTHONPATH": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    assert sourced.stdout.splitlines() == [
        str(venv / "lib/python3.12/site-packages/nvidia/cu13"),
        str(venv / "lib/python3.12/site-packages/nvidia/cu13"),
        str(nvcc),
        "Cuda compilation tools, release 13.0, V13.0.88",
    ]

    install_args = uv_log.read_text()
    assert "build>=1.2" in install_args
    assert "cuda-toolkit[nvcc]==13.0.3" in install_args
