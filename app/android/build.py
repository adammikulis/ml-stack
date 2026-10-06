"""Run Android build tasks under the maintained CPU broker."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import testslots


def main() -> int:
    gradle, plugin, api = sys.argv[1:]
    with testslots.lease(0, label="Android build") as admitted:
        return subprocess.run([gradle, "--no-daemon", f"--max-workers={admitted.workers}",
                               f"-PandroidPlugin={plugin}", f"-PandroidApi={api}",
                               "protocolChecks", "lintRelease", "assembleRelease"],
                              cwd=Path(__file__).parent).returncode


if __name__ == "__main__":
    raise SystemExit(main())
