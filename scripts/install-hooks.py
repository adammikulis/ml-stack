"""Install repository Git hooks with the running Python interpreter."""

import shlex
import subprocess
import sys
from pathlib import Path

HOOKS = ("pre-commit", "commit-msg", "pre-push", "post-merge", "post-commit")
MARKER = "# poolhouse managed hook"


def install() -> None:
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    destination = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-path", "hooks"], text=True).strip())
    destination.mkdir(parents=True, exist_ok=True)
    for name in HOOKS:
        hook = destination / name
        owned = (hook.is_symlink() and "scripts/hooks/" in hook.readlink().as_posix())
        if hook.exists() and not owned and MARKER not in hook.read_text(encoding="utf-8", errors="replace"):
            print(f"install-hooks: {hook} exists and is not ours; leaving it alone", file=sys.stderr)
            continue
        if hook.is_symlink():
            hook.unlink()
        source = (root / "scripts" / "hooks" / name).as_posix()
        interpreter = shlex.quote(Path(sys.executable).as_posix())
        hook.write_text("#!/bin/sh\n" + MARKER + "\n"
                        + f"export PYTHON={interpreter}\n"
                        + f"exec sh {shlex.quote(source)} \"$@\"\n", encoding="utf-8", newline="\n")
        hook.chmod(hook.stat().st_mode | 0o111)
        print(f"install-hooks: {name} -> scripts/hooks/{name}")


if __name__ == "__main__":
    install()
