"""What a run on another device prints: the lines of a local run, with the device and its platform named first."""

from __future__ import annotations


def where(platform: dict) -> str:
    """``linux (WSL), python 3.13.5, 16 cpus``: a Linux that is WSL is not the Windows around it."""
    system = f"{platform['system']} (WSL)" if platform.get("wsl") else platform["system"]
    return f"{system}, python {platform['python']}, {platform['cpus']} cpus"


def counts(result: dict) -> dict[str, int]:
    """Passed, failed, error and skipped over every file of a result."""
    return {key: sum(c[key] for c in result["files"].values()) for key in ("passed", "failed", "error", "skipped")}


def summary(result: dict) -> str:
    """A pytest-style last line for a remote result."""
    shown = ", ".join(f"{n} {key}" for key, n in counts(result).items() if n) or "no tests ran"
    return f"{shown} in {result['wall_s']:.1f}s"


def show(result: dict, name: str, reused: list[str] | None = None) -> list[str]:
    """The lines a remote run prints, in the shape of a local one."""
    lines = [f"+ ran on {name}: {where(result['platform'])}"]
    lines += [f"reused: {file} passed on {name} with this content" for file in reused or []]
    lines += [f"FAILED {f['nodeid']} - {f['message'].splitlines()[0][:200] if f['message'] else f['state']}"
              for f in result["failures"]]
    if result["exit"] and not result["failures"]:
        lines += result["output_tail"].splitlines()[-20:]
    lines.append(f"{summary(result)} (exit {result['exit']}, {result['cpu_s']:.1f}s cpu, on {name})")
    return lines


def board_line(device: dict, tier: str, files: int, result: dict, tree: str) -> str:
    """The one line posted to the board for a run: key=value pairs a reader can grep, keyed by device."""
    n = counts(result)
    return (f"test-result device={device['name']} fp={device['fingerprint'][:16]} tier={tier} files={files} exit={result['exit']} "
            f"passed={n['passed']} failed={n['failed'] + n['error']} skipped={n['skipped']} wall_s={result['wall_s']:.0f} "
            f"platform={result['platform']['system']}{'-wsl' if result['platform'].get('wsl') else ''} tree={tree[:12]}")
