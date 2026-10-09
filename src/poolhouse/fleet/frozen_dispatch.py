"""Python module and owned script entrypoints for the frozen executable."""

import importlib
import importlib.util
import runpy
import sys
from importlib.metadata import distribution
from pathlib import Path

from poolhouse import agent_dependency

from . import launch


def _module(argument):
    if argument.endswith(".py"):
        root = Path(__file__).resolve().parents[1]
        try:
            relative = Path(argument).resolve().relative_to(root)
        except ValueError:
            return ""
        return "poolhouse." + ".".join(relative.with_suffix("").parts)
    return argument


def _owned(module):
    if len(module) > 256 or not module.startswith("poolhouse."):
        return False
    if not all(part.isidentifier() for part in module.split(".")):
        return False
    files = {str(file) for file in distribution("poolhouse").files or ()}
    path = module.replace(".", "/")
    return path + ".py" in files or path + "/__main__.py" in files


def _target(module):
    targets = {point.value: point for point in distribution("poolhouse").entry_points
               if point.group == "console_scripts" and point.module == module}
    return next(iter(targets.values())).load() if len(targets) == 1 else None


def _runnable(module):
    spec = importlib.util.find_spec(module)
    if spec is None or spec.loader is None:
        return False
    if spec.submodule_search_locations is not None:
        return importlib.util.find_spec(module + ".__main__") is not None
    get_code = getattr(spec.loader, "get_code", None)
    code = get_code(module) if get_code is not None else None
    return code is not None and "__main__" in code.co_consts


def main(argv=None):
    """Dispatch owned module/script arguments or launch the device interface."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--check-agent-runtime"]:
        return agent_dependency.report()
    if args[:1] == ["-m"]:
        module = _module(args[1]) if len(args) > 1 else ""
        rest = args[2:]
    elif args and args[0].endswith(".py"):
        module, rest = _module(args[0]), args[1:]
    elif args[:1] == ["-c"]:
        sys.stderr.write("Unsupported frozen Python invocation\n")
        return 2
    else:
        return launch.main(args)
    if not _owned(module):
        sys.stderr.write("Unsupported frozen module or script\n")
        return 2
    sys.argv = [module, *rest]
    if module == "poolhouse.harnesshook":
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    target = _target(module)
    if target is not None:
        return target(rest) or 0
    if not _runnable(module):
        sys.stderr.write("Frozen module has no command entrypoint\n")
        return 2
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0
