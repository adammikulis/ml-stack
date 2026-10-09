"""Collect project modules and dynamic plugin metadata for frozen bundles."""

from importlib.metadata import distribution
from pathlib import Path


def collect_project():
    """Return project plugin metadata and Python module names."""
    from PyInstaller.utils.hooks import collect_entry_point

    installed = distribution("ml-stack")
    modules = set()
    datas = []
    for file in installed.files or ():
        if file.parts[0] != "ml_stack" or file.suffix != ".py":
            continue
        parts = list(file.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if all(part.isidentifier() for part in parts):
            modules.add(".".join(parts))
            # Source inspection and bootstrap extraction use real __file__ paths.
            datas.append((str(installed.locate_file(file)), str(Path(*file.parts[:-1]))))
    modules.update(entry.module for entry in installed.entry_points)
    groups = sorted({entry.group for entry in installed.entry_points
                     if entry.group.startswith("ml_stack.")})
    for group in groups:
        plugin_data, plugin_modules = collect_entry_point(group)
        datas.extend(plugin_data)
        modules.update(plugin_modules)
    return datas, sorted(modules)
