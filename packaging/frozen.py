"""Collect project modules and dynamic plugin metadata for frozen bundles."""

from importlib.metadata import distribution


def collect_project():
    """Return project plugin metadata and Python module names."""
    from PyInstaller.utils.hooks import collect_entry_point

    installed = distribution("ml-stack")
    modules = set()
    for file in installed.files or ():
        if file.parts[0] != "ml_stack" or file.suffix != ".py":
            continue
        parts = list(file.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if all(part.isidentifier() for part in parts):
            modules.add(".".join(parts))
    modules.update(entry.module for entry in installed.entry_points)
    datas = []
    groups = sorted({entry.group for entry in installed.entry_points
                     if entry.group.startswith("ml_stack.")})
    for group in groups:
        plugin_data, plugin_modules = collect_entry_point(group)
        datas.extend(plugin_data)
        modules.update(plugin_modules)
    return datas, sorted(modules)

