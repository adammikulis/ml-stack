"""Person-facing installed model metadata from maintained discovery."""


from ml_stack import hub
from ml_stack.fleet.catalogue import family_of
from ml_stack.hub.capabilities import capabilities
from ml_stack.hub.naming import aside


def library(roots):
    """Primary model entries with exact files, companions and runtime status."""
    rows = []
    for model in hub.discover(roots=roots):
        path = model.path.resolve()
        if aside(path.name):
            continue
        supported = model.format in {'gguf', 'mlx'}
        status = 'ready' if model.is_complete and supported else 'incomplete' if not model.is_complete else 'unsupported'
        parts = [file for part in model.files for file in
                 (hub.weight_paths([part]) if part.is_dir() else [part])]
        rows.append({**model.as_dict(), 'path': str(path), 'family': family_of(model.name),
                     'capabilities': capabilities(path, architecture=model.architecture),
                     'servable': status == 'ready', 'status': status,
                     'files': [{'name': file.name, 'size_bytes': file.stat().st_size} for file in parts if file.is_file()],
                     'companion': {'name': model.mmproj.name, 'path': str(model.mmproj.resolve())}
                                  if model.mmproj else None})
    return rows
