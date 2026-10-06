"""Named primary files preserved by an independently reviewed integration commit."""

from pathlib import Path

from ml_stack.workspace import integration_git as repo
from ml_stack.workspace.identity import Denied


def files(primary: Path, commit: str) -> list[str]:
    """Return changed primary paths whose working and staged bytes match the commit."""
    repo.clean(primary, allow_changes=True)
    entries = repo.git(primary, 'status', '--porcelain', '-z', binary=True).split(b'\0')
    names = []
    for entry in filter(None, entries):
        status, name = entry[:2], entry[3:].decode('utf-8')
        if status not in (b' M', b'M ', b'MM', b'A ', b'AM'):
            raise Denied('primary preservation requires named tracked regular files')
        path = Path(name)
        if path.anchor or '..' in path.parts or '\\' in name:
            raise Denied('primary preservation requires relative repository paths')
        target = primary / path
        if any(part.is_symlink() for part in (target, *target.parents)) or not target.is_file():
            raise Denied('primary preservation refuses symlinks and missing files')
        tracked = repo.git(primary, 'ls-tree', commit, '--', name)
        if not tracked.startswith(('100644 blob ', '100755 blob ')):
            raise Denied('primary preservation requires a committed regular file')
        content = repo.git(primary, 'show', f'{commit}:{name}', binary=True)
        if target.read_bytes() != content:
            raise Denied(f'primary working bytes differ from the reviewed commit: {name}')
        if status[:1] != b' ' and repo.git(primary, 'show', f':{name}', binary=True) != content:
            raise Denied(f'primary staged bytes differ from the reviewed commit: {name}')
        names.append(name)
    return names


def stage(primary: Path, commit: str, names: list[str]) -> None:
    """Stage the verified named files before the candidate fast-forward."""
    if files(primary, commit) != names:
        raise Denied('primary preservation paths changed during integration')
    if names:
        repo.git(primary, 'add', '--', *names)
        if repo.git(primary, 'diff', '--cached', '--name-only', commit, '--', *names):
            raise Denied('Git filters changed the reviewed primary staging bytes')
