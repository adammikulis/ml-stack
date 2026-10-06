"""Named primary files preserved by an independently reviewed integration commit."""

from pathlib import Path

from ml_stack.workspace import integration_git as repo
from ml_stack.workspace.identity import Denied


def merge_state(primary: Path, commit: str) -> bool:
    """Verify unfinished merge commits are contained in the reviewed commit."""
    for marker in ('CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply'):
        if (primary / repo.git(primary, 'rev-parse', '--git-path', marker)).exists():
            raise Denied(f'primary preservation refuses unfinished {marker}')
    marker = primary / repo.git(primary, 'rev-parse', '--git-path', 'MERGE_HEAD')
    if not marker.exists():
        return False
    if marker.is_symlink() or not marker.is_file():
        raise Denied('primary preservation refuses a nonregular merge marker')
    heads = marker.read_text().splitlines()
    if not heads or any(not repo.COMMIT.fullmatch(head) or not repo.ancestor(primary, head, commit)
                        for head in heads):
        raise Denied('the unfinished merge is not contained in the reviewed commit')
    if not repo.ancestor(primary, repo.git(primary, 'rev-parse', 'HEAD'), commit):
        raise Denied('the primary baseline is not contained in the reviewed commit')
    automatic = primary / repo.git(primary, 'rev-parse', '--git-path', 'AUTO_MERGE')
    if automatic.exists() and (automatic.is_symlink() or automatic.read_text().strip() !=
                               repo.git(primary, 'rev-parse', f'{commit}^{{tree}}')):
        raise Denied('the unfinished automatic merge tree differs from the reviewed commit')
    return True


def files(primary: Path, commit: str) -> list[str]:
    """Return changed primary paths whose working and staged bytes match the commit."""
    merge_state(primary, commit)
    entries = repo.git(primary, 'status', '--porcelain', '-z', binary=True).split(b'\0')
    names = []
    for entry in filter(None, entries):
        status, name = entry[:2], entry[3:].decode('utf-8')
        if status == b' M':
            raise Denied('primary preservation requires files already staged before agent integration')
        if status not in (b'M ', b'MM', b'A ', b'AM'):
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


def verify(primary: Path, commit: str, names: list[str]) -> None:
    """Verify named primary files before the candidate fast-forward."""
    if files(primary, commit) != names:
        raise Denied('primary preservation paths changed during integration')
