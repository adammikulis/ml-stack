"""Readable ownership output for shared claims."""

from ml_stack.workspace.render import text as _text


def test_unclaimed_owner_is_rendered_without_identity_conversion():
    assert _text({'kind': 'file', 'key': 'unclaimed.py', 'owner': None}) == (
        'kind: file\nkey: unclaimed.py\nowner: None')


def test_claimed_owner_is_rendered_readably():
    assert 'worker on shared project Board' in _text({
        'owner': 'board:' + 'a' * 32 + ':worker'})
