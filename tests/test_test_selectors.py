"""Explicit pytest paths cannot vanish into an empty successful quick run."""

import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner():
    loader = importlib.machinery.SourceFileLoader('tier_runner', str(ROOT / 'scripts' / 'test'))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize('path', ['tests/no-such-file.py', 'tests/no-such-directory',
                                  'tests/no-such-file.py::test_missing'])
def test_missing_explicit_path_is_rejected_before_broker(path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'test'), 'all', path, '-n', '2'],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 4
    assert 'test path does not exist' in result.stderr
    assert '+ ' not in result.stdout


def test_existing_node_selector_and_option_values_are_preserved():
    module = runner()
    module.testselectors.validate(['-k', 'test_missing', '-m', 'not slow',
                                  'tests/test_test_selectors.py::test_missing_explicit_path_is_rejected_before_broker'], ROOT)


def test_xdist_empty_explicit_node_is_an_error_even_for_quick(monkeypatch):
    module = runner()
    monkeypatch.setattr(module.testslots_runner, 'run_pytest', lambda *args, **kwargs: 5)
    monkeypatch.setattr(module, 'record_run', lambda *args: None)
    status = module.run(module.pytest_command(2, 'tests/test_test_selectors.py::test_nonexistent'), tier='quick')
    assert status == 4 and module.clean(status) == 4


@pytest.mark.parametrize(('options', 'workers'), [([], 0), (['-n', '0'], 0), (['-n', '1'], 1), (['-n', '3'], 3)])
def test_runner_defaults_to_the_brokers_automatic_pool_and_preserves_explicit_limits(monkeypatch, options, workers):
    module = runner()
    calls = []
    monkeypatch.setenv('DEV_TEST_JOB', 'inline')
    monkeypatch.setattr(sys, 'argv', ['scripts/test', 'all', 'tests/test_test_selectors.py', *options])
    monkeypatch.setattr(module, 'run', lambda command, **kwargs: calls.append(command) or 0)
    assert module.main() == 0
    assert calls[0][calls[0].index('-n') + 1] == str(workers)


def test_no_gate_step_writes_to_the_tree():
    """A gate only reads: a step that rewrote a tracked file dirtied the worktree it was judging."""
    module = runner()
    steps = module.gate_steps(0, [])
    assert [name for name, _ in steps] == ['budgets', 'redteam coverage', 'command reference', 'structure tests']
    for name, command in steps:
        assert not {'--write', '--update', '--fix'} & set(command), (name, command)
    assert [part for part in dict(steps)['command reference'] if part.startswith('--')] == ['--check']


def test_a_stale_command_table_fails_the_check_and_leaves_the_page_alone(tmp_path):
    page = tmp_path / 'commands.md'
    page.write_text('# The commands\n\n| stale | table |\n', encoding='utf-8')
    before = page.read_bytes()
    done = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'reference'), '--check', '--page', str(page)],
                          capture_output=True, text=True, check=False, cwd=ROOT)
    assert done.returncode != 0
    assert page.read_bytes() == before
