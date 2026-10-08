"""Literal source fixture planning without supervisor-side imports."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_fixture_plan import FixturePlan, resources


def source(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def plan(root: Path, selector: str, files):
    return FixturePlan([sys.executable, '-m', 'pytest', selector], root, files)


def test_literal_transitive_imported_fixture_aliases_plan_without_executing_test_code(tmp_path):
    helper = source(tmp_path, 'tests/helper.py', '''
import pytest
from test_fixture_plan import resources
raise RuntimeError("supervisor must not import test code")
@resources('browser', 'browser-endpoints')
@pytest.fixture
def page(tmp_path):
    return object()
''')
    test = source(tmp_path, 'tests/test_example.py', '''
import pytest
import helper as imported
page_alias = imported.page
@pytest.fixture
def conversation(page_alias):
    return page_alias
@pytest.mark.parametrize('size', [1, 2])
def test_chat(conversation, size):
    assert conversation
''')
    result = plan(tmp_path, 'tests/test_example.py::test_chat[1]', [helper, test])
    assert result.resources == frozenset({'browser', 'browser-endpoints'})
    assert len(result.proof()['sources']) == 2
    assert result.proof()['selected'] == [('tests/test_example.py', ('test_chat',))]


def test_autouse_usefixtures_class_selection_and_conftest_overrides(tmp_path):
    outer = source(tmp_path, 'conftest.py', '''
import pytest
from test_fixture_plan import resources
@resources('javascript')
@pytest.fixture(autouse=True)
def logging(monkeypatch):
    pass
@resources('browser')
@pytest.fixture
def display():
    pass
''')
    inner = source(tmp_path, 'tests/conftest.py', '''
import pytest
from test_fixture_plan import resources
@resources('browser-endpoints')
@pytest.fixture
def display():
    pass
''')
    test = source(tmp_path, 'tests/test_example.py', '''
import pytest
@pytest.mark.usefixtures('display')
class TestChat:
    def test_selected(self):
        pass
    def test_other(self, unresolved):
        pass
''')
    result = plan(tmp_path, 'tests/test_example.py::TestChat::test_selected', [outer, inner, test])
    assert result.resources == frozenset({'javascript', 'browser-endpoints'})
    assert result.proof()['selected'] == [('tests/test_example.py', ('TestChat', 'test_selected'))]


@pytest.mark.parametrize('declaration', [
    "@resources(resource_name)", "@resources('arbitrary-host-network')",
    "@unknown.resources('browser')", "@resources(**options)",
])
def test_dynamic_or_forged_resource_declarations_are_refused(tmp_path, declaration):
    test = source(tmp_path, 'tests/test_example.py', f'''
import pytest
from test_fixture_plan import resources
{declaration}
@pytest.fixture
def page():
    pass
def test_chat(page):
    pass
''')
    with pytest.raises(RuntimeError, match=r'declaration|decorator'):
        plan(tmp_path, 'tests/test_example.py::test_chat', [test])


def test_source_mutation_missing_dependency_and_unadmitted_import_are_refused(tmp_path):
    test = source(tmp_path, 'tests/test_example.py', 'def test_chat(tmp_path):\n    pass\n')
    result = plan(tmp_path, 'tests/test_example.py', [test])
    test.write_text('def test_chat(monkeypatch):\n    pass\n')
    with pytest.raises(RuntimeError, match='source identity changed'):
        result.recheck()
    test.write_text('def test_chat(missing):\n    pass\n')
    with pytest.raises(RuntimeError, match='unresolved fixture'):
        plan(tmp_path, 'tests/test_example.py', [test])
    with pytest.raises(RuntimeError, match='outside maintained snapshot'):
        plan(tmp_path, 'tests/test_example.py', [])


def test_dynamic_fixture_lookup_and_selection_filters_fail_closed(tmp_path):
    test = source(tmp_path, 'tests/test_example.py', '''
def test_chat(request):
    request.getfixturevalue(chosen_name)
''')
    with pytest.raises(RuntimeError, match='dynamic fixture'):
        plan(tmp_path, 'tests/test_example.py::test_chat', [test])
    with pytest.raises(RuntimeError, match='filtered invocation'):
        FixturePlan([sys.executable, '-m', 'pytest', 'tests/test_example.py', '-k', 'chat'], tmp_path, [test])
    with pytest.raises(ValueError, match='invalid literal'):
        resources('foreign-network')


def test_visible_unadmitted_conftest_and_module_usefixtures_are_accounted_for(tmp_path):
    configuration = source(tmp_path, 'tests/conftest.py', '''
import pytest
from test_fixture_plan import resources
@resources('javascript')
@pytest.fixture
def parsed_scripts():
    pass
''')
    test = source(tmp_path, 'tests/test_example.py', '''
import pytest
pytestmark = pytest.mark.usefixtures('parsed_scripts')
def test_chat():
    pass
''')
    with pytest.raises(RuntimeError, match='visible conftest'):
        plan(tmp_path, 'tests/test_example.py', [test])
    result = plan(tmp_path, 'tests/test_example.py', [test, configuration])
    assert result.resources == frozenset({'javascript'})


def test_autouse_override_uses_effective_fixture_and_self_dependency_reaches_parent(tmp_path):
    outer = source(tmp_path, 'conftest.py', '''
import pytest
from test_fixture_plan import resources
@resources('browser')
@pytest.fixture(autouse=True)
def context():
    pass
''')
    inner = source(tmp_path, 'tests/conftest.py', '''
import pytest
from test_fixture_plan import resources
@resources('javascript')
@pytest.fixture
def context():
    pass
''')
    test = source(tmp_path, 'tests/test_example.py', 'def test_chat():\n    pass\n')
    assert plan(tmp_path, 'tests/test_example.py', [outer, inner, test]).resources == frozenset({'javascript'})
    inner.write_text(inner.read_text().replace('def context():', 'def context(context):'))
    assert plan(tmp_path, 'tests/test_example.py', [outer, inner, test]).resources == frozenset({'browser', 'javascript'})


@pytest.mark.parametrize('selection', ['/tmp/test_example.py', './tests/test_example.py', 'tests/', '--ignore=tests/other.py', '--deselect=tests/test_example.py::test_chat'])
def test_unsupported_selection_forms_refuse_instead_of_dropping_sources(tmp_path, selection):
    test = source(tmp_path, 'tests/test_example.py', 'def test_chat():\n    pass\n')
    with pytest.raises(RuntimeError, match='unsupported invocation selection'):
        FixturePlan([sys.executable, '-m', 'pytest', 'tests/test_example.py', selection], tmp_path, [test])


def test_module_usefixtures_marks_apply_to_tests_and_do_not_modify_fixture_dependencies(tmp_path):
    test = source(tmp_path, 'tests/test_example.py', '''
import pytest
from test_fixture_plan import resources
pytestmark = pytest.mark.usefixtures('display')
@resources('browser-endpoints')
@pytest.fixture
def display():
    pass
def test_chat():
    pass
''')
    result = plan(tmp_path, 'tests/test_example.py::test_chat', [test])
    assert result.resources == frozenset({'browser-endpoints'})
    assert result.node_resources == {'tests/test_example.py::test_chat': frozenset({'browser-endpoints'})}
    assert result.proof()['node_resources'] == {'tests/test_example.py::test_chat': ['browser-endpoints']}
