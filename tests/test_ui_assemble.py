"""`ml_stack.ui.assemble`: a page out of component files, read back as a string."""

from __future__ import annotations

import re

import pytest

from ml_stack.graph import page as graph_page
from ml_stack.ui import Component, assemble, load

SHELL = """<title>__TITLE__</title>
<style>body {}</style>
__STYLES__
<main>
  <one-thing></one-thing>
  <div class="row"><!--mount:row--></div>
  <not-listed></not-listed>
</main>
__SCRIPTS__
"""


def component(tmp_path, name, text):
    (tmp_path / f"{name}.html").write_text(text, encoding="utf-8")
    return Component(name, tmp_path / f"{name}.html")


def test_a_component_lands_at_its_tag_with_its_style_and_script(tmp_path):
    one = component(tmp_path, "one-thing",
                    "<style>.one { color: red; }</style>\n"
                    "<template><p class=\"one\">hello</p></template>\n"
                    "<script>customElements.define('one-thing', class extends HTMLElement {});</script>")
    html = assemble(SHELL, [one])
    assert '<one-thing><p class="one">hello</p></one-thing>' in html
    assert "<style>\n.one { color: red; }\n</style>" in html
    assert "customElements.define('one-thing'" in html
    assert "__STYLES__" not in html and "__SCRIPTS__" not in html


def test_a_tag_with_no_component_behind_it_is_removed(tmp_path):
    one = component(tmp_path, "one-thing", "<template>x</template><script>1</script>")
    html = assemble(SHELL, [one])
    assert "not-listed" not in html


def test_a_listed_component_with_no_tag_in_the_shell_contributes_nothing(tmp_path):
    one = component(tmp_path, "one-thing", "<template>x</template><script>1</script>")
    other = component(tmp_path, "other-thing",
                      "<style>.other {}</style><template>y</template><script>OTHER</script>")
    html = assemble(SHELL, [one, other])
    assert ".other" not in html and "OTHER" not in html


def test_a_component_with_no_markup_is_script_and_style_alone_and_always_in(tmp_path):
    model = component(tmp_path, "page-model", "<script>window.model = 1;</script>")
    one = component(tmp_path, "one-thing", "<template>x</template><script>ONE</script>")
    html = assemble(SHELL, [model, one])
    assert html.index("window.model = 1") < html.index("ONE")


def test_a_mount_takes_every_listed_components_fragment_in_order(tmp_path):
    one = component(tmp_path, "one-thing",
                    "<template>x</template><template mount=\"row\"><b>from one</b></template>"
                    "<script>1</script>")
    two = component(tmp_path, "two-thing",
                    "<template mount=\"row\"><i>from two</i></template><script>2</script>")
    shell = SHELL.replace("<not-listed></not-listed>", "<two-thing></two-thing>")
    html = assemble(shell, [one, two])
    assert re.search(r'<div class="row"><b>from one</b>\n<i>from two</i></div>', html)


def test_a_template_may_hold_another_components_tag(tmp_path):
    one = component(tmp_path, "one-thing",
                    "<template><section><two-thing></two-thing></section></template><script>1</script>")
    two = component(tmp_path, "two-thing", "<template>inner</template><script>TWO</script>")
    html = assemble(SHELL, [one, two])
    assert "<section><two-thing>inner</two-thing></section>" in html
    assert "TWO" in html


def test_a_file_needs_exactly_one_script(tmp_path):
    none = component(tmp_path, "one-thing", "<template>x</template>")
    with pytest.raises(ValueError, match="one <script>"):
        assemble(SHELL, [none])


def test_load_names_files_in_a_directory(tmp_path):
    (tmp_path / "a-b.html").write_text("<script>1</script>")
    [found] = load(tmp_path, ["a-b"])
    assert found.name == "a-b" and found.path == tmp_path / "a-b.html"


def test_the_graph_page_is_the_shell_plus_every_component_it_lists():
    html = graph_page.template()
    for name in graph_page.COMPONENTS:
        if name == "page-model":
            continue
        assert f"customElements.define('{name}'" in html, name
    assert "window.graphModel = {}" in html
    assert "<!--mount:" not in html
    empty = set(re.findall(r"<([a-z]+-[a-z0-9-]+)></\1>", html))
    assert empty <= set(graph_page.COMPONENTS), "a tag nothing is behind was left in"


def test_a_component_left_out_takes_its_markup_style_and_script_with_it():
    without = [n for n in graph_page.COMPONENTS if n not in ("graph-3d", "review-queue")]
    html = graph_page.template(without)
    assert "graph3d" not in html and "customElements.define('graph-3d'" not in html
    assert "review-box" not in html and "customElements.define('review-queue'" not in html
    assert 'id="v3d"' not in html
    assert "customElements.define('graph-view'" in html
