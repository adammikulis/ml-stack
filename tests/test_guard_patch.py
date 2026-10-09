"""The guard-change diff as plain data: what parses, what is refused, what the hash covers and how it applies."""

from __future__ import annotations

import pytest

from poolhouse.workspace import guard_patch

PATH = "scripts/hooks/pre-push"
OLD = "#!/bin/sh\nexit 0\n"
NEW = "#!/bin/sh\necho checked\nexit 0\n"


@pytest.mark.parametrize("text", [
    "--- a/src/poolhouse/other.py\n+++ b/src/poolhouse/other.py\n@@ -1 +1 @@\n-a\n+b\n",
    f"--- a/{PATH}\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n",
    f"rename from {PATH}\nrename to scripts/hooks/pre-push2\n",
    f"old mode 100644\nnew mode 100755\n--- a/{PATH}\n+++ b/{PATH}\n@@ -1 +1 @@\n-a\n+b\n",
    "--- a/scripts/hooks/../../x\n+++ b/scripts/hooks/../../x\n@@ -1 +1 @@\n-a\n+b\n",
    f"--- a/{PATH}\r\n+++ b/{PATH}\r\n",
    f"--- a/{PATH}\n+++ b/{PATH}\n@@ -1,2 +1 @@\n-a\n",
    f"--- a/{PATH}\n+++ b/{PATH}\n@@ -1 +1 @@\n-a\n+b\n--- a/{PATH}\n+++ b/{PATH}\n@@ -1 +1 @@\n-a\n+b\n",
    "",
])
def test_a_diff_outside_the_protected_set_or_beyond_a_plain_edit_is_refused(text):
    with pytest.raises(guard_patch.Refused):
        guard_patch.parse(text)


def test_a_diff_too_long_to_read_is_refused():
    body = "".join(f"+line {n}\n" for n in range(guard_patch.MAX_LINES))
    with pytest.raises(guard_patch.Refused, match="split it"):
        guard_patch.parse(f"--- /dev/null\n+++ b/{PATH}\n@@ -0,0 +1,{guard_patch.MAX_LINES} @@\n{body}")


def test_the_hash_ignores_timestamps_and_headings_and_nothing_else():
    a = f"--- a/{PATH}\t2026-01-01\n+++ b/{PATH}\t2026-01-02\n@@ -1 +1 @@ heading\n-a\n+b\n"
    b = f"diff --git a/{PATH} b/{PATH}\nindex 1..2\n--- a/{PATH}\n+++ b/{PATH}\n@@ -1 +1 @@\n-a\n+b\n"
    c = b.replace("+b", "+c")
    assert guard_patch.parse(a).sha == guard_patch.parse(b).sha != guard_patch.parse(c).sha


def test_a_diff_made_here_parses_and_applies_to_exactly_the_text_it_was_made_from():
    patch = guard_patch.parse(guard_patch.diff(PATH, OLD, NEW))
    assert [f.path for f in patch.files] == [PATH]
    assert guard_patch.expected(patch.files[0], OLD) == NEW
    assert guard_patch.expected(patch.files[0], "#!/bin/sh\nexit 1\n") is None
    assert patch.sha in guard_patch.render(patch)


def test_a_new_file_diff_applies_only_where_no_file_exists():
    patch = guard_patch.parse(guard_patch.diff(PATH, None, NEW))
    assert patch.files[0].new
    assert guard_patch.expected(patch.files[0], None) == NEW
    assert guard_patch.expected(patch.files[0], OLD) is None
