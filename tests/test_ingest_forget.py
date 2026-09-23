"""``ml-stack-ingest forget``: the reads files beside a store deleted, the store left alone."""

from __future__ import annotations

from ml_stack.graph.store import GraphStore
from ml_stack.ingest import main
from ml_stack.ingest.reads import _keep_reads, reads_path
from ml_stack.ingest.sources import Sources

ROW = {"unit": "u1", "source": "Field Notes", "chapter": "1", "section": "1.1",
       "title": "Levels", "extracted": {"concepts": [{"name": "benchmark",
                                                      "definition": "a surveyed mark"}]},
       "raw": ""}


def _kept(tmp_path):
    out = tmp_path / "corpus.ladybug"
    with GraphStore(out) as store:
        store.upsert_node({"id": "concept:benchmark", "kind": "concept", "label": "benchmark"})
    for slug in ("field-notes", "almanac"):
        _keep_reads(out, slug, [dict(ROW, source=slug)])
    other = tmp_path / "corpus.ladybug2"
    _keep_reads(other, "field-notes", [ROW])
    return out, other


def test_forget_a_source_deletes_its_reads_alone(tmp_path, capsys):
    out, other = _kept(tmp_path)
    assert main(["forget", "--out", str(out), "--source", "field-notes"]) == 0
    assert not reads_path(out, "field-notes").exists()
    assert reads_path(out, "almanac").exists()
    assert reads_path(other, "field-notes").exists()
    assert str(reads_path(out, "field-notes")) in capsys.readouterr().out
    assert Sources(out).reads("field-notes") == []
    with GraphStore(out, read_only=True) as store:
        assert [n["id"] for n in store.nodes()] == ["concept:benchmark"]


def test_forget_deletes_every_reads_file_beside_that_store_and_no_other(tmp_path, capsys):
    out, other = _kept(tmp_path)
    assert main(["forget", "--out", str(out)]) == 0
    assert not reads_path(out, "field-notes").exists()
    assert not reads_path(out, "almanac").exists()
    assert reads_path(other, "field-notes").exists()
    assert out.exists()
    assert main(["forget", "--out", str(out)]) == 0
    assert "no reads kept" in capsys.readouterr().out


def test_forget_needs_a_store(capsys):
    assert main(["forget"]) == 2
