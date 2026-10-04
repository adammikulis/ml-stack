"""scripts/compare-harnesses: the fixture, the parsers, the metrics scrape and a row, with a stand-in harness."""

import importlib.machinery
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare-harnesses"


@pytest.fixture(scope="module")
def cmp():
    loader = importlib.machinery.SourceFileLoader("compare_harnesses", str(SCRIPT))
    spec = importlib.util.spec_from_loader("compare_harnesses", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class _Metrics(BaseHTTPRequestHandler):
    processed = 0

    def do_GET(self):
        if self.path == "/bump":
            type(self).processed += 30
        body = (f"# TYPE llamacpp:prompt_tokens_total counter\nllamacpp:prompt_tokens_total {type(self).processed}\n"
                "llamacpp:tokens_predicted_total 7\nllamacpp:prompt_seconds_total 1.5\n").encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Metrics)
    _Metrics.processed = 100
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Metrics
    httpd.shutdown()


def test_the_fixture_fails_as_shipped_and_passes_once_the_bug_is_fixed(cmp):
    where = cmp.workdir()
    try:
        assert cmp.passes(where) is False
        calc = where / "calc.py"
        calc.write_text(calc.read_text().replace("(len(values) + 1)", "len(values)"))
        assert cmp.passes(where) is True
    finally:
        shutil.rmtree(where.parent, ignore_errors=True)


def test_the_parsers_count_tool_calls_and_tokens(cmp):
    claude = "\n".join(json.dumps(e) for e in (
        {"type": "assistant", "message": {"content": [{"type": "tool_use"}, {"type": "text"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use"}]}},
        {"type": "result", "usage": {"input_tokens": 10, "cache_read_input_tokens": 90}}))
    assert cmp.parse_claude(claude + "\nnot json") == {"tool_calls": 2, "input_tokens": 100, "cached_tokens": 90}
    codex = "\n".join(json.dumps(e) for e in (
        {"type": "item.completed", "item": {"type": "command_execution"}},
        {"type": "item.completed", "item": {"type": "agent_message"}},
        {"type": "turn.completed", "usage": {"input_tokens": 50, "cached_input_tokens": 40}}))
    assert cmp.parse_codex(codex) == {"tool_calls": 1, "input_tokens": 50, "cached_tokens": 40}


def test_a_row_records_what_the_server_processed_while_the_harness_ran(cmp, server, tmp_path):
    url, _ = server
    assert cmp.scrape(url) == {"processed": 100.0, "generated": 7.0, "prompt_s": 1.5}
    fake = tmp_path / "harness.py"
    fake.write_text(
        "import json, pathlib, urllib.request\n"
        "p = pathlib.Path('calc.py'); p.write_text(p.read_text().replace('(len(values) + 1)', 'len(values)'))\n"
        f"urllib.request.urlopen('{url}/bump')\n"
        "print(json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'tool_use'}]}}))\n"
        "print(json.dumps({'type': 'result', 'usage': {'input_tokens': 20, 'cache_read_input_tokens': 80}}))\n")

    work = cmp.workdir()
    try:
        before = cmp.scrape(url)
        row = cmp.measure("fake", [sys.executable, str(fake)], cmp.parse_claude, url, work)
    finally:
        shutil.rmtree(work.parent, ignore_errors=True)
    assert row["pass"] is True and row["tool_calls"] == 1 and row["input_tokens"] == 100
    assert row["processed_tokens"] == 30 and before["processed"] == 100.0
    assert row["cache_hit_ratio"] == 0.7


def test_a_run_that_does_not_fix_the_bug_is_recorded_as_failed(cmp, server, tmp_path):
    url, _ = server
    work = cmp.workdir()
    try:
        row = cmp.measure("idle", [sys.executable, "-c", "print('nothing')"], cmp.parse_claude, url, work)
    finally:
        shutil.rmtree(work.parent, ignore_errors=True)
    assert row["pass"] is False and row["tool_calls"] == 0 and row["cache_hit_ratio"] is None


def test_the_report_and_the_refusals(cmp):
    rows = [{"harness": "a", "pass": True, "wall_s": 1.0}, {"harness": "b", "note": "skipped: not installed"}]
    text = cmp.report(rows, {"date": "2026-01-01", "command": "c", "model": "m", "serving": "s"})
    assert "| a | True | 1.0 |" in text and "skipped: not installed" in text and "`c`" in text
    for argv in (["--model", "Qwen3.8-Flash-Next-UD-Q4_K_XL"], []):
        assert cmp.main(argv) == 2
    done = subprocess.run([sys.executable, str(SCRIPT), "--on", "http://127.0.0.1:1", "--dry-run"],
                          capture_output=True, text=True, check=False,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SCRIPT.parent.parent / "src")})
    assert done.returncode == 0 and "ml_stack.claude" in done.stdout and "ml_stack.codex" in done.stdout
