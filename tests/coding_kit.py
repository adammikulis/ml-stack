"""A native subprocess fixture emitting the harness's documented JSONL contract."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SESSION = "13d70164-3caa-4e17-9201-d57511fcab62"


def native_cli() -> None:
    request = sys.stdin.read()
    home = Path(os.environ["CODEX_HOME"])
    session = home / "sessions" / "fixture.json"
    if "resume" in sys.argv and not session.exists():
        raise RuntimeError("native session history was not restored")
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text(request)
    print(json.dumps({"type": "thread.started", "thread_id": SESSION}), flush=True)
    print(json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "inspect project", "status": "completed"}}), flush=True)
    if request == "wait until cancelled":
        import time
        time.sleep(120)
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Inspected: " + request}}), flush=True)
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}}), flush=True)


def fixture_worker(root, cid, conversation, prompt, output, cancellation) -> None:
    from ml_stack import coding
    from ml_stack.workspace.coding_turns import worker
    def launch(model, role, project, **options):
        if prompt == "startup rejected":
            raise SystemExit(2)
        seat = options["seat_factory"](options["name"], Path(project), "", lambda text: None)
        try:
            with tempfile.TemporaryDirectory() as private:
                command = [sys.executable, str(Path(__file__)), *options["harness_args"]]
                return options["run_codex"](command, {**os.environ, "CODEX_HOME": private})
        finally:
            seat.revoke()
    coding.launch_coding_agent = launch
    worker(root, cid, conversation, prompt, output, cancellation)


if __name__ == "__main__":
    native_cli()
