"""A native window on the interface, instead of a browser tab."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from .launch import HTTP_PORT, already_running, wait_for_health
from .settings import Settings

__all__ = ["Bridge", "main", "run_app"]

TITLE = "ml-stack"
WIDTH, HEIGHT = 1180, 820
MIN_WIDTH, MIN_HEIGHT = 900, 640

BACKGROUND = "background"
QUIT = "quit"


def _require_webview() -> Any:
    try:
        import webview
    except ImportError:
        raise SystemExit(
            "the native window needs pywebview: pip install 'ml-stack-fleet[app]'"
        ) from None
    return webview


class Bridge:
    """Called from the page when someone answers the close question."""

    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path
        self.window: Any = None

    def close_choice(self, mode: str, remember: bool) -> dict[str, Any]:
        if mode not in (BACKGROUND, QUIT):
            return {"ok": False}
        if remember:
            settings = Settings.load(self.settings_path)
            settings.on_close = mode
            settings.save(self.settings_path)
        self._act(mode)
        return {"ok": True, "mode": mode, "remembered": bool(remember)}

    def _act(self, mode: str) -> None:
        if self.window is None:
            return
        if mode == BACKGROUND:
            self.window.hide()
        else:
            self.window.destroy()


def run_app(port: int = HTTP_PORT, *, root: Path | str = "~/.ml-stack/traind",
            daemon_args: list[str] | None = None) -> int:
    """Start the daemon if it is not already up, then open a window on it."""
    webview = _require_webview()
    settings_path = Path(root).expanduser() / "settings.json"

    if already_running(port) is None:
        from .daemon import main as daemon_main

        threading.Thread(
            target=daemon_main,
            args=(["--port", str(port), "--root", str(root), *(daemon_args or [])],),
            daemon=True, name="traind",
        ).start()
        if wait_for_health(port) is None:
            raise SystemExit(f"the daemon did not start on port {port}")

    bridge = Bridge(settings_path)
    window = webview.create_window(
        TITLE, f"http://127.0.0.1:{port}/ui/",
        width=WIDTH, height=HEIGHT, min_size=(MIN_WIDTH, MIN_HEIGHT),
        background_color="#0b0f14", js_api=bridge,
    )
    bridge.window = window

    def closing() -> bool:
        saved = Settings.load(settings_path).on_close
        if saved == QUIT:
            return True
        if saved == BACKGROUND:
            window.hide()
            return False
        window.evaluate_js("window.mlStackAskOnClose && window.mlStackAskOnClose()")
        return False

    window.events.closing += closing
    webview.start()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="ml-stack-app")
    ap.add_argument("--port", type=int, default=HTTP_PORT)
    ap.add_argument("--root", default="~/.ml-stack/traind")
    known, rest = ap.parse_known_args(argv)
    return run_app(known.port, root=known.root, daemon_args=rest)


if __name__ == "__main__":
    sys.exit(main())
