"""Drives a running bundle's page in headless Chromium: first run, sign-in, every tab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

SCREENS = ["chat", "cluster", "fit", "models", "settings"]


def launch_url(page: Page, base: str) -> str:
    """The page URL carrying a one-use sign-in ticket.

    A page opened by hand is refused ("Open from its own window"); the daemon's own window and
    the owner's terminal ask for a ticket with the secret the daemon wrote under its root.
    """
    record = json.loads((Path.home() / ".ml-stack" / "traind" / "launch" / "secret.json")
                        .read_text(encoding="utf-8"))
    answer = page.request.post(
        f"{base}/ui/launch/ticket", data="{}",
        headers={"X-ML-Stack-UI": "1", "X-ML-Stack-Launch": record["secret"],
                 "Content-Type": "application/json"})
    ticket = answer.json()["ticket"]
    return f"{base}/ui/?launch_ticket={ticket}"


def press(page: Page, label: str) -> None:
    """Presses the wizard button with exactly this label."""
    page.click(f"#first-run button:text-is('{label}')")


def first_run(page: Page, name: str, passphrase: str, group: str) -> None:
    """Walks the setup wizard, whatever steps it has, until it offers to open the cluster.

    Each step is answered by its heading, so a step added or reordered in the wizard is one
    more entry here and not a silent timeout.
    """
    page.wait_for_selector("#first-run:not([hidden]) h1")
    for _ in range(20):
        heading = page.locator("#first-run h1").first.inner_text()
        if heading == "Set up this machine":
            page.fill("#n", name)
            press(page, "Continue")
        elif heading == "Clusters":
            if page.locator("#first-run button:text-is('Pair manually')").count():
                press(page, "Pair manually")
            page.select_option("#setup-cluster-action", "create")
            page.fill("#setup-cluster-name", group)
            page.fill("#setup-cluster-passphrase", passphrase)
            press(page, "Create new pool")
            page.wait_for_selector("#first-run .ok:has-text('Created')")
            press(page, "Continue")
        elif heading in ("What should this machine do?", "When you are not using it"):
            press(page, "Continue")
        elif heading == "When should it start?":
            page.check("#autostart-manual")
            press(page, "Continue")
        elif heading == "Where should downloads come from?":
            page.check("#source-internet")
            press(page, "Save and continue")
        elif heading == "What should it be able to do?":
            press(page, "Not now")
        elif heading == "Give models more memory?":
            press(page, "Skip")
        elif heading.startswith("Joined") or heading.endswith("is ready"):
            page.click("#first-run button:text-matches('^Open ')")
            return
        else:
            raise SystemExit(f"the setup wizard showed a step this smoke test does not know: {heading!r}")
        page.wait_for_function(
            "h => document.querySelector('#first-run h1')?.innerText !== h", arg=heading)


def sign_in(page: Page, passphrase: str) -> None:
    """Types the passphrase if the page asks for it, and waits for a screen of the signed-in app.

    Finishing setup leaves the page signed in on the chat screen; a page that was signed out
    asks for the passphrase first.
    """
    signed_in = "#chat:not([hidden]), #cluster:not([hidden])"
    page.wait_for_selector(f"#signin:not([hidden]), {signed_in}")
    if page.locator("#signin:not([hidden])").count():
        page.fill("#p", passphrase)
        page.click("#signin-go")
    page.wait_for_selector(signed_in)


def every_screen(page: Page) -> None:
    """Goes to each screen by its route and waits for it to show."""
    for route in SCREENS:
        page.evaluate("route => { location.hash = route; }", route)
        page.wait_for_selector(f"#{route}:not([hidden])")
        page.wait_for_load_state("networkidle")


def main() -> int:
    """Returns 0 when every screen drew with no error in the page."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="the daemon, e.g. http://127.0.0.1:8899")
    ap.add_argument("--passphrase", required=True)
    ap.add_argument("--group", default="ci")
    ap.add_argument("--screenshot", help="where to save the last screen")
    args = ap.parse_args()

    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 950})
        page.set_default_timeout(20_000)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console: {m.text}")
                if m.type == "error" and "Failed to load resource" not in m.text else None)
        # 501 is the daemon saying a feature is not in this build, which the page draws
        page.on("response", lambda r: errors.append(f"{r.status} {r.url}")
                if r.status >= 400 and r.status != 501 else None)
        try:
            page.goto(launch_url(page, args.base))
            first_run(page, "ci-runner", args.passphrase, args.group)
            sign_in(page, args.passphrase)
            every_screen(page)
        finally:
            if args.screenshot:
                page.screenshot(path=args.screenshot, full_page=True)
            browser.close()
    for line in errors:
        print(line, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
