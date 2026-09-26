"""Drives a running bundle's page in headless Chromium: first run, sign-in, every tab."""

from __future__ import annotations

import argparse
import sys

from playwright.sync_api import Page, sync_playwright

TABS = ["Chat", "Cluster", "Fit", "Models", "Settings"]


def continue_(page: Page, heading: str) -> None:
    """Waits for the wizard step titled ``heading`` and presses its Continue."""
    page.wait_for_selector(f"#first-run h1:has-text('{heading}')")
    page.click("#first-run button:text-is('Continue')")


def first_run(page: Page, name: str, passphrase: str, group: str) -> None:
    """Walks the setup wizard from the first screen to the joined cluster."""
    page.wait_for_selector("#first-run:not([hidden]) h1:text-is('Set up this machine')")
    page.fill("#n", name)
    page.click("#first-run button:text-is('Continue')")
    page.wait_for_selector("#first-run h1:text-is('Clusters')")
    page.fill("#p1", passphrase)
    page.fill("#g", group)
    page.click("#first-run button:text-is('Join')")
    page.wait_for_selector(f"#first-run .row b:text-is('{group}')")
    page.click("#first-run button:text-is('Continue')")
    continue_(page, "What should this machine do?")
    continue_(page, "When should it start?")
    page.wait_for_selector("#first-run h1:has-text('What should it be able to do?')")
    page.click("#first-run button:text-is('Not now')")
    continue_(page, "When you are not using it")
    page.wait_for_selector(f"#first-run h1:has-text('{group}')")
    page.click("#first-run button:text-is('Open the cluster')")


def sign_in(page: Page, passphrase: str) -> None:
    """Types the passphrase if the page asks for it, and waits for the cluster screen."""
    page.wait_for_selector("#signin:not([hidden]), #cluster:not([hidden])")
    if page.locator("#signin:not([hidden])").count():
        page.fill("#p", passphrase)
        page.click("#signin-go")
    page.wait_for_selector("#cluster:not([hidden])")


def every_tab(page: Page) -> None:
    """Opens each tab and waits for its screen to show."""
    for label in TABS:
        page.click(f"nav.tabs a:text-is('{label}')")
        page.wait_for_selector(f"#{label.lower()}:not([hidden])")
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
            page.goto(f"{args.base}/ui/")
            first_run(page, "ci-runner", args.passphrase, args.group)
            sign_in(page, args.passphrase)
            every_tab(page)
        finally:
            if args.screenshot:
                page.screenshot(path=args.screenshot, full_page=True)
            browser.close()
    for line in errors:
        print(line, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
