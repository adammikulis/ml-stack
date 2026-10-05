"""Owner invite controls expose creation, scoped QR display, revoke and paste joining."""

import time

import pytest
import test_fleet_page as fleet_page
from playwright.sync_api import expect

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
open_page = fleet_page.open_page
no_release_lookup = fleet_page.no_release_lookup
pytestmark = pytest.mark.slow


def test_owner_creates_copies_revokes_and_expires_scoped_invitation(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie)
    panel = page.locator("#cluster fleet-invites")
    expect(panel.get_by_role("button", name="Create invite", exact=True)).to_be_visible()
    calls = []
    expires = [time.time() + 120]

    def answer(route):
        request = route.request
        calls.append((request.method, request.post_data_json))
        if request.method == "DELETE":
            route.fulfill(json={"revoked": True})
        else:
            route.fulfill(json={"id": "fixture", "invite": "ml-stack://enroll?data=opaque-fixture",
                "address": "https://fixture.invalid", "expires": expires[0],
                "qr": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnLz4="})

    page.route("**/ui/fleet/invites", answer)
    page.evaluate("Object.defineProperty(navigator, 'clipboard', {value: {writeText: async text => {window.copiedInvite = text;}}})")
    panel.get_by_role("button", name="Create invite", exact=True).click()
    field = panel.get_by_label("Created invitation", exact=True)
    expect(field).to_have_value("ml-stack://enroll?data=opaque-fixture")
    expect(panel.get_by_role("img", name="Invitation QR code")).to_be_visible()
    assert calls[0][1] == {"group": "home", "kind": "computer"}
    page.evaluate("document.querySelector('cluster-view').draw()")
    expect(field).to_have_value("ml-stack://enroll?data=opaque-fixture")
    panel.get_by_role("button", name="Copy invite", exact=True).click()
    page.wait_for_function("window.copiedInvite === 'ml-stack://enroll?data=opaque-fixture'")
    panel.get_by_role("button", name="Revoke invite", exact=True).click()
    expect(panel).to_contain_text("Invite revoked.")
    assert calls[-1] == ("DELETE", {"id": "fixture"})
    expect(panel.get_by_role("button", name="Create invite", exact=True)).to_be_enabled()
    expires[0] = time.time() - 1
    panel.get_by_role("button", name="Create invite", exact=True).click()
    expect(panel).to_contain_text("This invite has expired.")
    expect(field).to_have_value("")
    expect(panel.get_by_role("button", name="Copy invite", exact=True)).to_be_disabled()
    expect(panel.get_by_role("img", name="Invitation QR code")).to_have_count(0)
    assert calls[-1][1]["kind"] == "computer"
    expires[0] = time.time() + 120
    panel.get_by_role("button", name="Create invite", exact=True).click()
    expect(field).to_have_value("ml-stack://enroll?data=opaque-fixture")
    page.evaluate("window.fleetModel.go('models')")
    expect(field).to_have_count(0)
    expect(panel.get_by_label("Invitation", exact=True)).to_have_value("")
    assert panel.get_by_role("img", name="Invitation QR code").count() == 0
    page.evaluate("window.fleetModel.go('cluster')")
    panel.get_by_role("button", name="Create invite", exact=True).click()
    expect(field).to_have_value("ml-stack://enroll?data=opaque-fixture")
    page.locator("#nav-signout").click()
    expect(page.locator("#signin")).to_be_visible()
    expect(field).to_have_count(0)
    expect(panel.get_by_label("Invitation", exact=True)).to_have_value("")
    assert not errors


def test_invite_join_refuses_expired_then_clears_success_without_leaking_to_storage(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie)
    panel = page.locator("#cluster fleet-invites")
    button = panel.get_by_role("button", name="Join with invite", exact=True)
    expect(button).to_be_disabled()
    invite = panel.get_by_label("Invitation", exact=True)
    invite.fill("ml-stack://enroll?data=opaque-fixture")
    asked = []
    succeeds = [False]

    def answer(route):
        asked.append(route.request.post_data_json)
        route.fulfill(status=200 if succeeds[0] else 410, json={"ok": True, "group": "home"}
                      if succeeds[0] else {"error": "This invite has expired."})

    page.route("**/ui/fleet/join-invite", answer)
    button.click()
    expect(panel).to_contain_text("This invite has expired.")
    expect(invite).to_have_value("ml-stack://enroll?data=opaque-fixture")
    expect(button).to_be_enabled()
    succeeds[0] = True
    button.click()
    expect(invite).to_have_value("")
    assert asked == [{"invite": "ml-stack://enroll?data=opaque-fixture"}] * 2
    assert page.evaluate("JSON.stringify(localStorage) + JSON.stringify(sessionStorage)").find("opaque-fixture") == -1
    assert not errors


def test_first_run_has_join_invite_before_cluster_membership(daemon, open_page):
    page, errors = open_page(daemon)
    page.locator("#first-run").get_by_role("button", name="Continue", exact=True).click()
    panel = page.locator("#first-run fleet-invites")
    expect(panel.get_by_label("Invitation", exact=True)).to_be_visible()
    expect(panel.get_by_role("button", name="Join with invite", exact=True)).to_be_visible()
    expect(panel.get_by_role("button", name="Create invite", exact=True)).to_have_count(0)
    assert not errors
