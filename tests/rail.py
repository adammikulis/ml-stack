"""Reaching a screen the way a person does: the workspace rail, then the page links beside it."""

#: the rail group each screen belongs to (fleet-nav.html)
GROUP_OF = {
    "chat": "conversations", "board": "conversations",
    "models": "studio", "training": "studio", "gym": "studio", "benchmarks": "studio", "data": "studio",
    "tasks": "work", "projects": "work", "tools": "work", "knowledge": "work", "history": "work",
    "cluster": "pool", "fit": "pool",
}


def reach(page, route):
    """Click through to ``route``: Settings has a rail button of its own, every other screen is a
    page link under its group's rail button."""
    if route == "settings":
        page.click("#nav-settings")
    else:
        page.click(f'#nav-tabs a[data-workspace="{GROUP_OF[route]}"]')
        context = page.locator(f'#nav-context a[href="#{route}"]')
        if context.count() and context.get_attribute("aria-current") != "page":
            context.click()
    page.wait_for_function(f"() => window.fleetModel.route === '{route}'")
