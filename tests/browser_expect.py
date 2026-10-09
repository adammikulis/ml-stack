"""``playwright.sync_api.expect``, imported when a test calls it.

A module that imports playwright at the top fails collection on a machine without the
``scrape`` extra; every test using it takes the ``playwright`` fixture, which skips there.
"""


def expect(*args, **kwargs):
    from playwright.sync_api import expect as real

    return real(*args, **kwargs)
