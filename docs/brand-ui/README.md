# Poolhouse UI

The page carries the Poolhouse name: a wordmark and icon in the rail, the
page title and favicon (`src/poolhouse/fleet/web/poolhouse.svg`, a pool with a poolhouse beside it,
also usable as an app icon), and the product name in the page's own text.

## Vocabulary

One switch changes the words, never the layout: `professional` (the default; plain words such as
Devices, Projects, Settings, Join, Revoke, Pool) or `friendly` (home and poolhouse labels, headings
and onboarding copy). Friendly wording never replaces a command, a flag, an error code or an API
name, and the plain word stays beside it: a tooltip on every changed control and a visible hint
beside every changed heading.

Set it, first match wins:

1. `?vocab=friendly` or `?vocab=professional` in the address (also remembered in the browser).
2. Settings, Developer: flips it live with no reload, kept in this browser's local storage.
3. `POOLHOUSE_UI_VOCAB=friendly` in the server's environment, the default a browser with neither of
   the above gets.
4. `professional`.

The strings are in `src/poolhouse/fleet/vocabulary_strings.py`, one id with a plain and a friendly
wording each.

## Screenshots

`poolhouse-<vocabulary>-<screen>-<desktop|phone>.png` for the setup wizard, chat, devices, projects
and Settings (Developer). The devices, the shared projects and the board messages are invented
samples. Regenerate with `POOLHOUSE_UI_SHOTS=1 scripts/test all tests/test_poolhouse_shots.py`.
