# UI primitives

`ml-ui` is a set of custom elements and one stylesheet that any page can load: the daemon's
own page, the graph page, or another application's page (Preact, React, plain HTML). Every
element is hand-written ES, shadow DOM, no build step, no global state. Colours, radii and
fonts come from CSS custom properties, so a host themes them by overriding variables.

## Assets

| What | Where |
| --- | --- |
| Folder on disk | `ml_stack.ui.assets_dir()` returns a `pathlib.Path` |
| Served by the daemon | `/ui/ml-ui/<file>`, for example `/ui/ml-ui/ml-ui.css` and `/ui/ml-ui/ml-ui.js` |
| Gallery of every state | `/ui/gallery` (the file is `gallery.html` in the same folder) |
| Python side of the verdicts | `ml_stack.ui.verdict` (`verdict_of`, `THRESHOLDS`, `LABELS`) |

An embedding application copies or serves the folder and adds two tags:

```html
<link rel="stylesheet" href="/ml-ui/ml-ui.css">
<script type="module" src="/ml-ui/ml-ui.js"></script>
```

`ml-ui.js` imports its sibling modules by relative path, so the folder is served as it is,
flat. Stylesheets inside the shadow roots are constructed with `adoptedStyleSheets` and
dynamic values are set through the CSSOM, so a page with `style-src 'self'` and
`script-src 'self'` runs them with no `unsafe-inline`. Nothing here uses `innerHTML`,
`eval`, `document.write` or an inline handler: every string reaches the page through
`textContent` or `setAttribute`.

## Conventions

* An attribute is a string; the matching property has the typed value. Setting either
  re-renders. Object and array values (`segments`, `rows`, `steps`) are properties only.
* A property set before the element is defined (a framework creating the node first) is
  picked up when the element upgrades.
* Events are `CustomEvent`s named `ml-*`, bubbling and composed, with the data in `detail`.
  A form control also fires the native `input` and `change`.
* `hidden` and `disabled` behave as they do on a native element.
* Verdicts are the four words `green`, `yellow`, `red`, `none`. A verdict is never shown by
  colour alone: it carries an icon (check, exclamation, cross, dash) and a word.
* Motion respects `prefers-reduced-motion`.

## Tokens

Defined in `ml-ui.css`, dark by default, light under `prefers-color-scheme: light`, and
forced with `data-theme="dark"` or `data-theme="light"` on `<html>` or any ancestor.

| Token | Use |
| --- | --- |
| `--ml-bg`, `--ml-surface`, `--ml-sunken` | page, raised card, inset track |
| `--ml-line`, `--ml-line-strong` | hairlines, control borders (at least 3:1 on the surface) |
| `--ml-text`, `--ml-muted` | body and secondary text (both at least 4.5:1 on every surface) |
| `--ml-accent`, `--ml-accent-ink`, `--ml-accent-dim` | fills, text on a surface, selected backgrounds |
| `--ml-green`, `--ml-yellow`, `--ml-red` | verdict fills and strokes (at least 3:1 on the surface) |
| `--ml-green-ink`, `--ml-yellow-ink`, `--ml-red-ink` | verdict text (at least 4.5:1 on a tinted chip) |
| `--ml-series-1` to `--ml-series-6` | meter segment and chart series colours |
| `--ml-focus` | focus ring |
| `--ml-radius`, `--ml-font`, `--ml-mono` | shape and type |

## Elements

Filled in as each element lands; the sections below are the contract.

### `<ml-meter>`

A resource bar: segments of a capacity, a caption, a verdict.

| Name | Kind | Meaning |
| --- | --- | --- |
| `label` | attr | Left caption: what the bar measures |
| `capacity` | attr, number | The full width; segments past it overflow and mark the end of the capacity |
| `value` | attr, number | One unlabelled segment; ignored when `segments` is set |
| `segments` | property | `[{label, value, tone?}]`, drawn in order; `tone` is `1` to `6` (a series colour) |
| `format` | attr | `number` (default), `bytes` (decimal: kB, MB, GB), `bytes-iec` (KiB, MiB, GiB), `percent` |
| `unit` | attr | Suffix for `number`, for example `GB` |
| `digits` | attr | Decimals, default 1 |
| `verdict` | attr | `green`, `yellow`, `red`, `none`; when absent it is computed from `value` or the segment sum over `capacity` |
| `yellow-at`, `red-at` | attr | Fractions of capacity that turn the computed verdict yellow and red (0.8 and 0.95) |
| `caption` | attr | Replaces the right-hand text (default `12.0 GB of 24.0 GB`) |
| `verdict-text` | attr | Replaces the word beside the icon |
| `compact` | attr | A green verdict shows its icon only; yellow and red keep the word |
| `legend` | attr | `auto` (default: shown with more than one segment), `show`, `hide` |

Accessibility: the track is `role="meter"` with `aria-valuemin`, `aria-valuemax`,
`aria-valuenow` and an `aria-valuetext` that includes the verdict word.

#### Feeding it an estimate

An estimator that returns a breakdown in bytes, a room in bytes and a verdict maps one to
one:

```js
meter.format = "bytes";
meter.capacity = est.room;
meter.segments = est.breakdown.map(({ name, bytes }) => ({ label: name, value: bytes }));
meter.verdict = est.verdict;           // "green" | "yellow" | "red"
```

`ml_stack.serve.fit` words its outcomes as it likes; the mapping to the vocabulary above
is `fits` to `green`, `tight` to `yellow`, `does not fit` to `red`, unknown to `none`.
