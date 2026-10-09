# UI primitives

`ml-ui` is a set of custom elements and one stylesheet that any page can load: the daemon's
own page, the graph page, or another application's page (Preact, React, plain HTML). Every
element is hand-written ES, shadow DOM, no build step, no global state. Colours, radii and
fonts come from CSS custom properties, so a host themes them by overriding variables.

## Assets

| What | Where |
| --- | --- |
| Folder on disk | `poolhouse.ui.assets_dir()` returns a `pathlib.Path` |
| Served by the daemon | `/ui/ml-ui/<file>`, for example `/ui/ml-ui/ml-ui.css` and `/ui/ml-ui/ml-ui.js` |
| Gallery of every state | `/ui/gallery` (the file is `gallery.html` in the same folder) |
| Python side of the verdicts | `poolhouse.ui.verdict` (`verdict_of`, `THRESHOLDS`, `LABELS`) |

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

`poolhouse.serve.fit` words its outcomes as it likes; the mapping to the vocabulary above
is `fits` to `green`, `tight` to `yellow`, `does not fit` to `red`, unknown to `none`.

### `<ml-chip>`

A verdict or status label. The text is the `label` attribute or the element's text.

| Name | Meaning |
| --- | --- |
| `verdict` | `green`, `yellow`, `red`, `none`; absent gives a plain neutral chip with no icon |
| `label` | Text; defaults to the verdict word (OK, Tight, Over, Unknown) |
| `icon-only` | Hides the word visually (kept for screen readers and as the tooltip) |

### `<ml-progress>`

| Name | Meaning |
| --- | --- |
| `label` | What is in flight |
| `done`, `total` | Counts or bytes; the fraction, percentage, rate and ETA derive from them |
| `value` | A 0 to 1 fraction when there is no `done`/`total` |
| `format`, `unit` | As on `ml-meter` (`bytes` for downloads) |
| `rate` | Units per second; shown, and gives the ETA with `done` and `total` |
| `eta` | Seconds left, overriding the derived one |
| `state` | `running` (default), `paused`, `done`, `failed` |
| `indeterminate` | Forced when there is no fraction |
| `cancelable` | Shows a Cancel button while running or paused; `cancel-label` renames it |
| `message` | A line under the bar; `role="alert"` when `state="failed"` |

Event `ml-cancel` (no detail). The track is `role="progressbar"`; an indeterminate one has
no `aria-valuenow`.

### `<ml-stepper>`

A wizard shell. Each step's content is a child with `slot="step-<id>"`.

| Name | Meaning |
| --- | --- |
| `steps` | Property: `[{id, title}]` |
| `current` | Zero-based index |
| `validate` | Property: `(index) => true \| string \| Promise`; a string is shown as an alert and the step stays |
| `controls` | `none` hides Back/Next; the application drives `current` and the bar is not clickable |
| `back-label`, `next-label`, `finish-label` | Button text |
| `compact` | Hides step titles (the bar stays) |
| `state` | Property, read and write: `{current, completed}`, for resuming a wizard |

Events: `ml-before-next` (cancelable; `detail.from`, `detail.to`, `detail.id`), `ml-step`
(`detail.from`, `detail.to`, `detail.id`, `detail.state`), `ml-finish` (`detail.state`).
Completed steps are reachable from the bar; ArrowLeft, ArrowRight, Home and End move
between them; the current step carries `aria-current="step"`.

### `<ml-sheet>`

A modal dialog on the native `<dialog>`, so focus is trapped, Escape works, `aria-modal`
is implied and focus returns to the opener.

| Name | Meaning |
| --- | --- |
| `open` | Boolean attribute; set it to show, remove it to hide |
| `heading` | The title, which labels the dialog |
| `dismissible` | `false` disables Escape, the backdrop click and the close button |
| `close-label` | Accessible name of the close button |

Children are the body; a child with `slot="footer"` goes in the footer row. Event
`ml-close` (cancelable; `detail.reason` is `escape`, `backdrop`, `close-button`, `api` or
what the caller passed to `sheet.close(reason)`). The sheet removes `open` after an
uncancelled `ml-close`.

### `<ml-table>`

| Name | Meaning |
| --- | --- |
| `columns` | Property: `[{key, label, kind?, sortable?, unit?, digits?}]`; `kind` is `text`, `mono`, `number`, `bytes`, `verdict`, `name` |
| `rows` | Property: array of records; `id` identifies a row |
| `label` | Caption |
| `empty`, `loading`, `error` | Text, flag and text for the three non-row states |
| `selectable` | Rows take focus and Enter or click; `selected` is the chosen `id` |
| `sort-key`, `sort-dir` | `ascending` or `descending` |

A `verdict` column renders a dot with an icon and the word; `row[key + "Text"]` replaces
the word. A `name` column renders `row[key]` with `row[key + "Sub"]` under it. Events:
`ml-sort` (`detail.key`, `detail.direction`), `ml-row` (`detail.id`, `detail.row`).

### `<ml-sparkline>`

`values` (property, numbers; non-finite entries are dropped), `width`, `height`, `min`,
`max`, `verdict` (colours the line), `label` (accessible name, with latest, low and high),
`format`, `unit`, `area`. No data draws a dashed line.

### Form controls

All four share `label`, `hint`, `error`, `name`, `disabled`, take part in a surrounding
`<form>` through `ElementInternals`, and fire `input` and `change` from the host with the
value in `detail`.

| Element | Own attributes | `detail` |
| --- | --- | --- |
| `<ml-select>` | `value`, `placeholder`, `options` property `[{value, label, disabled?}]` | `{value}` |
| `<ml-slider>` | `value`, `min`, `max`, `step`, `format`, `unit`, `value-text`, `marks` property `[{value, label?}]`, `stops` property (a list of values to step through; `value` is the stop, not the index) | `{value}` |
| `<ml-toggle>` | `checked`, `on-label`, `off-label` (`role="switch"`) | `{checked}` |
| `<ml-path>` | `value`, `placeholder`, `browse` (button text; empty hides it), `verdict`, `verdict-text` | `{value}`; the button fires `ml-browse` |

### `<ml-toaster>`

Place one in the page. `toaster.show({message, title?, verdict?, timeout?, action?})`
returns an id; `toaster.dismiss(id)`. Red toasts go to an assertive live region and stay
until dismissed, the rest use a polite one and leave after `timeout` ms (6000), pausing
while hovered or focused. Events `ml-action` and `ml-dismiss` (`detail.id`). `position`
is `top` or the default bottom.

## Embedding the primitives

Plain HTML:

```html
<link rel="stylesheet" href="/ml-ui/ml-ui.css">
<script type="module" src="/ml-ui/ml-ui.js"></script>
<ml-meter id="vram" label="Video memory" format="bytes" capacity="24000000000"></ml-meter>
<script type="module">
  document.getElementById("vram").segments = [
    { label: "weights", value: 14e9 }, { label: "cache", value: 5e9 }];
</script>
```

Preact (or React): import the module once, set object properties from a ref, and listen to
`ml-*` events with `addEventListener`; JSX attributes only carry strings.

```tsx
import "/ml-ui/ml-ui.js";
import { useEffect, useRef } from "preact/hooks";

export function VramMeter({ room, parts, onCancel }) {
  const el = useRef<HTMLElement & { segments?: unknown }>(null);
  useEffect(() => { if (el.current) el.current.segments = parts; }, [parts]);
  useEffect(() => {
    const stop = () => onCancel?.();
    el.current?.addEventListener("ml-cancel", stop);
    return () => el.current?.removeEventListener("ml-cancel", stop);
  }, [onCancel]);
  return <ml-meter ref={el} label="Video memory" format="bytes" capacity={room} />;
}
```

With TypeScript, declare the tags once:
`declare module "preact" { namespace JSX { interface IntrinsicElements { "ml-meter": any; } } }`.
Preact sets a prop that exists on the element as a property, so `segments={parts}` also
works; React 19 does the same, React 18 needs the ref.

Theming: override tokens on any ancestor, or on `:root` after `ml-ui.css`. An application
with its own variables maps them once:

```css
:root {
  --ml-bg: var(--app-background);   --ml-surface: var(--app-panel);
  --ml-text: var(--app-ink);        --ml-muted: var(--app-ink-secondary);
  --ml-accent: var(--app-brand);    --ml-accent-ink: var(--app-brand-text);
  --ml-green: var(--app-ok);        --ml-yellow: var(--app-warn);   --ml-red: var(--app-error);
  --ml-green-ink: var(--app-ok-text); --ml-yellow-ink: var(--app-warn-text);
  --ml-red-ink: var(--app-error-text);
}
```

Keep the contrast pairs in the Tokens table when overriding: the `-ink` tokens must reach
4.5:1 on a tint of their fill, the fills 3:1 on the surface.

## Verdicts

`verdict.json` holds the definition: `yellow_at` 0.8 and `red_at` 0.95 of capacity, and the
words. `poolhouse.ui.verdict.verdict_of(used, capacity)` and the script's `verdictOf` give
the same answer, and `tests/test_ui_primitives.py` fails if the two drift. Anything that
decides a verdict by other means (a fit estimate with its own cutoffs) passes the result in
the `verdict` attribute; the mapping from an estimator's words is `fits` to `green`, `tight`
to `yellow`, `does not fit` to `red`, unknown to `none`.

## What the interface already had

| Pattern | Where it lived | Now |
| --- | --- | --- |
| Resource bars (`.meter`, `.track`, `.fill`) for memory, processors, video memory, GPU | `fleet/web/components/cluster-view.html`, `style.css` | `ml-meter` in the cluster cards |
| Slot pips for job slots | `cluster-view.html` (`.slots`, `.slot`) | unchanged, specific to the cluster card |
| Download bar | `models-view.html` `bar()` | `ml-progress` |
| Chips and badges (`.label`, `.badge`, temperature `.warm`/`.hot`) | `style.css`, `cluster-view.html` | unchanged for vendor badges and telemetry readings; verdict chips are `ml-chip` |
| Wizard progress dots (`.steps`) | `first-run.html`, `style.css` | `ml-stepper` with `controls="none"` |
| Close question (`.sheet`) | `close-sheet.html`, `style.css` | `ml-sheet` |
| Context slider with a readout | `fleet-model.html` `contextPicker`, used by `first-run.html` and `settings-view.html` | `ml-slider` with `stops` |
| Option rows with radio or checkbox (`label.opt`) | `settings-view.html`, `first-run.html`, `models-view.html` | unchanged |
| Segmented toggle (`.toggle`) | `fit-view.html`, `rates-view.html` | unchanged |
| Line charts with hover and zoom (hand-built SVG) | `fit-charts.html`, `rates-view.html` | unchanged, specific to the fit screens |
| Fit table (`table.fit`) | `fit-view.html` | unchanged |
| Rows (`.row`) for models, clusters, downloads | `models-view.html`, `first-run.html` | unchanged |
| Notices (`.err`, `.ok`, `.hint`, `.spin`) | `style.css` | unchanged |
| Navigation tabs | `fleet-nav.html` | unchanged |
| Toasts, sparklines | not present | `ml-toaster`, `ml-sparkline` |
| Graph page palette (`--ink`, `--surface`, `--k-*`) | `graph/web/shell.html`, its own tokens | unchanged; a second token set that does not use ml-ui yet |

Components of the daemon page are light-DOM custom elements assembled by
`poolhouse.ui.assemble` into one HTML string with inline scripts; `ml-ui` is the part
meant to be loaded by other pages, so it ships as separate files instead.
