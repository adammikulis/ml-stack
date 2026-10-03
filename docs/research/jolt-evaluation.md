# Jolt browser physics evaluation

2026-10-03, Apple-silicon macOS, JoltPhysics.js 1.1.0. The tested model is one dynamic
sphere above a static floor, stepped 300 times at 1/60 second. It reads the local
`app/node_modules/jolt-physics` installation, with no dataset or external service.

Run `npm ci --prefix app`, then `node scripts/evaluate-jolt.mjs`. To run in a browser,
serve the repository on loopback and load `/scripts/evaluate-jolt.mjs` as an ES module;
`window.joltEvaluation` resolves to the result. The script checks that the sphere settles
between 0.45 and 0.55 metres above the floor and removes its bodies before destroying
the physics world.

| Runtime | Contact check | Final height, metres |
| --- | --- | --- |
| Node | Passed | 0.480 |
| Chromium 151.0.7922.34 | Passed | 0.480 |
| Playwright WebKit 26.5 | Passed | 0.480 |

This checks WebAssembly initialization, rigid-body stepping, contact, and explicit
cleanup. It does not establish vehicle dynamics, sensor correctness, training throughput,
or compatibility with a packaged Tauri window. Playwright WebKit is a browser-engine
check, not a test of the desktop application.

Jolt is suitable for evaluating a future browser-native physical task. The initial
training environments use the physics and task definitions supplied by their specialist
libraries. Adding a Jolt task requires one authoritative physics implementation for both
live use and headless training, with a Gymnasium adapter and repeatability checks.

Source: [JoltPhysics.js](https://github.com/jrouwe/JoltPhysics.js). The evaluation uses
the single-threaded WebAssembly build and requires no cross-origin isolation headers.
