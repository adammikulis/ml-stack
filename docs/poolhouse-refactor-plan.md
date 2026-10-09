# Poolhouse refactor and rename plan

Status: prework is in progress. Structural refactoring has not started; the product rename to Poolhouse is done.

## Sequence

1. Finish outstanding work, integrate reviewed batches, sync development, install the resulting
   builds and verify the original behavior on the available Mac and Windows/WSL hardware.
2. Refactor structure and contracts in small independently reviewed batches, keeping the
   installed system usable and preserving active work.
3. Rename the product, package and public vocabulary together, then build, install and exercise
   Poolhouse on each available device.

Source changes, commits and unit tests are checkpoints. A fix is complete when it is activated
and verified in the setup the owner uses. The prework must finish before structural work starts.

## Prework exit criteria

- Trusted development devices automatically discover, connect and share project coordination;
  manual pairing remains available. Verify cross-device task execution against the actual model.
- Claude and other harnesses register and reach the shared board automatically in development
  mode, without a person-level credential obstacle.
- Inbox notifications reach the owning session. New board tasks notify the current coordinator;
  coordinator selection follows measured capability and preserves main/subagent eligibility.
- Agent names are distinct and model-based. Every profile records device provenance, exact
  model, model family, harness and parent/subagent identity. Codex is a harness, not a model.
- Setup includes full UI onboarding and install/repair actions. Downloads run in the background;
  only features whose prerequisites are missing remain unavailable.
- Runtime replacement and forced restart retain jobs, checkpoints and history. Preserve the
  current Qwen model and work; do not evict them to complete unrelated integration.
- Land and activate reviewed UI, hook, workspace, runtime and transport work. Required affected
  and combined checks must pass honestly; report unavailable hardware and failed guards.
- Account for original commits and unique files before removing completed worktrees. Keep
  development synchronized throughout integration, rather than accumulating ready branches.

## Structural scope

The entire codebase is in scope: Python packages, UI, CLI, harness hooks, transport, runtime,
configuration, build/install paths, tests, documentation and contracts.

- Define explicit boundaries for device discovery/enrollment, pool membership, project boards,
  agent registration, job execution and runtime lifecycle.
- Give each contract, configuration value and state representation one authoritative owner.
  Remove duplicate implementations and divergent defaults; update producers and consumers
  together. Evaluate libraries where they simplify and improve the implementation.
- Separate model identity from harness identity and presentation from authorization. Keep
  authenticated provenance and independent review intact across LAN and Tailscale devices.
- Store project work evidence and reputation in one authoritative database, with views for
  exact models, model families, agents and devices. Define project/pool scope and deliberate
  sharing/aggregation across projects without losing original provenance.
- Reconcile archive and live-state contracts, durable jobs, restart recovery, subscription
  cursors and source/runtime identity before replacing their storage or transport.
- Keep admission, fixture resources, owned subprocess cleanup and artifact evidence explicit
  so integration and tests remain usable without broad permission grants or weakened checks.

## Naming scope

- Done (2026-10-09): the product, package and identifiers became `poolhouse` in
  source, imports, console commands, distribution metadata, UI, configuration, environment variables,
  runtime paths, hooks, build artifacts, documentation and tests.
- Replace domain vocabulary `cluster` and `fleet` with `pool`. A device joins a pool; a project
  has its own board and work. Update routes, schemas, storage and their callers together.
- Specify the treatment of existing installed state before changing paths or keys. The owner
  allows a fresh board or pool if needed; that does not authorize silently discarding jobs,
  checkpoints or attribution evidence.
- Replace obsolete contracts together rather than retaining unneeded aliases and wrappers.

## Delivery

Keep one development publisher. Workers own bounded claimed changes; independent reviewers
review exact batches. Run affected checks and combined gates, publish promptly, then activate
and verify on hardware. Full suites run in the background under maintained admission.

Ask relevant agents for information they can supply before asking the owner. Publish major
plan changes to the shared board and update this document so agents and humans use the same plan.
