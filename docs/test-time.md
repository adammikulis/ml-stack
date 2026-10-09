# Test and hook time (2026-10-09)

Commit measured: ae9dcca0 (0.2dev tip at start). Machine: 16 logical CPUs, 128 GB, macOS, no GPU lease held by me.
Every number below carries its command and the machine load when it was taken. The machine was NOT quiet:
another agent's `scripts/test full --no-reuse` (job 20261009-094215-c2f8) was admitted at 09:42, the same minute as mine,
so for most of my run two 16-worker pools shared 16 cores.

## 0. Load log of the full-tier run
Command: `PYTHONPATH=<scratch> PHASE_OUT=<scratch>/ph/p PYTEST_ADDOPTS="-p phase_timing" scripts/test full --no-reuse`
(plugin `phase_timing.py` writes setup/call/teardown per test and per-fixture setup seconds; it only observes).
- Start 09:42, load avg (1/5/15) 5.5/5.6/9.7 at launch; 30.1/19.2/14.8 at +3 min; 30.7/26.9/19.7 at +8 min; 22.4/26.6/22.0 at +13 min; 12.0/17.6/19.4 at end (09:59). Mean 1-minute load over the run about 25 on 16 CPUs. Memory pressure: 96-97 percent free throughout (not a factor). Own workers: 16, plus 4 refill workers started at +154 s.
- Result: 13588 tests, 144 failed, 114 errors, 165 skipped, wall 876.8 s. The failures cluster in files that depend on timing or on the working tree (test_project_board_ui 31 errors, test_workspace_remote 28, test_device_agent 16, test_remote_workers 15, test_worker_completion 13, test_model_tiers 12, test_workspace_quickstart 10, test_workspace_live 7 "wakes within milliseconds"). I did not triage which are load and which are real; the previous full job 20261009-091503-dd42 (also at load about 10) had 126 failed/113 errors, so most are not load.
- Recorded idle baseline `tests/full-tier-time.json`: wall 183 s, cpu 1921 s, 9573 tests, 16 workers. My run is 4.8x the wall of the baseline; load is the dominant reason, but the suite also grew from 9573 to 13588 collected tests.

## 1. Where the wall goes (full tier, load about 25)
- Sum of test phases over all tests: setup 2085.9 s, call 7842.9 s, teardown 240.9 s = 10169.7 s. Divided by 16 workers = 636 s of the 877 s wall. So about 241 s (27 percent) of wall is not test phases: pool start 17 s (workers 16.3-18.1 s after submit), per-worker collection, load imbalance at the tail (one test, `test_two_hundred_waits_each_racing_one_send_never_miss`, is 236 s on its own, so the pool drains behind it), and the 4 refill workers at 154 s.
- Setup is 20.5 percent of test time (2086 s) and teardown 2.4 percent. Setup is where fixtures cost: see section 2.
- Collection: `scripts/test full --no-reuse --collect-only -q` took 44 s wall (10:01, load 17 falling to 10), through the broker with the worker pool. Collecting 13.5k tests across 770 files is paid by every worker.
- 6089 of the 13588 tests run under 50 ms and add up to only 99.7 s. The suite's time is not the many tiny tests; the top 200 tests are 37 percent (3806 s), the top 500 are 54 percent, the top 1000 are 70 percent.
- Idle history (EMA store `.git/test-history/durations.json`, mixed loads): total cpu 5081 s, wall 7778 s, so about 2700 s of per-test wall in history is waiting (sleeps, sockets, subprocess waits), not CPU.

### Imports (command: `python -c "import <m>"` timed in process, load 27-30, so inflated about 2x)
numpy 36 ms, psutil 9 ms, pytest 56 ms, playwright.sync_api 370 ms, ladybug 244 ms, mlx.core 388 ms, spacy 1105 ms, torch 5324 ms, transformers 7547 ms, presidio_analyzer 24618 ms (cold, includes spaCy and its model registry).
The `-X importtime` of what `tests/conftest.py` imports (poolhouse.testing.fakes, poolhouse.http, poolhouse.keystore, psutil, pytest): `poolhouse.testing` 214 ms cumulative, of which poolhouse.testing.fakes 129 ms, poolhouse.client 56 ms, poolhouse.http 51 ms, poolhouse.serve 40 ms, poolhouse.keystore 40 ms, poolhouse.sentinel 38 ms, pytest 30 ms, numpy 27 ms. Nothing heavy (no torch/transformers) is imported by conftest; roughly 0.25 s per worker, 16 workers = 4 s CPU total. Only 14 test files import mlx or playwright at the top (list: spec_qwen4, spec_serve, spec_decode, spec_qwen4_load, graph_engine_contract and 9 *_ui/*_browser files). So lazy imports of heavy deps would save almost nothing; import cost is not the problem.

### Fixtures by total setup seconds (function scope unless noted; from the full run)
| fixture | scope | calls | total s | mean |
|---|---|---|---|---|
| board | function | 138 | 226.4 | 1640 ms |
| kit | function | 437 | 111.8 | 256 ms |
| repo | function | 132 | 56.1 | 425 ms |
| tmp_path | function | 13437 | 55.2 | 4 ms |
| world | function | 155 | 54.2 | 350 ms |
| project | function | 140 | 54.1 | 387 ms |
| store | function | 258 | 50.0 | 194 ms |
| inactive | function | 30 | 31.2 | 1041 ms |
| setup | function | 37 | 24.5 | 661 ms |
| served | function | 172 | 20.8 | 121 ms |
| installed_metadata | module | 1 | 19.4 | 19372 ms |
| paired | function | 13 | 19.3 | 1485 ms |
| checkout | function | 33 | 18.2 | 551 ms |
| replicas | function | 6 | 13.4 | 2227 ms |
| team | function | 18 | 12.9 | 717 ms |
| stopped | function | 7 | 12.1 | 1723 ms |
| seeded_store | module | 1 | 11.5 | 11468 ms |
| long_thread | module | 1 | 10.4 | 10370 ms |
The autouse fixtures are cheap: `_no_internet` 0.3 ms and `_no_machine_binary` 0.2 ms per test (13437 tests, 4.1 s and 3.1 s total); `_real_home` session fixture 280 ms x 15 workers. Per-file fixed overhead from conftest is therefore negligible; the cost is the heavyweight function-scoped `board` (a started board daemon: 138 x 1.64 s), `kit`, `repo`/`checkout` (real git repos), `world`, `project`, `store`.
Totals: board+kit+repo+world+project+store+inactive+setup+paired+checkout = about 640 s of the 2086 s setup, at load about 25.

## 2. Slowest files (full run, this load) against their CPU seconds in the history store
Columns: n tests, setup/call/teardown seconds in my run, my total, history cpu, history wall, my/history-wall. my/history-wall about 2 is the load factor (load about 25 on 16 CPUs); values far above 2 (live 7.7, surface 4.3, land 3.8, budgets 3.7, task_integration 3.5, quickstart 3.2) are timing-sensitive files that degrade worse than the machine does.
| file | n | setup | call | tear | total | hist cpu | hist wall | ratio |
|---|---|---|---|---|---|---|---|---|
| test_workspace_local_agent | 58 | 9 | 339 | 3 | 352 | 125 | 165 | 2.1 |
| test_workspace_live | 20 | 6 | 336 | 1 | 344 | 28 | 45 | 7.7 |
| test_workspace_files | 61 | 182 | 151 | 1 | 333 | 91 | 172 | 1.9 |
| test_task_integration | 9 | 175 | 137 | 0 | 312 | 57 | 90 | 3.5 |
| test_workspace_swarm | 25 | 1 | 292 | 1 | 294 | 119 | 145 | 2.0 |
| test_task_worktree_recovery | 30 | 248 | 44 | 0 | 292 | 97 | 153 | 1.9 |
| test_workspace_quickstart | 74 | 9 | 238 | 1 | 248 | 48 | 79 | 3.2 |
| test_workspace_board | 24 | 8 | 220 | 0 | 229 | 80 | 112 | 2.1 |
| test_workspace_quiet | 17 | 4 | 213 | 2 | 219 | 87 | 103 | 2.1 |
| test_workspace_bus | 31 | 4 | 205 | 0 | 209 | 84 | 114 | 1.8 |
| test_worktree_rules | 100 | 103 | 39 | 1 | 142 | 30 | 61 | 2.3 |
| test_person_hooks | 58 | 36 | 93 | 0 | 129 | 52 | 76 | 1.7 |
| test_workspace_agent_invites | 27 | 7 | 118 | 0 | 125 | 27 | 52 | 2.4 |
| test_person_guards | 120 | 11 | 108 | 1 | 120 | 52 | 64 | 1.9 |
| test_graph_bench | 191 | 10 | 107 | 1 | 119 | 30 | 57 | 2.1 |
| test_workspace_screen_tiers | 25 | 3 | 111 | 2 | 117 | 42 | 52 | 2.3 |
| test_world_check | 36 | 1 | 111 | 0 | 113 | 34 | 52 | 2.2 |
| test_task_credit | 9 | 37 | 71 | 0 | 109 | 42 | 75 | 1.5 |
| test_serve_llamacpp | 22 | 4 | 96 | 0 | 100 | 43 | 108 | 0.9 |
| test_board_coordination | 14 | 3 | 91 | 0 | 94 | 46 | 54 | 1.8 |
| test_keystore | 42 | 2 | 90 | 0 | 92 | 29 | 47 | 1.9 |
| test_taskboard | 7 | 43 | 46 | 0 | 89 | 29 | 51 | 1.7 |
| test_peer_first | 47 | 1 | 49 | 37 | 87 | 12 | 47 | 1.9 |
| test_serve_no_bypass | 12 | 0 | 86 | 0 | 87 | 42 | 43 | 2.0 |
| test_workspace_nudge | 11 | 4 | 81 | 0 | 85 | 22 | 31 | 2.7 |
| test_land | 17 | 6 | 79 | 0 | 85 | 18 | 23 | 3.8 |
| test_serve_admission | 23 | 0 | 81 | 1 | 82 | 33 | 39 | 2.1 |
| test_workspace_coordinator | 27 | 51 | 30 | 0 | 82 | 13 | 39 | 2.1 |
| test_ingest_sources | 37 | 2 | 79 | 0 | 81 | 18 | 41 | 2.0 |
| test_workspace_local_inbox | 6 | 15 | 64 | 0 | 79 | 27 | 39 | 2.1 |
| test_workspace_surface | 11 | 1 | 77 | 0 | 78 | 10 | 18 | 4.3 |
| test_workspace_backlog | 14 | 17 | 61 | 0 | 78 | 17 | 42 | 1.9 |
| test_sandbox_seatbelt | 23 | 0 | 76 | 0 | 77 | 8 | 39 | 2.0 |
| test_ingest | 87 | 3 | 70 | 1 | 74 | 18 | 42 | 1.8 |
| test_budgets | 64 | 3 | 66 | 0 | 69 | 18 | 19 | 3.7 |
| test_peer_pairing | 26 | 43 | 14 | 11 | 68 | 24 | 39 | 1.7 |
| test_onboard_cli | 12 | 7 | 57 | 3 | 67 | 17 | 36 | 1.9 |
| test_graph_store | 53 | 1 | 65 | 0 | 67 | 8 | 32 | 2.1 |
Reading: of a typical slow file's time, about half is load (ratio about 2). The history cpu column is the load-independent floor: for the 40 files above the total is about 2000 cpu-seconds out of 7000 measured seconds. The `workspace_*`, `board_*`, `task_*`, `peer_*` files together are 3300 s of 10170 s measured (32 percent) and are the board-daemon-on-a-port cluster: they are what the in-process node would delete.

### 60 slowest tests (top of the list; full list in the scratch file analysis.txt)
1 `workspace_swarm::test_two_hundred_waits_each_racing_one_send_never_miss` 236 s call; 2-3 `workspace_live::test_a_waiting_agent_wakes_within_milliseconds_of_a_send[dm-40]` / `[board-40]` 100 s each (a latency assertion run for N waiters, 8 params in the file 20 to 100 s each); 4 `workspace_bus::test_concurrent_senders_get_one_total_order` 98 s; 5 `workspace_quickstart::test_setup_walks_through_six_steps_with_a_scripted_person` 73 s; 6 `workspace_coordinator::test_real_distribution_registration_and_two_roots_share_message` 53 s (41 s setup: builds a real wheel); 7 `budgets::test_metric_is_within_its_budget[pyright-errors]` 48 s (runs pyright); 8-9 `workspace_board::test_digests_are_bounded...` 49 s and `workspace_screen_tiers::test_normal_traffic_is_never_held_for_a_token_holder` 47 s; then seven `task_integration` tests at 30-42 s each, 15-27 s of it setup (real git repo + registered daemon); `test_land::test_infrastructure_diff_starts_one_background_full_run` 42 s; `no_data_files::test_the_tracked_tree_passes` 35 s; `keystore::test_a_burst_of_700_fresh_readers...` 33 s; `affected_scripts` 28.5 s; `net_scan::test_real_clamav...` 21 s; `graph_thread::test_recall_finds_the_fact_from_turn_one_two_hundred_turns_later` 21 s (all setup: module fixture `long_thread` 10 s); `graph_engine_contract::test_every_relation_reads_the_same_from_both_ends_after_a_write` 23 s (all setup: graph open); 12 `workspace_local_agent` tests at 20-36 s.
Top 10 tests are 852 s (8 percent), top 60 are 2250 s (22 percent). One `test_workspace_swarm` test alone sets the lower bound of the whole run's wall (236 s at this load; its history cpu is about 30-50 s).

### What dominates (static counts over tests/, 770 files, 10596 test functions; command: a regex scan, `static.py` in scratch)
- 527 `subprocess.run/Popen/check_output` call sites in 198 files; 1299 daemon/board/serve_forever references in 172 files; 564 socket/http-server references in 99 files; 247 UDP/multicast references in 23 files; 25 real `git init` sites in 16 files; 182 playwright references in 44 files (10 import it at top); 76 wheel/`build` references in 34 files; 295 TLS/cryptography references in 44 files; 769 graph-store (ladybug) references in 82 files; 344 `sleep(` calls in 118 files (321 with a literal duration; many are inside child-process script strings, so the literal sum 5597 s is NOT time slept by the tests); 1589 timeout=/wait()/join() sites in 339 files.
- Dynamic sleep/Popen/timeout counts per test: a tracer plugin (patches time.sleep, Popen.__init__, Popen.wait, Event.wait and attributes them to the running test) was started on the 36 slowest files at about 10:03, load about 28; the files ran (job `scripts/test full --no-reuse <36 files>`) but my worktree was removed before I could read the traces (see Blocked). The raw trace files are in `scratchpad/ph2/p.*` (records `"k":"trace"`: sleep_n, sleep_s, popen, git, ewait_*, pwait_*); the lead can aggregate them with the one-screen snippet in the report.

## 3. Overtesting
- Copy-paste near-duplication is small. Normalised-AST grouping of test bodies of 3+ statements and 300+ AST chars found 22 groups holding 49 tests (the biggest is 6 tests in `test_no_real_names.py` and 3 in `test_redact.py` that differ only in a string constant; they should be one parametrized test, saving nothing in time because they are cheap). No two test files share more than half of their test names. No test file imports a missing `poolhouse` module and no module-level skip marks or "parked" skips exist, so there are no dead files by that test. (Command: `dups.py` in scratch.)
- The overtesting here is structural, not copy-paste: one end-to-end route per assertion. Measured evidence: `board` fixture 138 x 1.64 s = 226 s, `kit` 437 x 0.26 s, real git `repo`/`checkout`/`project` fixtures 132+33+140 x 0.4-0.55 s, `test_task_integration` 9 tests = 312 s (about 35 s each) where each starts a registered daemon and a real git repo to check one integration rule, `test_worktree_rules` 100 tests with 103 s of setup (about 1 s each, each initialising a real repo), `test_task_worktree_recovery` 30 tests with 248 s of setup (8 s each), `test_workspace_files` 61 tests with 182 s setup.
- Parametrisation multiplying a slow setup: `workspace_live` `[dm|board]-{10,40,...}` runs the wake-latency test at several fan-outs (the 40-waiter cases are 100 s each at load, 28 s cpu for the file). `test_peer_first` 37 s teardown (47 tests). `test_graph_bench` 191 tests, 119 s.
- Tests of generated files / repo-wide scans re-run on every tier: `test_budgets` (pyright 48 s, others 3-10 s), `no_data_files::test_the_tracked_tree_passes` 35 s, `affected_scripts` 28 s, plus the structural gates that `scripts/test gate` already runs (budgets, red-team coverage, command reference, layers). Those are 150-250 s of the full run duplicated with the gate.

## 4. Hooks (this is the part the coordinator added)
Hooks timed directly (`hooktime.py`, each hook alone, staged index of a throwaway worktree). Loads: small commit measured at load 22.0/17.0/18.5 rising to 30/19/19; large merge at load 28.8/19.3/19.3 rising to 33/23/21 (another agent's full run active: 16 CPUs, so wall about 2x cpu).
Small commit = 2 staged files (1 .py, 1 .md, +4 lines). Large merge = real `git merge --no-commit --no-ff ae9dcca0` into HEAD~120, 675 files changed, 53k insertions, MERGE_HEAD present.
| hook | small commit wall / cpu | large merge wall / cpu | loads | scope |
|---|---|---|---|---|
| primary-only | 0.12 / 0.06 s | 0.10 / 0.05 s | git rev-parse only | n/a |
| no-data-files | 0.12 / 0.06 | 5.66 / 4.86 | staged names | all staged names, regex by path/extension |
| no-real-names | 20.66 / 8.53 | 0.26 / 0.11 | spaCy en_core_web_sm via Presidio (engine load 4.7 s alone), the shape rules JSON, git show of each staged file | small: reads the WHOLE staged file and runs the recogniser over the whole file, then keeps only findings on added lines (hook.py: `blob = git show :path`, `added_lines` filter after `_findings`). Merge: a merge checks only files that differ from BOTH parents (here none) so 0.26 s |
| budgets | 5.97 / 2.58 | 65.75 / 64.16 | runs every gate checker (`gates.run`) twice on two temp trees holding the staged .py files (new and HEAD) plus pyproject | only staged .py files, but all checkers run on both copies; scales with file count: 675 files = 64 s cpu |
| budgets-only-fall | 0.07 / 0.06 | 0.10 / 0.09 | budgets.json diff only | n/a |
| commit-msg (name check) | 0.30 / 0.09 | 1.94 / 1.65 (includes weakened-assertions) | reads ~/.config/pii-deny.txt and fixtures; regex per name | message only |
| weakened-assertions | 0.05 / 0.05 | 1.66 / 1.55, REFUSED | `git diff --cached -U0 -- tests`, fnmatch + regex | whole staged test diff, against first parent |
| pre-push `pushed` | 1 commit: 1.62 / 1.02 | 120 commits: 15.68 / 11.52 | `ast.parse` of every changed .py (git show each), `no-data-files --range` (0.10 / 9.25 s), `runtime-refresh`, leftover worktree scan | pushed range only |
Pre-commit chain total: small 27.2 s wall / 11.3 s cpu (no-real-names 76 percent, budgets 22 percent). Large merge 71.8 s wall (budgets 92 percent).
No hook runs ruff or pyright. The staged-file hooks do not run `scripts/test`.

### False positives, reproduced (script `fp.py`, calls `poolhouse.redact.hook._findings` with the real recogniser; tested against the tip's rules (ae9dcca0, without c5743a2f) and against the tree at c5743a2f (which is NOT an ancestor of this tip: `git merge-base --is-ancestor c5743a2f HEAD` says no))
| input | tip | with c5743a2f |
|---|---|---|
| date `<digits>` read as phone | REFUSED (phone shaped) | REFUSED |
| ISO timestamp `<digits>T09:15:<digits>:00`, job id `<digits>-dd42` | clear | clear |
| model id `<digits>` bare | REFUSED (phone) | REFUSED |
| model id `claude-sonnet-<digits>` / `<digits>` | REFUSED | clear |
| `claude-haiku-<digits>` in a sentence | REFUSED | REFUSED (the bare `<digits>` part) |
| version strings `<digits>`, `<digits>` | clear | clear |
| SVG path with `<digits>` | clear | clear |
| SVG path `<digits>` | REFUSED (phone) | clear |
| `App Attest`, `Secure Enclave`, `NaCl Box` | REFUSED as a person | clear |
| heading `4.3 Disputes` | REFUSED as a person | clear |
| lower-case `worktree path`, `console metadata` | `console metadata` REFUSED, `worktree path` clear | clear |
| Title Case `worktree path`, `console metadata` | REFUSED | STILL REFUSED |
| real person, real email, real phone (`<digits>`, `<digits>`) | refused | refused (correctly) |
Still misfiring after c5743a2f: dates written `<digits>`, bare model-id tails `<digits>`, Title Case technical phrases (`worktree path`, `console metadata`). weakened-assertions on a merge: reproduced; a clean merge of 120 commits was REFUSED for "test function removed", "6 fewer assertions", "added pytest.mark.skip" in 8 guard test files, because it diffs the index against the first parent and so judges the incoming branch's already-reviewed history as if it were this commit. `budgets` refusing on files the commit did not touch: not reproduced; it only compares staged .py files, but on a merge it counts every file the merge brought in, which are not "this commit's" changes (my large merge did pass: it added no sites).
Mechanism of the phone misfire (hook.py): `PHONE = \+?\d[\d ().-]{8,}\d` accepts any digit run with dashes; it clears only if it has two or more dots, is `YYYY-M-D`, has a 4+ digit fraction, matches a timestamp pattern, or is inside a uuid. Everything else with 7-15 digits is a phone number.

## 5. Ranked cuts (estimated seconds saved from the measurements above; "now" = can be done today, "refactor" = depends on in-process node / Rust daemon / one lease service)
1. Pre-commit no-real-names: scan only added lines (or added hunks with 3 lines of context) through spaCy instead of the whole staged file; keep the engine warm or load only `en_core_web_sm` with the parser/lemmatizer disabled (`spacy.load(..., disable=["parser","lemmatizer","attribute_ruler"])`). Small commit: 20.7 s wall / 8.5 cpu down to about 5 s (engine load 4.7 s is the floor without a warm server). Large single-file commits scale at about 35 KB/s (271 KB took 7.7 s); added-lines-only makes that independent of file size. Now.
2. budgets pre-commit on merges: skip it when MERGE_HEAD exists and the file is identical to one parent (same rule no-real-names already uses), and run only the checkers that read staged files (not all of `gates.run`) on a plain commit. Merge: 65.8 s to about 0 s (the gates run in `scripts/test gate` anyway); small commit 6 s to about 2 s. Now.
3. weakened-assertions on a merge: judge only `git diff --cc` (the conflict resolution) when MERGE_HEAD exists. Removes the false refusals of correct merges, and 1.7 s. Now.
4. In-process board node for the `board` fixture instead of a daemon on a port: 138 x 1.64 s = 226 s of setup (at load; about 100 s idle-equivalent) plus the sleeps in the 32 percent `workspace_*/board_*/task_*/peer_*` cluster (3300 s measured, about 1400 s history cpu). Refactor (in-process node / Rust node daemon).
5. Session-scoped template git repo copied (`cp -a` or `git worktree`) per test instead of `git init` + commits per test: `repo`/`checkout`/`project`/`world` fixtures are 132+33+140+155 calls, about 180 s of setup; `test_task_worktree_recovery` 248 s and `test_worktree_rules` 103 s and `test_task_integration` 175 s of setup are mostly git. A template copy costs about 20 ms. Estimated 250-350 s of measured seconds (about 100-150 s of cpu). Now.
6. Replace the wake-latency scale tests with event-based assertions and one small fan-out: `workspace_swarm` 236 s single test, `workspace_live` 8 params 200 s of 344, `workspace_bus::test_concurrent_senders...` 98 s. These dominate the tail; the run cannot finish before the 236 s test does. Keep one 40-waiter case in the slow marker. Saves about 400 s of measured test time and about 200 s of wall tail. Partly refactor (in-process node makes them microseconds), but the parametrisation cut is now.
7. Move whole-tree scans out of `full`: `budgets::pyright-errors` 48 s, `no_data_files::test_the_tracked_tree_passes` 35 s, `affected_scripts` 28 s and the other `test_budgets` metric cases (about 20 s) are re-runs of `scripts/test gate`. Mark them `gate` and run once per landing. About 130 s measured in the full tier. Now.
8. Build the wheel once per session (`workspace_coordinator` 41 s setup for its real-distribution test; 34 files reference a wheel build). Cache the built wheel under a session-scoped, content-keyed directory. About 60-100 s. Now.
9. Module-scoped fixtures for the graph and long-thread stores: `installed_metadata` 19 s, `seeded_store` 11 s, `long_thread` 10 s, `graph_engine_contract` 23 s setup are already module scope but each worker that gets one of these files rebuilds it. Build the store once per session into a shared read-only directory and open it read-only per worker (the store is ladybug and stays). About 40-60 s of cpu, and removes 3 of the 4 one-test-takes-20-s files. Now.
10. Fix the pool tail and startup: 17 s worker start plus 4 refill workers at 154 s plus the long-pole 236 s test. Sort by recorded duration (the history store already has it) so the longest tests start first (`--dist loadfile` with longest-first ordering), and run the 4 refill workers at start. Saves about 60-120 s of wall at this load. Now.
Also (not ranked): collapse the 22 duplicate-body groups (49 tests) into parametrised tests: cleanup only, under 1 s saved. A `pytest --collect-only` takes 44 s through the broker at load 17; collecting from a per-file cache (the test-reuse store already hashes files) would remove most of that for `quick`, but the full tier pays it per worker. Now.

### Hook recommendations (ranked by seconds saved and misfires removed)
1. Per commit, keep only: primary-only, no-data-files (names), budgets-only-fall, commit-msg name check (message only, 0.3 s), weakened-assertions (non-merge). Total under 1 s.
2. Make no-real-names incremental (added lines only, small spaCy pipeline, or run spaCy per push on the added range instead of per commit). Per commit keep only the exact checks: the exact-name list, email regex, and a strict phone grammar. Saves about 20 s per small commit (76 percent of the chain).
3. Replace the phone rule with a strict one: require a leading `+` and country code, or `(ddd) ddd-dddd`, or exactly `ddd-ddd-dddd` / `ddd.ddd.dddd` / `+d{1,3} d{1,4} d{3,4} d{3,4}` with a word boundary on both sides, and refuse a candidate whose neighbours are `-`, `_`, letters or digits (so a model-id tail, a month-day-year date, a grouped identifier, SVG path numbers and dated ids never qualify). This removes the remaining date, model-id and path misfires with no model.
4. Move `budgets` (all checkers, twice) to the landing gate (`scripts/test gate`, which already runs the same checkers on the whole tree) and keep per commit only `budgets-only-fall` (0.07 s). Saves 6 s per commit and 66 s per merge.
5. Run the person-name recogniser as a pre-push check over the pushed range's added lines (one engine load per push, not per commit), where 4.7 s of load is amortised over all the commits; still keep the exact-name list on every commit.
6. weakened-assertions: on a merge, judge only the combined diff; on a rebase/cherry-pick replays, only the replayed commit. Removes correct-merge refusals.
7. Title Case technical phrases (`worktree path`, `console metadata`): do not call Presidio PERSON on prose in `.md` and `src` files unless the phrase's first word is in a first-name list or the line has a person-context cue; or only report a PERSON hit when at least one token is in the US census forename/surname lists. Removes the capitalised-term misfires without a case-by-case allowlist.
8. pre-push: `pushed` 15.7 s for 120 commits (`ast.parse` of every changed .py and `no-data-files --range` 9.3 s): both can run on the net diff (`base..tip`) instead of per-commit lists (already the case for diff names) and use the parsed-AST cache the budgets already keep.

## Blocked
My worktree directory `.../.claude/worktrees/agent-ab<digits>a911ed7` was removed while I worked (the Bash tool then refused every command: "working directory no longer exists"), so I could not read the 36-file tracer output, re-time the small-commit hooks at lower load, or commit this document. The text above is final apart from the dynamic sleep/subprocess counts. The document lives at `scratchpad/test-time.md`; copy it to `docs/test-time.md` and commit it with a `chore:` prefix.
