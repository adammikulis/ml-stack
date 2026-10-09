# Where the full tier's time goes

Measured on one 16-core machine shared with other agents' test runs (load average 7 to 30 while these ran), with
`scripts/test full --durations=0`. The owner's demand: "10-20 minutes of testing is too much".

| run | code | wall | workers | sum of test times | note |
|---|---|---|---|---|---|
| 1 | before | 9 min 31 s | 16 | 3026 s | one test took 419 s and held the tail |
| 2 | after the deadlock fix, the in-process scoreboard, reputation, PDF, decoy corpus | **3 min 03 s** | 16 | 2440 s | load average 9 |
| 3 | plus the 20 ms server poll and the flake fixes | 5 min 49 s | 6 (the machine was at load 29, the broker granted 6) | **1921 s** | wall is the contention, not the tests |

The sum of test times fell by 36 % (3026 s to 1921 s); with the machine to itself, 1921 s over 16 workers is about
2 minutes of work plus 15 s of start-up and collection.

## What the time was

Run 1, by cause (setup + call + teardown, seconds summed over 10,169 timed phases):

* **A deadlock, 419 s on one worker.** `test_running_on_linux.py::test_the_runner_says_which_command_told_it_docker_is_missing`
  runs `scripts/test-on-linux`, which first queues for slots in the machine-wide budget. Inside a full run the run
  holds every slot, so the child waited for slots its parent held until the test was killed (it then failed with
  -15). Every other worker was idle behind it: wall 570 s against 3026 / 16 = 190 s of work. Fixed by giving the
  child `DEV_TEST_WORKERS` (the runner's own "a lease is already held" signal) and a 60 s timeout.
* **Server shutdown, about 417 s of worker time in teardown.** `socketserver.BaseServer.shutdown()` waits for
  `serve_forever` to notice, and it looks once per `poll_interval`, 0.5 s by default. About a thousand tests start a
  real in-process server; each stop cost a quarter of a second on average. Teardown time fell from 496 s to 79 s with
  one change in `tests/conftest.py` (a 20 ms poll in the test process; child processes are not touched).
* **The structure scoreboard run as a subprocess, 65 s in two tests.** `scripts/budgets` measures every metric
  (pyright 17 s) and collects the whole suite for the floor; two tests ran it only to read one caveat line and one
  empty-ledger message. They call the real table printer and the real `verify` in process now (2 s).
* **Timeouts waited out.** The 100,000-deep PDF object was refused after a 30 s kill; 4 s proves the same refusal.
* **Write amplification in a test.** `test_the_store_is_bounded` encrypted and wrote the whole reputation store
  once per source (1003 times); clean runs are batched by `flush_s`, so it batches (33 s to 1 s).
* **One test with a 24 s body** (48 agent runs against decoys) is now one case per shape (7 cases of about 3 s).
* **Collection and import**: `pytest --collect-only` takes 8.7 s of CPU for 9,573 tests, paid once per worker (16
  workers: about 140 CPU-s, 9 s of wall). Importing the heavy packages is inside that. No change made.
* **Fixture setup** is 251 s (13 %): five module-scoped graph fixtures take 8 to 16 s each (a 200-turn thread, a
  detach-delete store, bench grids, a red-team lab); they are built once per module and per worker.
* **Worker imbalance**: with the deadlock gone the tail is short: run 2 had 183 s wall against 152 s of ideal
  (2440 / 16), the rest being start-up and the last long tests (`--dist load`, xdist's default, kept; a module-level
  split by `loadfile` would be worse because `test_graph_bench.py` alone is 74 s).

## The 60 slowest test phases, run 3 (final code)

```
17.6 s call tests/test_no_data_files.py::test_the_tracked_tree_passes
15.7 s setup tests/test_graph_thread.py::test_recall_finds_the_fact_from_turn_one_two_hundred_turns_later
14.3 s call tests/test_keystore.py::test_one_instance_locking_and_reading_in_a_loop_is_capped_too
14.2 s setup tests/test_graph_engine_contract.py::test_every_relation_reads_the_same_from_both_ends_after_a_write[detach-delete]
14.1 s call tests/test_workspace_load.py::test_the_shipped_rate_limit_shows_up_as_counted_failures_not_a_failed_run
13.6 s call tests/test_net_scan.py::test_real_clamav_finds_the_eicar_test_string
13.0 s call tests/test_serve_llamacpp.py::test_prune_keeps_three_good_builds_and_asks_before_deleting
12.1 s call tests/test_sentinel_score_wiring.py::test_sandbox_runs_are_reported_and_counted_for_the_session_running_the_tool
11.8 s call tests/test_serve_admission.py::test_two_processes_that_do_not_fit_together_do_not_both_start
10.9 s call tests/test_keystore.py::test_forty_processes_starting_together_never_see_busy_and_stay_under_the_read_ceiling
10.7 s call tests/test_keystore.py::test_readers_slow_down_with_jitter_as_they_near_the_read_ceiling
8.8 s call tests/test_onboard_cli.py::test_nearby_hears_a_listener_that_announces_to_loopback
8.5 s call tests/test_keystore.py::test_a_burst_of_700_fresh_readers_reaches_the_backend_at_most_the_read_ceiling
7.9 s call tests/test_graph_store_scale.py::test_a_long_write_does_not_keep_a_plan_per_statement
7.8 s call tests/test_workspace_surface.py::test_every_command_prints_json_and_reports_errors_as_json
7.3 s call tests/test_graph_engine_contract.py::test_a_read_only_handle_used_after_another_rewrote_the_store_crashes
7.2 s call tests/test_fleet_salt.py::test_the_right_cluster_is_picked_among_two_on_one_network
6.9 s call tests/test_ingest.py::test_folding_a_few_hundred_units_of_one_source_costs_a_second_or_two
6.7 s call tests/test_sandbox_seatbelt.py::test_a_wall_clock_limit_kills_the_group_and_only_that_group
6.2 s call tests/test_serve_admission.py::test_servers_that_fit_run_together_and_a_third_that_does_not_is_refused
5.8 s call tests/test_workspace_quickstart.py::test_connect_says_what_to_check_when_nothing_answers_and_when_nobody_joins
5.5 s call tests/test_sandbox_seatbelt.py::test_output_over_the_cap_is_cut_and_the_command_stopped
5.4 s call tests/test_no_live_calls.py::test_a_credential_alone_does_not_run_a_live_test[live_api]
5.4 s call tests/test_workspace_board.py::test_the_commands_drive_boards_dms_subscriptions_and_status
5.3 s call tests/test_keystore.py::test_200_fresh_readers_never_see_busy_and_make_at_most_one_read_each
5.3 s call tests/test_world_simulate.py::test_run_absorbs_a_second_worlds_graph_into_the_store_the_first_left
5.3 s call tests/test_sentinel_heal.py::test_a_decider_run_killed_while_fitting_leaves_no_pins
5.1 s call tests/test_serve_admission.py::test_a_start_that_would_be_red_goes_ahead_when_memory_comes_free
5.1 s call tests/test_sandbox_redteam.py::test_injected_commands_are_stopped_by_the_sandbox_when_the_guard_allowed_them
5.1 s call tests/test_serve_no_bypass.py::test_a_connection_to_a_server_goes_through_the_request_queue
5.0 s call tests/test_jobs_cli.py::test_wait_blocks_on_the_named_kind_and_says_when_it_has_ended
5.0 s call tests/test_redteam_coverage.py::test_the_tree_passes_the_gate
5.0 s call tests/test_memory_chat.py::test_fifty_sessions_and_an_empty_recall_touch_the_keystore_zero_times
4.8 s call tests/test_keystore.py::test_500_create_and_delete_attempts_stop_at_the_write_ceiling
4.8 s call tests/test_setup.py::test_the_printed_report_names_the_missing_command_and_the_line
4.7 s call tests/test_serve_broker.py::test_a_conflicting_model_waits_for_the_holder_and_kills_nothing_it_holds
4.7 s call tests/test_serve.py::TestStartGuards::test_a_foreign_process_on_the_port_is_refused_not_killed
4.6 s call tests/test_gate.py::test_requests_are_served_in_the_order_they_arrived
4.6 s call tests/test_no_live_calls.py::test_the_switch_runs_a_live_test_and_the_other_switch_does_not[live_api-POOLHOUSE_LIVE_API]
4.5 s call tests/test_fleet_salt.py::test_words_no_cluster_here_accepts_are_an_error_not_a_new_cluster
4.5 s call tests/test_gate.py::test_requests_sent_at_the_same_instant_never_overlap
4.4 s call tests/test_no_live_calls.py::test_the_switch_runs_a_live_test_and_the_other_switch_does_not[live_net-POOLHOUSE_LIVE_NET]
4.3 s call tests/test_ingest.py::test_resume_skips_what_is_already_done_and_asks_the_model_nothing_more
4.1 s call tests/test_serve_llamacpp.py::test_a_build_that_fails_its_smoke_test_leaves_the_old_build_active_and_is_kept
4.1 s call tests/test_pdf_engine.py::test_work_that_takes_too_long_is_killed
4.0 s call tests/test_pdf_engine.py::test_an_object_nested_a_hundred_thousand_deep_is_refused_not_crashed
4.0 s call tests/test_sandbox_integration.py::test_the_shell_tool_reads_the_project_and_writes_only_its_scratch
3.9 s call tests/test_workspace_quickstart.py::test_two_children_in_real_processes_labels_and_the_children_filter
3.9 s call tests/test_serve_broker.py::test_a_held_server_is_not_stopped_on_request_and_a_foreign_one_never
3.8 s call tests/test_serve_broker.py::test_a_dead_holders_lease_is_reaped_and_the_queue_advances
3.8 s call tests/test_fleet_salt.py::test_a_machine_that_joins_learns_the_salt_and_ends_with_the_same_key
3.8 s call tests/test_serve_broker.py::test_a_timed_out_ask_says_who_holds_what_it_wanted
3.7 s call tests/test_serve_llamacpp.py::test_pin_stops_tracking_and_update_refuses_until_unpinned
3.7 s call tests/test_serve_llamacpp.py::test_a_second_update_keeps_the_first_for_rollback
3.6 s call tests/test_sandbox_integration.py::test_a_download_and_run_command_is_stopped_even_though_no_guard_saw_it
3.6 s call tests/test_serve_llamacpp.py::test_the_compile_cannot_write_outside_its_directory
3.4 s call tests/test_keystore.py::test_six_processes_starting_together_create_one_master
3.3 s call tests/test_ingest.py::test_core_only_reads_a_section_under_the_core_lists_alone
3.3 s call tests/test_testslots.py::test_status_lists_running_and_waiting
3.3 s call tests/test_graph_tidy_judge.py::test_the_run_tidies_each_source_on_the_way_out_with_its_own_model
```

## The 30 slowest modules, run 3 (sum of setup, call and teardown)

```
66.0 s tests/test_keystore.py
61.4 s tests/test_graph_bench.py
50.8 s tests/test_ingest.py
50.5 s tests/test_serve_llamacpp.py
44.9 s tests/test_graph_thread.py
42.9 s tests/test_graph_bench_report.py
42.0 s tests/test_peer_first.py
38.9 s tests/test_serve_admission.py
37.9 s tests/test_sandbox_seatbelt.py
34.8 s tests/test_graph_store.py
34.1 s tests/test_serve_broker.py
31.6 s tests/test_graph_engine_contract.py
30.4 s tests/test_graph_tidy_judge.py
29.3 s tests/test_peer_pairing.py
29.1 s tests/test_onboard_cli.py
28.1 s tests/test_ingest_sources.py
27.6 s tests/test_graph_tidy.py
26.5 s tests/test_ingest_migrate.py
26.1 s tests/test_fleet_bench.py
25.6 s tests/test_ingest_ask.py
24.9 s tests/test_graph_serve.py
23.7 s tests/test_workspace_quickstart.py
23.0 s tests/test_fleet_ui.py
22.4 s tests/test_fleet_salt.py
22.0 s tests/test_fleet_join.py
21.7 s tests/test_no_data_files.py
21.2 s tests/test_world_check.py
21.0 s tests/test_testslots.py
20.8 s tests/test_no_live_calls.py
19.6 s tests/test_graph_bench_report_ingest.py
```

## What stayed slow, and why

Tests that start real subprocesses, real servers or a real scanner and wait for them are the remaining floor:
`test_graph_*` (embedded database builds), `test_serve_admission.py` and `test_serve_llamacpp.py` (fake llama
servers in child processes, kill and reap), `test_keystore.py` (forty processes against one file, by design),
`test_sandbox_seatbelt.py` (real `sandbox-exec`), `test_net_scan.py` (a real ClamAV), `test_fleet_*` (real daemons).
They are what the tests are for; none was weakened or marked `slow`. The tests that wait out a time bound on purpose
(`test_work_that_takes_too_long_is_killed`, the PDF depth test) take the bound they assert, 4 s each.

## Load flakes: what broke under load, and the fix

| symptom | cause | fix |
|---|---|---|
| the run hangs 7 min, then `-15 == 2` | nested slot lease (above) | child gets `DEV_TEST_WORKERS` |
| `the real state root ... was written during the run` on 5 to 16 tests | the guard compares mtimes of the whole state root, and every other agent on the machine writes `workspace/`, `activity/` (every `scripts/test` run ends by logging there), `sentinel/` and `harness/` | those four are live writers like the broker files already were |
| `host 127.0.0.1 is watch` in 9 `test_serve_llamacpp` tests | a test left the process-wide reputation observer installed; whichever test ran next on that xdist worker inherited it | autouse fixture restores the observer after each test |
| `cannot resolve 'docs.example'` | `test_the_reader_the_agent_really_uses...` relied on a neighbour having started the lab that resolves that name | the test asks for the `stood_up` fixture (failed alone before, passes now) |
| `ModuleNotFoundError: tests` in 3 `test_no_live_calls` tests | the child's `PYTHONPATH` lacked the repo root (deterministic, not load) | added |
| `test_a_server_stops_when_the_host_is_terminated` timed out at 20 s | process-tree kill under load 20+ | the wait is 60 s (it returns at once when it works) |

Not fixed: `test_embedding.py::test_importing_them_loads_only_the_standard_library_and_the_core_dependencies` fails
every time (importing `poolhouse.serve` and `poolhouse.client` now imports `cryptography`, through the requests
model), a regression on the integration branch and not a flake; and one failure of
`test_sentinel_notice_storm.py::test_four_changed_files_scanned_by_many_processes_are_one_dialog` at load 29 (no
dialog started; six repeats at load 18 passed), which may be a lost heads-up under load in the product code.
