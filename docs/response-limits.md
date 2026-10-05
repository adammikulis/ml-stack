# Response limits and streaming

Reasoning effort and answer length are separate settings. Turning thinking off does not impose a 2,048-token answer limit. The local agent wrapper preserves explicit `n_predict` values and otherwise inherits its maintained client's request settings. Its legacy client no longer installs a separate 4,096-token ceiling.

In Chat, Advanced options exposes **Maximum output tokens**. Blank uses the selected model server's output default; a positive integer passes through the SDK as the output ceiling. The server's available context still limits generation. This setting does not change context length, task turns, or tool permissions.

Chat uses real model deltas through the local/peer transport, SDK event stream, Fleet SSE route and browser reader. Text renders before generation finishes; saved conversations are finalized when the response completes. Disconnect and Stop close the upstream generation. CLI chat and the general SDK agent also forward live model text/thinking callbacks. An absent Agents SDK is an explicit unavailable error in Fleet Chat, not a buffered alternate generation path. Typed decision and extraction APIs retain their own bounded structured-output contracts.

| Limit | Current default | Configuration / purpose |
| --- | --- | --- |
| Direct maintained client output | 16,384 tokens | `Request(n_predict=…)`, or explicit per-call `n_predict`; generation ceiling independent of thinking |
| Fleet Chat output | Selected server default | Maximum output tokens in Advanced; forwarded unchanged |
| General SDK agent turns / tool calls | 8 / 64 | `Budget(max_steps=…, max_tool_calls=…)`; execution bounds, not response-token caps |
| General SDK agent total output tokens | No aggregate cap | Optional `Budget.max_tokens`; assessed between model calls |
| Repair turns / parallel tools | 2 / 4 | `Budget.max_repairs` / `parallel`; prevents endless rejected-call loops and controls concurrency |
| General agent tool result shown to model | 4,000 characters | `Budget.max_result_chars`; bounds tool context, not generated answer length |
| CLI chat / task model turns | 12 / 40 | `--rounds`; shows exhausted-turn status rather than claiming completion |
| CLI chat tool result shown to model | 6,000 characters | Existing `chat.CUT`; still fixed and should be assessed separately for large artifacts |
| Context compaction | At 80%, target 50%, keep 6 recent units | `Compaction`; uses selected server context or explicit context-size override |
| SDK peer stream timeout | 600 seconds | Current fixed upstream wait; transport reads available bytes with `read1(4096)`, which does not buffer a whole answer |

Native coding currently has separate turn and wall-time budgets; these are execution controls and are being exposed independently of thinking and output settings. Network parser, upload, authentication and security size limits are not output-token budgets and remain enforced. These defaults are an audit of the current paths, not a claim that every fixed limit is ideal for every workload.
