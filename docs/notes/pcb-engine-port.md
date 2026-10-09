# What pcb-engine's local-model code had, and where it is here

| Capability | State | Where |
|---|---|---|
| Process lifecycle, binary discovery, Windows DLL environment | present | `poolhouse.serve` (`ServerManager`, `find_binary`, `child_env`) |
| `hf:<owner>/<repo>/<file>.gguf` references | present | `poolhouse.hub`, `poolhouse.serve.serving` |
| Chat client: `build_body` apart from `chat`, `id_slot`, `cache_prompt`, `<think>` handling, grammar retry | present | `poolhouse.client.Client` |
| Slot save/restore around a relaunch | present | `poolhouse.serve.escalation` |
| Slot dumps with a guard file refusing a different model, context, slot count or build | added | `poolhouse.serve.slotdump`, `poolhouse-serve slots` |
| Config discovery | not ported | project-specific |
| Tool-calling loop over MCP tools (`agent_loop.py`, `tool_bridge.py`, `mcp_client.py`) | partial: `poolhouse-chat` runs a synchronous loop | `poolhouse.agent` |
| Schema validation before dispatch, repair of malformed arguments | added | `poolhouse.agent.schema` |
| Result summarisation (`feedback.py`) | added as a hook | `Agent(summarise=...)`, `Budget.max_result_chars` |
| Context counting and compaction | added | `poolhouse.agent.compact`, `docs/compaction.md` |
| Synthetic tool-call conversations, held-out split, LoRA, GGUF export | present | `poolhouse-train-tools` |
| Schema snapshot and drift refusal | added | `poolhouse.train.tools.drift`, manifest `schema_hash` |
| Held-out tool-call accuracy | added | `poolhouse.train.tools.evaluate`, `poolhouse-train-tools eval` |
| Base-versus-tuned comparison in one command | not ported | run `eval` against each server |
