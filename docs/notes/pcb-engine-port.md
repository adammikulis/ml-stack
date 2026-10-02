# What pcb-engine's local-model code had, and where it is here

| Capability | State | Where |
|---|---|---|
| Process lifecycle, binary discovery, Windows DLL environment | present | `ml_stack.serve` (`ServerManager`, `find_binary`, `child_env`) |
| `hf:<owner>/<repo>/<file>.gguf` references | present | `ml_stack.hub`, `ml_stack.serve.serving` |
| Chat client: `build_body` apart from `chat`, `id_slot`, `cache_prompt`, `<think>` handling, grammar retry | present | `ml_stack.client.Client` |
| Slot save/restore around a relaunch | present | `ml_stack.serve.escalation` |
| Slot dumps with a guard file refusing a different model, context, slot count or build | added | `ml_stack.serve.slotdump`, `ml-stack-serve slots` |
| Config discovery | not ported | project-specific |
| Tool-calling loop over MCP tools (`agent_loop.py`, `tool_bridge.py`, `mcp_client.py`) | partial: `ml-stack-do` had a synchronous loop | `ml_stack.agent` |
| Schema validation before dispatch, repair of malformed arguments | added | `ml_stack.agent.schema` |
| Result summarisation (`feedback.py`) | added as a hook | `Agent(summarise=...)`, `Budget.max_result_chars` |
| Context counting and compaction | added | `ml_stack.agent.compact`, `docs/compaction.md` |
| Synthetic tool-call conversations, held-out split, LoRA, GGUF export | present | `ml-stack-train-tools` |
| Schema snapshot and drift refusal | added | `ml_stack.train.tools.drift`, manifest `schema_hash` |
| Held-out tool-call accuracy | added | `ml_stack.train.tools.evaluate`, `ml-stack-train-tools eval` |
| Base-versus-tuned comparison in one command | not ported | run `eval` against each server |
