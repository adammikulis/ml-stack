# Parked branches (unmerged on purpose)

| branch | what it is | why it is not merged |
|---|---|---|
| `feat/decide-finetune` | fine-tuning path for the pointer-head decision model | the owner: do not commit fine-tuned deciders; Strands 2B stays the default |
| `feat/decider-gemma4` | FunctionGemma 270M base pin and test | paused by the owner; progress in `docs/notes/gemma-decider.md` |
| `agent/redteam-integ` | older red-team integration branch | superseded by the coverage map; check for anything not already on `0.2dev` before deleting |
| `feat/test-speed` | one commit (`-rfE` so FAILED/ERROR lines always print) | in flight: lands after Codex's scheduler fix |

Codex's worktrees (`/private/tmp/poolhouse-*`, branches `feat/gym-*`, `feat/ui-*`, `feat/world-*`,
`feat/board-person-participation`, `feat/chat-workspace-v3`, `feat/workspace-*`,
`feat/vision-discovery`, `feat/native-world-construction`, `feat/studio-gym-docs`) belong to Codex:
it lists and removes its own. `feat/scheduler` is merged: that worktree can go.
