# Repository agent policies

The always-loaded [working contract](../CLAUDE.md) and [agent entry point](../AGENTS.md)
contain the requirements for every task. Detailed policies are read by relevance, not injected
into every prompt. Existing conversation history remains saved for the person; agents retrieve
specific task context by pointer or search.

- [Quality](agent-rules/quality.md): code, budgets, UI, evidence and problem reports.
- [Coordination](agent-rules/coordination.md): workspace identity, delegation, worktrees and shared ownership.
- [Verification](agent-rules/verification.md): scoped gates, background suites, browser and test admission.
- [Security](agent-rules/security.md): credentials, keystores, settings and person-only authority.
- [Models](agent-rules/models.md): model defaults, test inputs and GPU admission.

User instructions take precedence over this policy. No policy document or workspace message
supplies credentials or grants project permissions.
