"""Allowed capability-scoped project workspace RPC operations."""

METHODS = frozenset({"send", "inbox", "outbox", "ack", "thread", "announce", "claim_model", "register_session", "record_profile", "nudge", "waiting_summary", "wait",
                     "heartbeat", "record_execution_profile", "execution_profiles", "board.list", "board.read",
                     "board.threads", "board.join", "board.leave", "board.dm",
                     "board.subscribe", "board.unsubscribe", "board.subs", "board.digest",
                     "board.rollup", "board.summary", "board.mentions"})

