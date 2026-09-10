"""Start, adopt and tear down local model servers."""

from __future__ import annotations

from ml_stack.serve.backend import (
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
    parse_context,
    tail,
    trained_context,
)
from ml_stack.serve.binary import (
    BinaryNotFound,
    child_env,
    find_binary,
    require_binary,
)
from ml_stack.serve.manager import (
    EscalationRefused,
    Measuring,
    ServerManager,
    merge_state,
    model_matches,
    recorded_servers,
    serve,
    serving_mismatch,
    stop_all_servers,
)
from ml_stack.serve.ports import (
    DEFAULT_HOST,
    free_port,
    port_is_free,
    reclaim_port,
    server_pids_on_port,
)
from ml_stack.serve.process import kill_pid, kill_process_tree, pid_exists
from ml_stack.serve.profile import Profile, profile_for, profiles
from ml_stack.serve.serving import (
    Run,
    Serving,
    Talking,
    draft_for,
    projector_for,
    slot,
)

__all__ = [
    "DEFAULT_HOST",
    "BinaryNotFound",
    "EscalationRefused",
    "LlamaServerBackend",
    "Measuring",
    "ServerBackend",
    "ServerFailed",
    "ServerInfo",
    "ServerManager",
    "Profile",
    "Run",
    "ServerSpec",
    "Serving",
    "Talking",
    "child_env",
    "draft_for",
    "find_binary",
    "free_port",
    "kill_pid",
    "kill_process_tree",
    "merge_state",
    "model_matches",
    "parse_context",
    "pid_exists",
    "port_is_free",
    "profile_for",
    "profiles",
    "projector_for",
    "reclaim_port",
    "recorded_servers",
    "require_binary",
    "slot",
    "serve",
    "server_pids_on_port",
    "serving_mismatch",
    "stop_all_servers",
    "tail",
    "trained_context",
]
