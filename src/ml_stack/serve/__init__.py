"""Start, adopt and tear down local model servers."""

from __future__ import annotations

from ml_stack.serve.backend import (
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
    tail,
)
from ml_stack.serve.binary import (
    BinaryNotFound,
    CACHE_ROOT,
    child_env,
    find_binary,
    require_binary,
)
from ml_stack.serve.manager import (
    STATE_FILE,
    ServerManager,
    merge_state,
    model_matches,
    recorded_servers,
    serve,
    shape_mismatch,
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
from ml_stack.serve.shape import Shape, draft_for, projector_for, seat

__all__ = [
    "CACHE_ROOT",
    "DEFAULT_HOST",
    "STATE_FILE",
    "BinaryNotFound",
    "LlamaServerBackend",
    "ServerBackend",
    "ServerFailed",
    "ServerInfo",
    "ServerManager",
    "ServerSpec",
    "Shape",
    "child_env",
    "draft_for",
    "find_binary",
    "free_port",
    "kill_pid",
    "kill_process_tree",
    "merge_state",
    "model_matches",
    "pid_exists",
    "port_is_free",
    "projector_for",
    "reclaim_port",
    "recorded_servers",
    "require_binary",
    "seat",
    "serve",
    "server_pids_on_port",
    "shape_mismatch",
    "stop_all_servers",
    "tail",
]
