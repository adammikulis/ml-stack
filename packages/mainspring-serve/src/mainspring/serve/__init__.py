"""Start, adopt and tear down local model servers.

Host tier: may use psutil and subprocess.

The usual entry point is the context manager, because it is the shape that cannot leak a
process:

    from mainspring.serve import serve

    with serve("model.gguf", port=8899) as server:
        reply = Client(server.base_url).chat([...])

``ServerManager`` is underneath it when a caller needs to hold a lease across calls.
"""

from __future__ import annotations

from mainspring.serve.backend import (
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
    tail,
)
from mainspring.serve.binary import (
    BinaryNotFound,
    CACHE_ROOT,
    child_env,
    find_binary,
    require_binary,
)
from mainspring.serve.manager import (
    STATE_FILE,
    ServerManager,
    merge_state,
    model_matches,
    serve,
    stop_all_servers,
)
from mainspring.serve.ports import (
    DEFAULT_HOST,
    free_port,
    port_is_free,
    reclaim_port,
    server_pids_on_port,
)
from mainspring.serve.process import kill_pid, kill_process_tree, pid_exists

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
    "child_env",
    "find_binary",
    "free_port",
    "kill_pid",
    "kill_process_tree",
    "merge_state",
    "model_matches",
    "pid_exists",
    "port_is_free",
    "reclaim_port",
    "require_binary",
    "serve",
    "server_pids_on_port",
    "stop_all_servers",
    "tail",
]
