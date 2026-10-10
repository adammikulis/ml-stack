"""Start, adopt and tear down local model servers."""

from __future__ import annotations

from poolhouse.serve.backend import (
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
    parse_context,
    tail,
    trained_context,
)
from poolhouse.serve.binary import (
    BinaryNotFound,
    child_env,
    find_binary,
    require_binary,
)
from poolhouse.serve.escalation import EscalationRefused
from poolhouse.serve.leases import merge_state, recorded_servers
from poolhouse.serve.manager import (
    Measuring,
    ServerManager,
    serve,
    stop_all_servers,
)
from poolhouse.serve.matching import model_matches, serving_mismatch
from poolhouse.serve.ports import (
    DEFAULT_HOST,
    free_port,
    port_is_free,
    reclaim_port,
    server_pids_on_port,
    wait_until_free,
)
from poolhouse.serve.process import kill_pid, kill_process_tree, pid_exists
from poolhouse.serve.profile import Profile, profile_for, profiles
from poolhouse.serve.public import Served, down, status, up
from poolhouse.serve.serving import (
    Config,
    Serving,
    Talking,
    draft_for,
    projector_for,
    slot,
)
from poolhouse.serve.slotdump import (
    SlotDump,
    SlotGuardRefused,
    restore_all,
    restore_slot,
    save_all,
    save_slot,
)

__all__ = [
    "DEFAULT_HOST",
    "BinaryNotFound",
    "Config",
    "EscalationRefused",
    "LlamaServerBackend",
    "Measuring",
    "Profile",
    "Served",
    "ServerBackend",
    "ServerFailed",
    "ServerInfo",
    "ServerManager",
    "ServerSpec",
    "Serving",
    "SlotDump",
    "SlotGuardRefused",
    "Talking",
    "child_env",
    "down",
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
    "restore_all",
    "restore_slot",
    "save_all",
    "save_slot",
    "serve",
    "server_pids_on_port",
    "serving_mismatch",
    "slot",
    "status",
    "stop_all_servers",
    "tail",
    "trained_context",
    "up",
    "wait_until_free",
]
