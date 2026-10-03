"""Run commands under an OS-level sandbox with a deny-by-default policy.

    from ml_stack import sandbox

    policy = sandbox.Policy("scratch", read=[project], write=[scratch], exec=["/bin/echo"],
                            env={"PATH": "/usr/bin:/bin"}, net=sandbox.Net.deny())
    result = sandbox.run(["/bin/echo", "hi"], policy)

With no sandbox available `run` raises `SandboxUnavailable`; running without one takes an
explicit `AllowUnsandboxed(reason)`.
"""

from __future__ import annotations

from ml_stack.sandbox.backend import Availability, Backend, SandboxUnavailable
from ml_stack.sandbox.policy import (
                                     AllowUnsandboxed,
                                     Limits,
                                     Net,
                                     NetMode,
                                     Policy,
                                     PolicyError,
                                     checked_path,
)
from ml_stack.sandbox.run import Result, SandboxViolation, backend, run, wrapped

__all__ = ["AllowUnsandboxed", "Availability", "Backend", "Limits", "Net", "NetMode", "Policy",
           "PolicyError", "Result", "SandboxUnavailable", "SandboxViolation", "backend",
           "checked_path", "run", "wrapped"]
