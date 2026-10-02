"""The switches that let a test reach a real remote service, and the checks behind them.

A test marked ``live_api`` calls a paid or quota-limited API and runs only when
``ML_STACK_LIVE_API=1``; one marked ``live_net`` reaches a public endpoint and runs only when
``ML_STACK_LIVE_NET=1``. A credential in the environment switches nothing on.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable, Mapping

__all__ = ["CREDENTIALS", "LIVE_API", "LIVE_NET", "MARKERS", "PAID_SDKS", "outside", "skip_reason"]

LIVE_API = "ML_STACK_LIVE_API"
LIVE_NET = "ML_STACK_LIVE_NET"
MARKERS = {"live_api": LIVE_API, "live_net": LIVE_NET}

CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
               "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY",
               "MISTRAL_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "TOGETHER_API_KEY",
               "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
PAID_SDKS = ("anthropic", "openai", "google.generativeai", "google.genai", "cohere", "mistralai",
             "groq", "together", "replicate", "litellm", "boto3")
LOCAL_SUFFIXES = (".local", ".localdomain", ".lan", ".home.arpa", ".test", ".invalid", ".example",
                  ".localhost")


def skip_reason(marks: Iterable[str], environ: Mapping[str, str]) -> str:
    """Why a test carrying these marker names is skipped, empty when it may run."""
    for mark in marks:
        switch = MARKERS.get(mark)
        if switch is not None and environ.get(switch) != "1":
            return f"reaches a real remote service; set {switch}=1 to run it on purpose"
    return ""


def outside(host: object) -> bool:
    """Whether a host, as a name or an address, is somewhere beyond this machine and its LAN."""
    if not isinstance(host, str) or host in ("", "<broadcast>", "localhost"):
        return False
    text = host.strip("[]").split("%")[0]
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        name = text.lower().rstrip(".")
        return not (name.endswith(LOCAL_SUFFIXES) or name == socket.gethostname().lower()
                    or "." not in name)
    return not (address.is_private or address.is_loopback or address.is_link_local
                or address.is_multicast or address.is_unspecified)
