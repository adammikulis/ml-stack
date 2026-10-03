"""What a fetch or a download tells the reputation ledger about the source it came from. Every
value is read from the request and the answer, never from the text of a page."""

from __future__ import annotations

from ml_stack.httpguard import Refused
from ml_stack.net.policy import NeedsApproval, host_of
from ml_stack.sentinel import observers

__all__ = ["downloaded", "fetched", "refused", "rejected"]


def fetched(url: str, final_url: str, redirects: tuple[str, ...], size: int, content_type: str) -> None:
    """A fetch of ``url`` that came back: a clean run, its redirect targets and its shape."""
    host = host_of(url)
    if not host:
        return
    observers.clean("host", host)
    for hop in (*redirects, final_url):
        target = host_of(hop)
        if target and target != host:
            observers.trait("host", host, "redirect", target)
    observers.trait("host", host, "shape", f"{size.bit_length()}:{content_type.split(';')[0].strip()}")


def refused(url: str, error: Refused) -> None:
    """A fetch the network layer refused (not one that only needs a person's approval)."""
    host = host_of(url)
    if host and not isinstance(error, NeedsApproval):
        observers.observe("host", host, "denial")


def downloaded(url: str) -> None:
    """A file that passed every check."""
    host = host_of(url)
    if host:
        observers.clean("host", host)


def rejected(url: str, digest: str, event: str) -> None:
    """A file that failed a check: ``event`` is ``hash_change``, ``scan_hit`` or ``denial``."""
    host = host_of(url)
    if host:
        observers.observe("host", host, event)
    if digest and event == "scan_hit":
        observers.observe("hash", digest, event)
