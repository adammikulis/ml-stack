"""What upstream llama.cpp is now: its latest release tag and its master commit, through the net pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ml_stack import net
from ml_stack.http import ServerError
from ml_stack.httpguard import Refused

__all__ = ["Newest", "Unresolved", "Upstream", "latest", "resolve", "tag_number"]

SHA = re.compile(r"[0-9a-f]{40}")
_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,100}")
_TAG = re.compile(r"b(\d+)")
_RELEASE = re.compile(r"b\d+|v\d+(?:\.\d+)*")


class Unresolved(RuntimeError):
    """A ref could not be turned into a commit."""


@dataclass(frozen=True, slots=True)
class Upstream:
    """Where llama.cpp comes from: the repository, GitHub's git host and its API host."""

    repo: str = "ggml-org/llama.cpp"
    git_base: str = "https://github.com"
    api_base: str = "https://api.github.com"

    @property
    def git_url(self) -> str:
        return f"{self.git_base}/{self.repo}"

    @property
    def mainline(self) -> bool:
        return self.repo.lower() == "ggml-org/llama.cpp"


@dataclass(frozen=True, slots=True)
class Newest:
    """What upstream is now: ``tag`` the newest build tag (``b11379``, which follows master),
    ``tagged`` the commit it is on, ``stable`` the newest stable release (``v0.5.0``) and
    ``master`` its tip; empty fields when unknown."""

    tag: str = ""
    tagged: str = ""
    stable: str = ""
    master: str = ""
    reason: str = ""

    @property
    def known(self) -> bool:
        return bool(self.tag or self.master)


def tag_number(tag: str) -> int | None:
    """The build number in a release tag like ``b11349``, or None."""
    found = _TAG.fullmatch(tag.strip())
    return int(found.group(1)) if found else None


def _json(upstream: Upstream, path: str, pipeline: net.Pipeline) -> object:
    return pipeline.json(f"{upstream.api_base}/repos/{upstream.repo}/{path}",
                         net.Ask(purpose="llama.cpp upstream", max_bytes=1 << 20))


def latest(upstream: Upstream | None = None, pipeline: net.Pipeline | None = None) -> Newest:
    """The newest build tag, stable release and master commit. A host that needs approval is
    raised (`net.NeedsApproval`); any other failure is an unknown answer saying why."""
    upstream, pipeline = upstream or Upstream(), pipeline or net.default()
    tag = tagged = stable = master = ""
    reasons: list[str] = []
    try:
        newest = _json(upstream, "releases?per_page=1", pipeline)
        if isinstance(newest, list) and newest and isinstance(newest[0], dict):
            tag, tagged = str(newest[0].get("tag_name") or ""), str(newest[0].get("target_commitish") or "")
        commit = _json(upstream, "commits/master", pipeline)
        master = str(commit.get("sha") or "") if isinstance(commit, dict) else ""
        release = _json(upstream, "releases/latest", pipeline)
        stable = str(release.get("tag_name") or "") if isinstance(release, dict) else ""
    except net.NeedsApproval:
        raise
    except (ServerError, Refused, OSError, ValueError) as exc:
        reasons.append(str(exc))
    return Newest(tag if _TAG.fullmatch(tag) else "", tagged if SHA.fullmatch(tagged) else "",
                  stable if _RELEASE.fullmatch(stable) else "",
                  master if SHA.fullmatch(master) else "", "; ".join(reasons))


def resolve(ref: str, upstream: Upstream | None = None,
            pipeline: net.Pipeline | None = None) -> tuple[str, str]:
    """``(commit, tag)`` for a branch, tag or commit: the full sha upstream reports for it."""
    upstream, pipeline = upstream or Upstream(), pipeline or net.default()
    ref = ref.strip()
    if not _REF.fullmatch(ref) or ".." in ref or ref.endswith(("/", ".lock")):
        raise Unresolved(f"{ref!r} is not a branch, tag or commit")
    try:
        found = _json(upstream, f"commits/{ref}", pipeline)
    except net.NeedsApproval:
        raise
    except (ServerError, Refused, OSError, ValueError) as exc:
        raise Unresolved(f"could not resolve {ref!r} upstream: {exc}") from exc
    sha = str(found.get("sha") or "") if isinstance(found, dict) else ""
    if not SHA.fullmatch(sha):
        raise Unresolved(f"upstream answered {ref!r} with no commit")
    return sha, ref if _RELEASE.fullmatch(ref) else ""
