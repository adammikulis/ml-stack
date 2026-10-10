"""Asking a Hugging Face endpoint what it holds: references, listings and searches.

Everything here is plain HTTP against ``$HF_ENDPOINT`` (default https://huggingface.co), so
it needs no extra package and a test can point it at a local server.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from dataclasses import dataclass, replace

from poolhouse import credentials, http, net
from poolhouse.httpguard import Refused
from poolhouse.hub.naming import _SHARD, QUANT, aside
from poolhouse.net.untrusted import clean_text

DEFAULT_ENDPOINT = "https://huggingface.co"

TOKEN_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

_QUANT_TAG = re.compile(r"^[A-Za-z0-9_]+$")


class RemoteError(RuntimeError):
    """The endpoint refused or could not answer."""


class NotFound(RemoteError):
    """The repository or file is not there."""


class GatedRepo(RemoteError):
    """The repository needs a licence accepted or a token."""


def endpoint() -> str:
    """The Hub address: ``$HF_ENDPOINT`` or https://huggingface.co."""
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def token() -> str:
    """The Hugging Face access token from poolhouse's credential sources."""
    for name in TOKEN_VARIABLES:
        if os.environ.get(name):
            return os.environ[name].strip()
    try:
        found = credentials.get("HF_TOKEN")
    except credentials.CredentialError:
        return ""
    return str(found or "")


def hint(repo: str, status: int) -> str:
    """What to do about a 401 or 403 from ``repo``."""
    base = endpoint()
    if token():
        return (f"{repo} refused the token in $HF_TOKEN (HTTP {status}). Accept its licence "
                f"at {base}/{repo} with the account that owns the token.")
    return (f"{repo} needs a Hugging Face account (HTTP {status}). Accept its licence at "
            f"{base}/{repo}, create a token at {base}/settings/tokens and set HF_TOKEN.")


@dataclass(frozen=True, slots=True)
class Ref:
    """A parsed model reference: ``hf:owner/repo[/path/file.gguf][:QUANT][@revision]``."""

    repo: str
    file: str = ""
    quant: str = ""
    revision: str = "main"

    @property
    def text(self) -> str:
        tail = f"/{self.file}" if self.file else ""
        return f"hf:{self.repo}{tail}"


def parse(ref: str) -> Ref:
    """Split a reference into repository, file, quantisation tag and revision."""
    text = ref.strip()
    text = text[3:] if text.startswith("hf:") else text
    text, _, revision = text.partition("@")
    text, _, quant = text.partition(":")
    parts = [p for p in text.split("/") if p]
    if len(parts) < 2 or (quant and not _QUANT_TAG.match(quant)):
        raise ValueError(f"{ref!r} should look like hf:owner/repo[/file.gguf][:QUANT]")
    return Ref(f"{parts[0]}/{parts[1]}", "/".join(parts[2:]), quant, revision or "main")


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One file in a repository."""

    path: str
    size: int
    sha256: str = ""
    oid: str = ""
    """The git blob id, which names a small file that is not LFS in the Hub cache."""

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def quantization(self) -> str:
        found = QUANT.search(self.name)
        return found.group(1).upper() if found else ""

    @property
    def companion(self) -> bool:
        return bool(aside(self.name))

    @property
    def build(self) -> str:
        """The folder of a quantisation, or the shard-less name of a single file."""
        if "/" in self.path:
            return self.path.split("/")[0]
        return _SHARD.sub("", self.name)


def _get(url: str, *, auth: str, repo: str) -> object:
    try:
        return net.default().json(url, net.Ask(purpose="model hub", token=auth, tries=3))
    except http.ServerError as exc:
        if exc.status in (401, 403):
            raise GatedRepo(hint(repo, exc.status)) from exc
        if exc.status == 404:
            raise NotFound(f"{repo} is not on {endpoint()}") from exc
        raise RemoteError(str(exc)) from exc
    except Refused as exc:
        raise RemoteError(str(exc)) from exc


def listing(repo: str, revision: str = "main", auth: str | None = None) -> list[RemoteFile]:
    """Every file of a repository with its size and, for large files, its sha256."""
    quoted = urllib.parse.quote(repo, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    got = _get(f"{endpoint()}/api/models/{quoted}/tree/{rev}?recursive=true",
               auth=token() if auth is None else auth, repo=repo)
    out = []
    for row in got if isinstance(got, list) else ():
        if isinstance(row, dict) and row.get("type") == "file":
            lfs = row.get("lfs") if isinstance(row.get("lfs"), dict) else {}
            out.append(RemoteFile(str(row["path"]), int(lfs.get("size") or row.get("size") or 0),
                                  str(lfs.get("oid") or ""), str(row.get("oid") or "")))
    return out


def commit(repo: str, revision: str = "main", auth: str | None = None) -> str:
    """The 40-character commit a revision of a repository resolves to, as the Hub cache names it."""
    if re.fullmatch(r"[a-f0-9]{40}", revision):
        return revision
    quoted = urllib.parse.quote(repo, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    got = _get(f"{endpoint()}/api/models/{quoted}/revision/{rev}",
               auth=token() if auth is None else auth, repo=repo)
    sha = got.get("sha") if isinstance(got, dict) else None
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise RemoteError(f"{endpoint()} did not name a commit for {repo}@{revision}")
    return sha


def members(files: list[RemoteFile], ref: Ref) -> list[RemoteFile]:
    """The files a reference means: the named file and the rest of its shards, or the
    build whose name carries the quantisation tag; an empty list when none match."""
    ggufs = [f for f in files if f.path.lower().endswith(".gguf") and not f.companion]
    if ref.file:
        wanted = next((f for f in files if f.path == ref.file), None)
        if wanted is None:
            return []
        if not _SHARD.search(wanted.name):
            return [wanted]
        stem, folder = _SHARD.sub("", wanted.name), wanted.path.rpartition("/")[0]
        return sorted((f for f in files if f.path.rpartition("/")[0] == folder
                       and _SHARD.sub("", f.name) == stem), key=lambda f: f.path)
    tag = ref.quant.lower()
    chosen = [f for f in ggufs if tag in f.path.lower()] if tag else []
    if not chosen:
        return []
    first = min(chosen, key=lambda f: f.path)
    return [f for f in chosen if f.build == first.build]


def projector(files: list[RemoteFile]) -> RemoteFile | None:
    """The vision projector in a listing, the most precise one."""
    from_repo = [f for f in files if f.name.lower().startswith("mmproj")
                 and f.name.lower().endswith(".gguf")]
    order = ("f32", "bf16", "f16", "q8_0")
    ranked = sorted(from_repo, key=lambda f: next(
        (i for i, q in enumerate(order) if q in f.name.lower()), len(order)))
    return ranked[0] if ranked else None


@dataclass(frozen=True, slots=True)
class Repo:
    """A repository found by a search."""

    id: str
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    files: tuple[RemoteFile, ...] = ()

    def builds(self) -> list[tuple[str, int, int, str]]:
        """``(build, total bytes, shards, quantisation)`` per build, largest first."""
        grouped: dict[str, list[RemoteFile]] = {}
        for one in self.files:
            if one.path.lower().endswith(".gguf") and not one.companion:
                grouped.setdefault(one.build, []).append(one)
        rows = [(name, sum(f.size for f in fs), len(fs), fs[0].quantization)
                for name, fs in grouped.items()]
        return sorted(rows, key=lambda r: -r[1])


@dataclass(frozen=True, slots=True)
class Filters:
    """What a search keeps: repositories with a build of at most ``max_bytes`` (0 for any),
    of ``quant`` (``Q4_K_M``), from ``owner``, and not gated unless ``gated``."""

    max_bytes: int = 0
    quant: str = ""
    owner: str = ""
    gated: bool = True
    limit: int = 20
    files: bool = True


def _keep(repo: Repo, want: Filters) -> Repo | None:
    if want.owner and repo.id.split("/")[0].lower() != want.owner.lower():
        return None
    if repo.gated and not want.gated:
        return None
    if not want.files:
        return repo
    kept = tuple(
        f for f in repo.files
        if (not want.quant or want.quant.lower() in f.name.lower() or f.companion)
        and (not want.max_bytes or f.size <= want.max_bytes or f.companion))
    if not any(not f.companion and f.path.lower().endswith(".gguf") for f in kept):
        return None
    return replace(repo, files=kept)


def search(query: str, filters: Filters | None = None) -> list[Repo]:
    """GGUF repositories matching ``query``, most downloaded first, with their files.

    A repository with no file that passes the filters is left out. Each result costs one
    listing request; ``Filters(files=False)`` skips them.
    """
    want = filters or Filters()
    auth = token()
    params = urllib.parse.urlencode({"search": query, "filter": "gguf", "sort": "downloads",
                                     "direction": "-1", "limit": str(max(want.limit * 2, 10))})
    got = _get(f"{endpoint()}/api/models?{params}", auth=auth, repo="search")
    out: list[Repo] = []
    for row in got if isinstance(got, list) else ():
        if not isinstance(row, dict) or "id" not in row:
            continue
        repo = Repo(str(row["id"]), int(row.get("downloads") or 0), int(row.get("likes") or 0),
                    bool(row.get("gated")))
        if want.files:
            try:
                repo = replace(repo, files=tuple(listing(repo.id, "main", auth)))
            except RemoteError:
                continue
        kept = _keep(repo, want)
        if kept:
            out.append(kept)
        if len(out) >= want.limit:
            break
    return out


MOST_TEXT = 1 << 20


def text_file(repo: str, path: str, revision: str = "main") -> str:
    """A small text file of a repository (its README), cleaned of invisible characters and
    cut at one MiB. It is the publisher's text and is untrusted. `NotFound` when absent."""
    rev = urllib.parse.quote(revision, safe="")
    url = f"{endpoint()}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
    try:
        got = net.default().get(url, net.Ask(purpose="model hub", token=token(), max_bytes=MOST_TEXT))
    except Refused as exc:
        raise RemoteError(str(exc)) from exc
    if got.status in (401, 403):
        raise GatedRepo(hint(repo, got.status))
    if got.status >= 400:
        raise NotFound(f"{repo} has no {path}")
    return clean_text(got.body.decode("utf-8", "replace"))[0]


def models(term: str, limit: int = 100) -> list[dict[str, object]]:
    """The raw model rows of a Hub search for ``term`` (id, downloads, likes)."""
    params = urllib.parse.urlencode({"search": term, "limit": str(limit)})
    got = _get(f"{endpoint()}/api/models?{params}", auth=token(), repo="search")
    return [row for row in got if isinstance(row, dict) and "id" in row] if isinstance(got, list) else []
