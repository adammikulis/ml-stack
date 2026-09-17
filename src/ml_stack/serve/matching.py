"""Whether a server that is already running is serving what a spec asked for."""

from __future__ import annotations

from pathlib import Path

from ml_stack.client.health import ServingParams
from ml_stack.serve.backend import ServerSpec

__all__ = ["model_matches", "repo_only", "serving_mismatch"]


def repo_only(model: str | Path) -> str:
    """The ``owner/repo`` a reference names when it names no file in it; else ""."""
    text = str(model).removeprefix("hf:").strip("/")
    parts = text.split("/")
    return text.lower() if len(parts) == 2 and not parts[1].endswith(".gguf") else ""


def model_matches(reported: str, wanted: str | Path, *, loaded_file: str | None = None) -> bool:
    """Whether a server reporting ``reported`` is serving ``wanted``.

    ``loaded_file`` is what the server's own ``/props`` says it actually loaded (its
    ``model_path``), when a caller has one. A reference to a repository with no file
    in it -- what a server started from a repository reports over ``/v1/models`` --
    is not evidence it holds any particular file of that repository: two files of one
    repository are not the same weights. ``loaded_file`` is what breaks the tie; a
    bare-repository report with no file requested either still matches every file of
    that repository, since nothing named a file to check.
    """
    if loaded_file and repo_only(reported):
        reported = loaded_file
    wanted_repo, reported_repo = repo_only(wanted), repo_only(reported)
    if reported_repo and not wanted_repo:
        # the report names only the repository, a specific file was asked for, and
        # which file actually loaded is not known -- do not assume it is this one
        return False
    repo = reported_repo or wanted_repo
    if repo:
        other = wanted if reported_repo else reported
        return repo in str(other).removeprefix("hf:").lower().replace("_", "/")
    wanted_name = Path(str(wanted).removeprefix("hf:")).name.lower()
    reported_name = Path(reported).name.lower()
    if not wanted_name or not reported_name:
        return False
    return wanted_name in reported_name or reported_name in wanted_name


def serving_mismatch(
    spec: ServerSpec,
    models: list[str],
    params: ServingParams | None,
) -> list[str]:
    """Each field in which a running server differs from ``spec``. Empty when it fits."""
    out: list[str] = []

    loaded_file = params.model if params else None
    if models and not any(model_matches(m, spec.model, loaded_file=loaded_file) for m in models):
        serving = ", ".join(repr(Path(name).name) for name in models)
        asked = Path(str(spec.model).removeprefix("hf:")).name
        out.append(f"model: asked for {asked!r}, serving {serving}")

    if params is None:
        return out

    slots = max(int(spec.parallel or 1), 1)
    if params.total_slots is not None and params.total_slots < slots:
        out.append(f"slots: asked for {slots}, serving {params.total_slots}")

    # llama-server reports the context of one slot: --ctx-size divided by -np.
    per_slot = int(spec.context) // slots
    if params.n_ctx is not None and params.n_ctx < per_slot:
        out.append(f"context: asked for {per_slot} per slot, serving {params.n_ctx}")

    # a spec that brings its own template wants one that renders a late system message;
    # a server still on the model's own, which refuses one, is not what was asked for
    if spec.chat_template_file and params.chat_template:
        from ml_stack.serve.chat_template import needs_forgiving

        if needs_forgiving(params.chat_template):
            out.append("chat template: asked for one that renders a late system message, "
                       "serving one that refuses it")

    return out
