"""Package commands with admitted indexes, artifact URLs and Git over HTTPS."""

from __future__ import annotations

import ast
import os
import subprocess
import tempfile
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement

from ml_stack.httpguard import Refused
from ml_stack.net import git, policy

NETWORK = {"install", "wheel", "download"}
OPTIONS = {"-i": "index-url", "--index-url": "index-url",
           "--extra-index-url": "extra-index-url", "-f": "find-links", "--find-links": "find-links"}


def _pip(python, args, environment, timeout):
    return subprocess.run([str(python), "-m", "pip", *args], capture_output=True,
                          text=True, timeout=timeout, env=environment)


def _config(python, command, environment, timeout):
    result = _pip(python, ["config", "list"], environment, timeout)
    if result.returncode:
        raise Refused("pip configuration could not be checked")
    values = {}
    for scope in ("global", command, ":env:"):
        for line in result.stdout.splitlines():
            name, separator, value = line.partition("=")
            if not separator or not name.startswith(scope + "."):
                continue
            try:
                values[name.removeprefix(scope + ".")] = str(ast.literal_eval(value))
            except (SyntaxError, ValueError) as exc:
                raise Refused("invalid pip configuration") from exc
    return values


def _options(args, configured):
    values = {name: configured.get(name, "").split() for name in ("index-url", "extra-index-url", "find-links")}
    offline = configured.get("no-index", "").lower() in {"1", "true", "yes", "on"}
    remaining = []
    iterator = iter(args)
    for argument in iterator:
        if argument.startswith(("-r", "-c", "-e")) and not argument.startswith("--"):
            raise Refused("unchecked package requirement files are not supported")
        if argument.startswith(("-i", "-f")) and not argument.startswith("--") and len(argument) > 2:
            option, separator, supplied = argument[:2], "=", argument[2:]
        else:
            option, separator, supplied = argument.partition("=")
        if option in OPTIONS:
            value = supplied if separator else next(iterator, "")
            if not value:
                raise Refused("pip index and find-links options need a value")
            name = OPTIONS[option]
            if name == "index-url":
                values[name] = [value]
            else:
                values[name].append(value)
        elif option == "--no-index":
            offline = True
        elif option in {"--trusted-host", "--proxy", "--isolated", "--editable", "--build-constraint",
                         "--target", "--prefix", "--root", "--user", "-r", "--requirement", "-c", "--constraint"}:
            raise Refused(f"unsupported package network option: {option}")
        else:
            remaining.append(argument)
    return remaining, values, offline


def _admit(value, hosts):
    if value.startswith(("//", "\\\\")):
        raise Refused("remote filesystem package sources are not supported")
    if value.startswith("git+"):
        git.guarded(value.removeprefix("git+"), hosts)
        return
    parsed = urlsplit(value)
    scheme = parsed.scheme
    if scheme == "file":
        if parsed.netloc:
            raise Refused("remote filesystem package sources are not supported")
        return
    windows = PureWindowsPath(value)
    if not scheme or (len(scheme) == 1 and windows.is_absolute() and not parsed.netloc):
        return
    if scheme != "https":
        raise Refused("package sources require HTTPS")
    standard = (parsed.hostname == "pypi.org" and parsed.path.rstrip("/") == "/simple")
    artifact = parsed.hostname == "files.pythonhosted.org"
    if (standard or artifact) and parsed.port in (None, 443) and not parsed.username:
        hosts = policy.Policy((*hosts.allowed, parsed.hostname), path=hosts.path, clock=hosts.clock)
    hosts.admit(value, "Python packages")


def run(python, args, *, timeout, env=None):
    """Run pip after admitting its configured package sources."""
    environment = dict(os.environ if env is None else env)
    if not args or args[0] not in NETWORK:
        if not args or args[0] not in {"inspect", "uninstall", "check", "show"}:
            raise Refused("unsupported pip command")
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return _pip(python, args, environment, timeout)
    configured = _config(python, args[0], environment, timeout)
    for name in ("requirement", "constraint", "build-constraint", "editable", "target", "prefix", "root"):
        if configured.get(name):
            raise Refused(f"unsupported package source or target configuration: {name}")
    if configured.get("user", "").lower() in {"1", "true", "yes", "on"}:
        raise Refused("package installs use the allocated interpreter environment")
    remaining, values, offline = _options(args, configured)
    for name in ("trusted-host", "proxy"):
        if not offline and configured.get(name):
            raise Refused(f"unsupported package network configuration: {name}")
    hosts = policy.default()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if environment.get(name):
            if offline:
                environment.pop(name)
            else:
                _admit(environment[name], hosts)
    if not offline:
        values["index-url"] = values["index-url"] or ["https://pypi.org/simple"]
        for name in ("index-url", "extra-index-url"):
            for value in values[name]:
                _admit(value, hosts)
    for value in values["find-links"]:
        _admit(value, hosts)
    for argument in remaining[1:]:
        if argument.startswith("-"):
            continue
        try:
            requirement = Requirement(argument)
            if requirement.url:
                _admit(requirement.url, hosts)
        except InvalidRequirement:
            _admit(argument, hosts)
    rewritten = [remaining[0]]
    if offline:
        rewritten.append("--no-index")
    for name, entries in values.items():
        if offline and name != "find-links":
            continue
        for value in entries:
            rewritten.extend(["--" + name, value])
    rewritten.extend(remaining[1:])
    with tempfile.TemporaryDirectory(prefix="ml-stack-pip-") as temporary:
        guarded = git.environment("https", Path(temporary))
        for name in list(environment):
            if name.startswith("GIT_") or name in {
                "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_NO_INDEX",
                "PIP_TRUSTED_HOST", "PIP_PROXY", "PIP_CONFIG_FILE"}:
                environment.pop(name, None)
        environment.update({name: value for name, value in guarded.items() if name.startswith("GIT_")})
        environment.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1")
        return _pip(python, rewritten, environment, timeout)
