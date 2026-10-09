"""Labels for the commands of a shell line: what each program does with the arguments it is given.

`scan` parses a line with `poolhouse.guard.shellscan` and returns one finding per thing it does.
Nothing is run and nothing is expanded; a command the table does not know is ``unsure``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from poolhouse.guard import codescan, workspacecmds
from poolhouse.guard.harm import Finding, outside, protected, resolve
from poolhouse.guard.shellscan import Segment, has_expansion, has_glob, segments
from poolhouse.guard.verbs import verb_finding, words_of

__all__ = ["Ctx", "analyse", "scan"]

MAX_DEPTH = 3
BULK = 25
NULLS = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout", "/dev/tty", "-"})
D, R, S, U = "destructive", "reversible", "safe", "unsure"


@dataclass(frozen=True, slots=True)
class Ctx:
    """What a command is read against: the ``roots`` writes may go to, how deep the line is
    nested and whether another command's output feeds the arguments (``fed``)."""

    roots: tuple[str, ...] = ()
    depth: int = 0
    fed: bool = False


READ_ONLY = frozenset([
    "ls", "cat", "head", "tail", "pwd", "echo", "printf", "wc", "grep", "egrep", "fgrep", "rg",
    "ag", "ack", "which", "type", "whoami", "date", "df", "du", "stat", "file", "uname",
    "printenv", "id", "hostname", "ps", "pgrep", "top", "htop", "diff", "cmp", "uniq", "cut",
    "tr", "basename", "dirname", "realpath", "readlink", "tree", "less", "more", "jq", "yq",
    "true", "false", "test", "[", "[[", "seq", "sleep", "nl", "tac", "rev", "fold", "column",
    "paste", "comm", "join", "od", "hexdump", "xxd", "strings", "md5", "md5sum", "shasum",
    "sha1sum", "sha256sum", "cksum", "lsof", "uptime", "free", "vmstat", "ping", "host", "dig",
    "nslookup", "traceroute", "whereis", "man", "help", "cal", "groups", "locale", "ulimit",
    "sort", "tput", "clear", "lscpu", "nproc", "arch", "sw_vers", "system_profiler", "ioreg",
    "pmset", "jobs", "wait", "getconf", "expr", "bc", "tty", "stty", "column", "fmt", "expand",
    "unexpand", "look", "nm", "otool", "ldd", "size", "lipo", "journalctl", "dmesg", "last",
    "who", "w"
])
REVERSIBLE_CMDS = frozenset([
    "mkdir", "touch", "black", "ruff", "prettier", "isort", "autopep8", "gofmt", "rustfmt",
    "clang-format", "make", "cmake", "ninja", "pytest", "tox", "nox", "mypy", "pyright", "tsc",
    "eslint", "javac", "gcc", "g++", "clang", "cc", "cd", "pushd", "popd", "export", "set",
    "unset", "open", "code", "mktemp", "tar", "unzip", "zip", "gzip", "gunzip", "bzip2", "xz",
    "npm", "yarn", "pnpm", "cargo", "go", "uv", "poetry", "pipx", "gradle", "mvn", "rake",
    "bundle", "swift", "xcodebuild", "brew-services"
])
SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh", "ash"})
INTERPRETERS = frozenset({"python", "python3", "node", "nodejs", "ruby", "perl", "php", "lua",
                          "deno", "bun", "osascript", "pwsh", "powershell"})
EVALS = frozenset({"eval", "source", ".", "exec", "alias", "trap", "builtin", "coproc", "fc"})
DOWNLOADERS = frozenset({"curl", "wget", "fetch", "http", "https", "xh", "aria2c"})
SENDERS = frozenset([
    "scp", "sftp", "ftp", "nc", "ncat", "netcat", "socat", "telnet", "ssh", "mail", "mailx",
    "sendmail", "mutt", "mosh", "rclone", "s3cmd", "gsutil", "tftp", "lftp", "rcp"
])
STOPPERS = frozenset([
    "kill", "pkill", "killall", "skill", "xkill", "shutdown", "reboot", "halt", "poweroff",
    "init", "telinit", "swapoff", "umount", "lvremove", "vgremove", "pvremove", "rmmod",
    "userdel", "groupdel", "deluser", "delgroup", "iptables", "ip6tables", "ufw", "pfctl", "nft",
    "mkfs", "newfs", "mke2fs", "wipefs", "fdisk", "sfdisk", "gdisk", "parted", "format", "srm",
    "shred", "rm", "rmdir", "unlink", "truncate", "mkswap", "tune2fs", "resize2fs", "cryptsetup",
    "vipw", "visudo"
])
WRAPPERS = frozenset({"sudo", "doas", "env", "command", "nohup", "time", "nice", "ionice",
                      "stdbuf", "timeout", "xargs", "watch", "caffeinate", "setsid", "chronic",
                      "unbuffer", "exec"})
VALUE_FLAGS = {"sudo": "ugChpCDRTU", "doas": "uC", "env": "uC", "nice": "n", "ionice": "cnp",
               "stdbuf": "ioe", "timeout": "ks", "xargs": "IiLnPsdEaJ", "watch": "ndp"}
LOADER_VARS = frozenset({"LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
                         "PATH", "BASH_ENV", "ENV", "IFS", "PROMPT_COMMAND", "PYTHONPATH", "NODE_OPTIONS",
                         "PYTHONSTARTUP", "PERL5OPT", "RUBYOPT"})
HELP = frozenset({"--help", "--version", "-V", "-h", "-version", "version"})


def _base(name: str) -> str:
    return Path(name).name


def split_args(args: list[str]) -> tuple[set[str], list[str]]:
    """The flags (short ones one letter each, long ones with ``--``) and the other words."""
    flags: set[str] = set()
    words: list[str] = []
    rest = False
    for a in args:
        if rest or not a.startswith("-") or a == "-":
            words.append(a)
        elif a == "--":
            rest = True
        elif a.startswith("--"):
            flags.add(a.split("=", 1)[0])
        else:
            flags.update(a[1:])
    return flags, words


def _where(paths: list[str], ctx: Ctx) -> str:
    if any(protected(p) and outside(p, ctx.roots) for p in paths):
        return " in a protected directory"
    if any(outside(p, ctx.roots) for p in paths):
        return " outside the project"
    return ""


def _risky(paths: list[str], ctx: Ctx) -> bool:
    return any(has_expansion(p) or has_glob(p) for p in paths)


def _rm(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    flags, paths = split_args(args)
    recursive = " -r" if flags & {"r", "R", "--recursive"} else ""
    many = f" ({len(paths)} paths)" if len(paths) > BULK else ""
    return [Finding(D, f"deletes files ({name}{recursive}){_where(paths, ctx)}{many}")]


def _target_exists(sources: list[str], target: str, ctx: Ctx) -> bool:
    full = resolve(target, ctx.roots)
    if not full.exists() and not full.is_symlink():
        return False
    if full.is_dir():
        return any((full / Path(s.rstrip("/")).name).exists() for s in sources)
    return True


def _mv_cp(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    flags, words = split_args(args)
    if len(words) < 2:
        return [Finding(U, f"{name} without two paths")] if not ctx.fed else [
            Finding(U, f"{name} on paths another command supplies")]
    *sources, target = words
    if _risky(words, ctx):
        return [Finding(U, f"{name} with a pattern or variable in a path (what it touches is not known)")]
    where = _where(words, ctx)
    if where == " in a protected directory":
        return [Finding(D, f"{name} into a protected directory")]
    if not flags & {"n", "--no-clobber"} and _target_exists(sources, target, ctx):
        return [Finding(D, f"{name} over an existing file ({target}){where}")]
    if where:
        return [Finding(D, f"{name} writes{where}")]
    return [Finding(R, f"{name} to a new path")]


def _tee(args: list[str], ctx: Ctx) -> list[Finding]:
    flags, files = split_args(args)
    if not files:
        return [Finding(S, "tee with no file only copies its input")]
    if flags & {"a", "--append"}:
        return [Finding(R, f"appends to a file{_where(files, ctx)}")]
    return [Finding(D, f"overwrites a file (tee){_where(files, ctx)}")]


def _dd(args: list[str], ctx: Ctx) -> list[Finding]:
    out = [a[3:] for a in args if a.startswith("of=")]
    if not out:
        return [Finding(S, "dd with no output file writes to the terminal")]
    return [Finding(D, f"writes raw data over {out[0]}{_where(out, ctx)}")]


def _chmod(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    flags, words = split_args(args)
    if flags & {"R", "--recursive"}:
        return [Finding(D, f"changes {name} on a whole tree{_where(words, ctx)}")]
    return [Finding(R, f"changes {name} of a file{_where(words, ctx)}")]


def _ln(args: list[str], ctx: Ctx) -> list[Finding]:
    flags, _ = split_args(args)
    if flags & {"f", "--force"}:
        return [Finding(D, "replaces an existing file with a link (ln -f)")]
    return [Finding(R, "creates a link")]


def _sed(args: list[str], ctx: Ctx) -> list[Finding]:
    flags, words = split_args(args)
    if flags & {"i", "--in-place"} or any(a.startswith("-i") for a in args):
        return [Finding(D, "edits files in place (sed -i)")]
    if any(re.search(r"(^|[;{}\s])[we]\b|/[we]\s*$|/[gpI]*e", a) for a in words[:1]):
        return [Finding(U, "a sed script that can write files or run commands")]
    return [Finding(S, "sed prints its result")]


def _find(args: list[str], ctx: Ctx) -> list[Finding]:
    found: list[Finding] = []
    at = 0
    while at < len(args):
        a = args[at]
        if a == "-delete":
            found.append(Finding(D, "deletes every file find matches (find -delete)"))
        elif a in ("-fprint", "-fprint0", "-fls", "-fprintf"):
            found.append(Finding(D, f"writes find's output to a file ({a})"))
        elif a in ("-exec", "-execdir", "-ok", "-okdir"):
            end = next((i for i in range(at + 1, len(args)) if args[i] in (";", "+", "\\;")),
                       len(args))
            inner = [x for x in args[at + 1:end] if x != "{}"]
            found += analyse(inner, Ctx(ctx.roots, ctx.depth + 1, True)) if inner else [
                Finding(U, "find -exec with no command")]
            at = end
        at += 1
    return found or [Finding(S, "find only lists files")]


def _git(args: list[str], ctx: Ctx) -> list[Finding]:
    rest = list(args)
    while rest and rest[0].startswith("-"):
        rest = rest[2:] if rest[0] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace") else rest[1:]
    if not rest:
        return [Finding(S, "git with no subcommand")]
    sub, more = rest[0], rest[1:]
    flags, words = split_args(more)
    if sub in ("status", "log", "diff", "show", "blame", "describe", "rev-parse", "ls-files",
               "ls-tree", "shortlog", "grep", "reflog", "cat-file", "rev-list", "whatchanged",
               "ls-remote", "fsck", "count-objects", "version", "help", "show-ref", "check-ignore",
               "name-rev", "diff-tree", "merge-base", "var", "archive", "bisect", "notes"):
        if sub == "reflog" and words[:1] in (["expire"], ["delete"]):
            return [Finding(D, "deletes reflog entries (git reflog expire)")]
        return [Finding(S, f"git {sub} only reads")]
    if sub == "reset":
        return [Finding(D, "discards changes (git reset --hard)")] if flags & {"--hard", "--merge", "--keep"} \
            else [Finding(R, "moves the branch or unstages (git reset)")]
    if sub == "clean":
        if flags & {"n", "--dry-run"}:
            return [Finding(S, "git clean as a dry run")]
        return [Finding(D, "deletes untracked files (git clean)")]
    if sub == "restore":
        return [Finding(D, "discards changes to files (git restore)")]
    if sub in ("checkout", "switch"):
        picks_files = sub == "checkout" and len(words) >= 2 and not flags & {"b", "B", "c", "C"}
        if "--" in more or "." in words or picks_files or flags & {"f", "--force", "--discard-changes"}:
            return [Finding(D, f"discards changes to files (git {sub})")]
        return [Finding(R, f"git {sub} changes the branch")]
    if sub == "branch":
        if "D" in flags or (flags & {"d", "--delete"} and flags & {"f", "--force"}):
            return [Finding(D, "deletes a branch (git branch -D)")]
        if flags & {"d", "--delete"}:
            return [Finding(R, "deletes a merged branch (git branch -d)")]
        if flags & {"m", "M", "c", "C"} or (words and not flags & {"l", "--list", "a", "r", "v"}):
            return [Finding(R, "creates or renames a branch")]
        return [Finding(S, "git branch lists branches")]
    if sub == "push":
        return [Finding(D, "sends commits to a remote"
                        + (" and can overwrite its history (force)" if flags & {"f", "--force", "--force-with-lease", "--mirror", "--delete", "d"} or any(w.startswith(("+", ":")) for w in words) else ""))]
    if sub == "stash":
        if words[:1] in (["drop"], ["clear"]):
            return [Finding(D, f"deletes stashed changes (git stash {words[0]})")]
        return [Finding(S, "git stash list or show")] if words[:1] in (["list"], ["show"]) else [Finding(R, "stashes changes")]
    if sub == "commit" and "--amend" in flags:
        return [Finding(D, "rewrites the last commit (git commit --amend)")]
    if sub == "submodule" and words[:1] == ["deinit"]:
        return [Finding(D, "removes a submodule's working tree (git submodule deinit)")]
    if sub in ("gc", "prune", "filter-branch", "filter-repo", "replace", "update-ref", "worktree", "submodule", "remote", "config", "tag", "rebase", "merge", "cherry-pick", "revert", "am", "apply", "pull", "fetch", "clone", "add", "rm", "mv", "commit", "init", "sparse-checkout", "lfs"):
        return _git_changes(sub, flags, words)
    return [Finding(U, f"git {sub} is not a subcommand the classifier knows")]


def _git_changes(sub: str, flags: set[str], words: list[str]) -> list[Finding]:
    if sub == "commit" and "--amend" in flags:
        return [Finding(D, "rewrites the last commit (git commit --amend)")]
    if sub == "submodule" and words[:1] == ["deinit"]:
        return [Finding(D, "removes a submodule's working tree (git submodule deinit)")]
    if (sub in ("gc", "prune") and flags & {"--prune=now", "--prune", "--aggressive", "--prune=all"}) or sub == "prune":
        return [Finding(D, f"permanently deletes unreachable objects (git {sub})")]
    if sub in ("filter-branch", "filter-repo"):
        return [Finding(D, f"rewrites history (git {sub})")]
    if sub == "rm":
        return [Finding(D, "deletes files from the working tree (git rm)")]
    if sub == "worktree" and words[:1] in (["remove"], ["prune"]):
        return [Finding(D, "removes a worktree")]
    if sub == "tag" and flags & {"d", "--delete"}:
        return [Finding(D, "deletes a tag")]
    if sub == "remote" and words[:1] in (["remove"], ["rm"]):
        return [Finding(D, "removes a remote")]
    if sub in ("config", "remote", "tag", "worktree", "submodule") and not words and not flags - {"l", "list", "--list", "v"}:
        return [Finding(S, f"git {sub} lists")]
    if sub == "config" and flags & {"--get", "--get-all", "--list", "l"}:
        return [Finding(S, "git config reads")]
    if sub in ("rebase", "update-ref", "replace", "lfs"):
        return [Finding(D if sub != "lfs" else R, f"rewrites or moves history (git {sub})")]
    if sub in ("pull", "merge", "fetch", "clone", "cherry-pick", "revert", "am", "apply"):
        return [Finding(R, f"git {sub} changes the working tree")]
    return [Finding(R, f"git {sub} changes the repository")]


def _cloud(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    words = [w for a in args for w in words_of(a) if not a.startswith("-")]
    if name == "terraform" or name == "tofu":
        if words[:1] in (["plan"], ["show"], ["validate"], ["output"], ["version"], ["fmt"], ["init"], ["providers"], ["graph"]):
            return [Finding(S if words[0] != "init" else R, f"{name} {words[0]} does not change infrastructure")]
        return [Finding(D, f"{name} {' '.join(words[:2])} changes real infrastructure")]
    hit = verb_finding(words[:4], f"{name} {' '.join(words[:2])}")
    if name in ("aws", "gcloud", "az", "gsutil", "gh", "heroku", "fly", "flyctl", "vercel", "netlify", "wrangler"):
        if hit and hit.label == S:
            return [Finding(S, f"{name} only reads")]
        return [Finding(D, f"{name} {' '.join(words[:3])} changes or sends something to a remote service")]
    if hit is None or any(w in ("exec", "run", "debug", "attach", "cp", "port-forward", "shell", "ssh") for w in words[:3]):
        return [Finding(U, f"{name} with a subcommand the classifier cannot place")]
    return [hit]


def _pkg(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    _, words = split_args(args)
    sub = words[0] if words else ""
    if not sub and name in ("make", "cmake", "ninja", "rake", "gradle", "mvn", "tox", "nox"):
        return [Finding(R, f"{name} builds the project")]
    if sub in ("run", "exec", "x") and len(words) > 1 and verb_finding(words_of(words[1]), "") \
            and verb_finding(words_of(words[1]), "").label == D:
        return [Finding(D, f"{name} runs a script named {words[1]}")]
    if sub in ("remove", "uninstall", "purge", "autoremove", "erase", "del", "delete", "rm", "clean", "prune", "unlink", "unpublish", "deprecate", "cache", "yank", "owner", "logout"):
        return [Finding(D, f"removes packages ({name} {sub})")]
    if sub in ("publish", "push", "upload", "release"):
        return [Finding(D, f"publishes a package ({name} {sub})")]
    if sub in ("list", "ls", "show", "search", "info", "freeze", "check", "outdated", "view", "help", "version", "--version", "config", "tree", "why", "audit", "doctor", "env", "home", "root", "bin", "prefix", "query", "find", "which", "list-installed"):
        return [Finding(S, f"{name} {sub} only reads")]
    if sub in ("install", "add", "i", "update", "upgrade", "ci", "build", "init", "sync", "run", "test", "start", "lock", "dev", "pull", "fetch", "tap", "link", "develop", "exec", "x", "restart", "stop", "services"):
        return [Finding(R, f"{name} {sub} changes installed software")]
    return [Finding(U, f"{name} {sub or ''} is not a subcommand the classifier knows".replace("  ", " "))]


def _docker(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    words = [a for a in args if not a.startswith("-")]
    if any(w in ("build", "buildx", "pull", "tag", "create") for w in words[:2]):
        return [Finding(R, f"{name} {words[0]} builds or fetches an image")]
    if words[:1] in (["images"], ["image"], ["info"], ["stats"], ["top"], ["port"], ["history"], ["events"],
                     ["version"], ["context"], ["buildx"]) and not set(words) & {"rm", "prune", "rmi"}:
        return [Finding(S, f"{name} {words[0]} only reads")]
    if any(w in ("run", "exec", "attach", "cp") for w in words[:3]):
        return [Finding(U, f"{name} starts or enters a container (what runs inside is not read)")]
    if any(w in ("push", "login") for w in words[:3]):
        return [Finding(D, f"{name} sends an image to a registry")]
    hit = verb_finding(words[:3], name)
    if hit is None:
        return [Finding(U, f"{name} {' '.join(words[:2])} is not a subcommand the classifier knows")]
    if hit.label == D:
        return [Finding(D, f"removes or stops-and-deletes containers, images or volumes ({name} {' '.join(words[:2])})")]
    return [hit]


def _service(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    words = [a for a in args if not a.startswith("-")]
    if name == "service" and len(words) > 1:
        words = words[1:]
    verb = next((w for w in words if w in ("stop", "disable", "unload", "bootout", "kill", "mask", "remove", "delete", "reset-failed", "unmask", "uninstall", "deactivate", "isolate", "poweroff", "reboot", "halt", "kickstart", "bootstrap", "load", "start", "restart", "enable", "reload", "daemon-reload", "list", "status", "print", "is-active", "is-enabled", "show", "cat", "list-units", "list-unit-files", "blame", "dumpstate")), "")
    if verb in ("stop", "disable", "unload", "bootout", "kill", "mask", "remove", "delete", "uninstall", "isolate", "poweroff", "reboot", "halt", "deactivate"):
        return [Finding(D, f"stops or removes a system service ({name} {verb})")]
    if verb in ("start", "restart", "enable", "reload", "load", "bootstrap", "unmask", "reset-failed", "daemon-reload", "kickstart"):
        return [Finding(R, f"{name} {verb} changes a service")]
    if verb:
        return [Finding(S, f"{name} {verb} only reads")]
    return [Finding(U, f"{name} with an action the classifier cannot place")]


def _diskutil(args: list[str], ctx: Ctx) -> list[Finding]:
    verb = next((a for a in args if not a.startswith("-")), "")
    if verb in ("list", "info", "activity", "listfilesystems", "listfilesystems", "apfs", "cs", "mount", "unmount", "eject", "unmountdisk", "mountdisk", "verifyvolume", "verifydisk"):
        if verb == "apfs" and any(w in args for w in ("delete", "deleteVolume", "deletecontainer", "eraseVolume")):
            return [Finding(D, "deletes an APFS volume")]
        return [Finding(S if verb in ("list", "info", "activity", "listfilesystems") else R, f"diskutil {verb}")]
    return [Finding(D, f"diskutil {verb or ''} can erase or repartition a disk".replace("  ", " "))]


def _http(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    flags, words = split_args(args)
    joined = " ".join(args).lower()
    sends = bool(flags & {"d", "F", "T", "--data", "--data-raw", "--data-binary", "--data-urlencode", "--form", "--upload-file", "--json", "--post-data", "--post-file", "--body-data", "--body-file", "--method"})
    if re.search(r"(?:^|\s)(?:-x|--request)[= ]?\s*(?:post|put|delete|patch)", joined):
        sends = True
    if name in ("http", "https", "xh") and ((words[:1] and words[0].upper() in ("POST", "PUT", "DELETE", "PATCH"))
                                           or any(re.match(r"^[\w.-]+(=|:=|@)", w) for w in words[1:])):
        sends = True
    if "--method" in flags and any(m in joined for m in ("delete", "put", "post", "patch")):
        sends = True
    if sends:
        return [Finding(D, f"sends data to a remote server ({name})")]
    if flags & {"o", "O", "--output", "--remote-name", "-O"} or name in ("wget", "aria2c"):
        return [Finding(R, f"downloads a file ({name})")]
    return [Finding(S, f"{name} fetches a page")]


def _queued_tests(args, ctx):
    if not args or len(args) < 2 or args[1] not in ("fast", "quick", "all", "full"):
        return False
    if not ctx.roots:
        return False
    script = resolve(args[0], ctx.roots).resolve()
    root = Path(ctx.roots[0]).resolve()
    return (script == root / "scripts" / "test" and script.is_file()
            and (root / ".git").is_file())


def _pyenv(name, args, ctx):
    if args in (["versions"], ["versions", "--bare"], ["version"], ["version-name"]):
        return [Finding(S, "pyenv lists installed interpreters")]
    return [Finding(U, "pyenv changes or executes an interpreter configuration")]


def _interpreter(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    _, words = split_args(args)
    code_flag = next((a for a in args if a in ("-c", "-e", "-r", "-E", "--eval", "-p", "--print", "-pe", "-ne", "-pi", "-lne")), "")
    if (args and args[0] in HELP) or (len(args) == 1 and args[0] in ("-V", "--version", "-v")):
        return [Finding(S, f"{name} prints its version")]
    if name in SHELLS:
        if "-c" in args:
            at = args.index("-c")
            text = args[at + 1] if at + 1 < len(args) else ""
            if ctx.depth >= MAX_DEPTH:
                return [Finding(U, "shell commands nested too deeply to read")]
            inner = scan(text, ctx.roots, ctx.depth + 1)
            return inner or [Finding(U, "an empty shell -c string")]
        return [Finding(U, f"{name} runs a script or reads commands from input")]
    if code_flag:
        at = args.index(code_flag)
        code = args[at + 1] if at + 1 < len(args) else ""
        return [codescan.check(code, name)]
    if name in ("python", "python3") and args[:1] == ["-m"] and len(args) > 1:
        mod = args[1]
        if mod in ("pytest", "unittest", "mypy", "ruff", "black", "isort", "pyright", "build", "venv", "compileall", "py_compile"):
            return [Finding(R, f"python -m {mod} runs project code")]
        if mod == "pip":
            return _pkg("pip", args[2:], ctx)
        if mod in ("json.tool", "http.server", "this", "platform", "sysconfig", "site"):
            return [Finding(S, f"python -m {mod}")]
    if name in ("python", "python3") and _queued_tests(args, ctx):
        return [Finding(R, "runs the maintained test queue in the authorized isolated worktree")]
    if not args:
        return [Finding(U, f"{name} with no program reads commands from input")]
    return [Finding(U, f"{name} runs a program the classifier does not read ({words[0] if words else name})")]


def _pytest(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    if set(args) & {"--collect-only", "--co", "--version"}:
        return [Finding(S, "pytest lists tests without running them")]
    return [Finding(R, "pytest runs project code")]


def _tar(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    flags, _ = split_args(args)
    if "--remove-files" in flags:
        return [Finding(D, "deletes the files it archives (tar --remove-files)")]
    if name == "unzip" and flags & {"o"}:
        return [Finding(D, "overwrites existing files without asking (unzip -o)")]
    return [Finding(R, f"{name} writes files into the project")]


def _awk(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    text = " ".join(args)
    if re.search(r"system\s*\(|\|\s*getline|print[^;}]*>|\|\s*\"", text):
        return [Finding(U, f"{name} program that can run commands or write files")]
    return [Finding(S, f"{name} prints its result")]


def _sort(args: list[str], ctx: Ctx) -> list[Finding]:
    flags, _ = split_args(args)
    if "o" in flags or "--output" in flags or any(a.startswith("--output=") for a in args):
        return [Finding(D, "sort -o overwrites a file")]
    return [Finding(S, "sort prints its result")]


def _crontab(args: list[str], ctx: Ctx) -> list[Finding]:
    flags, _ = split_args(args)
    return [Finding(D, "replaces or removes the cron table")] if flags - {"l"} or not flags else [Finding(S, "lists cron jobs")]


def _rsync(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    flags, _ = split_args(args)
    if any(f.startswith("--delete") or f == "--remove-source-files" for f in flags):
        return [Finding(D, f"{name} deletes files at the destination")]
    return [Finding(D, f"{name} copies files over existing ones or to another machine")]


def _wrapper(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    take = VALUE_FLAGS.get(name, "")
    at = 0
    while at < len(args):
        a = args[at]
        if name == "timeout" and not a.startswith("-") and re.fullmatch(r"[0-9.]+[smhd]?", a):
            at += 1
            break
        if name in ("env", "sudo", "doas") and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", a):
            at += 1
        elif a.startswith("-") and a != "-":
            at += 2 if len(a) == 2 and a[1] in take else 1
        else:
            break
    inner = args[at:]
    if not inner:
        return [Finding(S if name in ("env", "time", "command") else R, f"{name} with no command")]
    fed = ctx.fed or name == "xargs"
    found = analyse(inner, Ctx(ctx.roots, ctx.depth + 1, fed))
    if name in ("sudo", "doas"):
        found.append(Finding(R, "runs with administrator rights"))
    if name == "exec":
        found.append(Finding(U, "exec replaces the shell with another program"))
    return found


Handler = Callable[[str, list[str], Ctx], list[Finding]]
HANDLERS: dict[str, Handler] = {
    "mv": _mv_cp, "cp": _mv_cp, "install": _mv_cp, "rsync": _rsync,
    "tee": lambda n, a, c: _tee(a, c), "dd": lambda n, a, c: _dd(a, c),
    "chmod": _chmod, "chown": _chmod, "chgrp": _chmod, "chflags": _chmod, "setfacl": _chmod,
    "ln": lambda n, a, c: _ln(a, c), "sed": lambda n, a, c: _sed(a, c), "find": lambda n, a, c: _find(a, c),
    "git": lambda n, a, c: _git(a, c), "diskutil": lambda n, a, c: _diskutil(a, c),
    "awk": _awk, "gawk": _awk, "mawk": _awk, "nawk": _awk, "sort": lambda n, a, c: _sort(a, c),
    "crontab": lambda n, a, c: _crontab(a, c), "tar": lambda n, a, c: _tar(n, a, c),
    "unzip": lambda n, a, c: _tar(n, a, c),
}
for _n in ("kubectl", "helm", "terraform", "tofu", "aws", "gcloud", "az", "gsutil", "gh", "heroku", "fly", "flyctl", "vercel", "netlify", "wrangler", "oc", "kustomize", "pulumi", "ansible", "ansible-playbook", "vagrant", "minikube", "kind"):
    HANDLERS[_n] = _cloud
for _n in ("pip", "pip3", "npm", "yarn", "pnpm", "cargo", "gem", "apt", "apt-get", "aptitude", "dpkg", "dnf", "yum", "rpm", "pacman", "apk", "zypper", "conda", "mamba", "brew", "port", "snap", "flatpak", "uv", "poetry", "pipx", "go", "bundle", "composer", "nix", "nix-env", "choco", "winget", "scoop", "gradle", "mvn", "swift", "tox", "nox", "make", "cmake", "ninja", "rake"):
    HANDLERS[_n] = _pkg
for _n in ("docker", "podman", "docker-compose", "nerdctl", "ctr", "crictl", "colima", "lima"):
    HANDLERS[_n] = _docker
for _n in ("systemctl", "launchctl", "service", "rc-service", "sv", "initctl", "brew-services"):
    HANDLERS[_n] = _service
def _sysctl(name: str, args: list[str], ctx: Ctx) -> list[Finding]:
    flags, words = split_args(args)
    if "w" in flags or any("=" in w for w in words):
        return [Finding(D, "changes a kernel setting (sysctl -w)")]
    return [Finding(S, "sysctl reads settings")]


HANDLERS["sysctl"] = _sysctl
HANDLERS["pytest"] = _pytest
HANDLERS["pyenv"] = _pyenv
for _n in DOWNLOADERS:
    HANDLERS[_n] = _http
for _n in (*SHELLS, *INTERPRETERS):
    HANDLERS[_n] = _interpreter
for _n in WRAPPERS:
    HANDLERS[_n] = _wrapper


def _fixed(name: str, args: list[str], ctx: Ctx) -> list[Finding] | None:
    if name in ("rm", "rmdir", "unlink", "shred", "srm"):
        return _rm(name, args, ctx)
    if name == "truncate":
        return [Finding(D, "truncates files" + _where(split_args(args)[1], ctx))]
    if name.startswith("mkfs") or name in STOPPERS:
        what = "formats or repartitions a disk" if name.startswith(("mkfs", "mke2", "newfs")) or name in (
            "fdisk", "sfdisk", "gdisk", "parted", "format", "wipefs", "mkswap", "cryptsetup") else \
            "stops processes, removes accounts or changes the firewall or system state"
        if name in ("kill", "pkill", "killall", "skill", "xkill"):
            flags, _ = split_args(args)
            if "0" in flags and len(args) <= 2:
                return [Finding(S, f"{name} -0 only checks that a process exists")]
            what = "stops processes"
        return [Finding(D, f"{what} ({name})")]
    if name in SENDERS:
        return [Finding(D, f"sends data to another machine ({name})")]
    return None


def analyse(argv: list[str], ctx: Ctx) -> list[Finding]:
    """The findings for one command, ``argv`` being its words with the assignments removed."""
    if not argv:
        return []
    first = argv[0]
    if has_expansion(first) or has_glob(first):
        return [Finding(U, "the command name comes from a variable or pattern")]
    name = _base(first).lower()
    args = argv[1:]
    if name in EVALS:
        return [Finding(U, f"{name} runs text as commands")]
    if args and args[-1] in HELP and name not in WRAPPERS and name not in ("git",):
        return [Finding(S, f"{name} prints help or its version")]
    fixed = _fixed(name, args, ctx)
    if fixed is not None:
        return fixed
    if name == "poolhouse-workspace":
        return workspacecmds.effect(args)
    if name in HANDLERS:
        return HANDLERS[name](name, args, ctx)
    if name in READ_ONLY:
        return [Finding(S, f"{name} only reads")]
    if name in REVERSIBLE_CMDS:
        paths = [a for a in args if not a.startswith("-")]
        extra = [Finding(D, f"{name} writes into a protected directory")] if any(
            protected(p) and outside(p, ctx.roots) for p in paths if "/" in p) else []
        return [Finding(R, f"{name} changes files in the project"), *extra]
    return [Finding(U, f"{name} is a program the classifier does not know")]


def _redirect(op: str, target: str, ctx: Ctx) -> Finding | None:
    if target in NULLS or target.isdigit() or target.startswith("&") or op.startswith("<"):
        return None
    if has_expansion(target) or has_glob(target):
        return Finding(U, "redirects output to a path built from a variable or pattern")
    where = _where([target], ctx)
    if op in (">>", "&>>"):
        return Finding(R, f"appends to a file{where}")
    return Finding(D, f"overwrites a file with a redirect (>){where}")


def _stdin_code(seg: Segment, prev: Segment | None) -> Finding | None:
    if not (seg.piped and prev and prev.argv and seg.argv):
        return None
    name = _base(seg.argv[0]).lower()
    runs_stdin = (name in SHELLS | INTERPRETERS and not any(
        not a.startswith("-") for a in seg.argv[1:])) or (name in ("sudo", "xargs") and any(
        _base(a).lower() in SHELLS | INTERPRETERS for a in seg.argv[1:3]))
    if not runs_stdin:
        return None
    source = _base(prev.argv[0]).lower()
    if source in DOWNLOADERS:
        return Finding(D, f"runs code downloaded by {source} (pipe to a shell)")
    return Finding(U, f"pipes text into a {name} that runs it as code")


def scan(command: str, roots: tuple[str, ...] = (), depth: int = 0) -> list[Finding]:
    """The findings for a shell line: one per command, redirect and piece of obfuscation."""
    parsed, found = segments(command)
    found = list(found)
    ctx = Ctx(roots, depth)
    prev: Segment | None = None
    for seg in parsed:
        argv = list(seg.argv)
        while argv and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
            if argv[0].split("=", 1)[0] in LOADER_VARS:
                found.append(Finding(U, "sets a variable that changes how programs load code"))
            argv.pop(0)
        piped_ctx = Ctx(roots, depth, seg.piped)
        found += analyse(argv, piped_ctx)
        if prev is not None:
            hit = _stdin_code(seg, prev)
            if hit is not None:
                found.append(hit)
        for op, target in seg.redirects:
            hit = _redirect(op, target, ctx)
            if hit is not None:
                found.append(hit)
        prev = seg
    if not parsed and not found:
        found.append(Finding(S, "an empty command"))
    return found
