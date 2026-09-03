#!/bin/sh
# Network installer for ml-stack, in four modes. Re-running any of them upgrades in place.
#
#   curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- --headless
#   curl -fsSL .../install.sh | sh -s -- --dev
#   curl -fsSL .../install.sh | sudo sh -s -- --system
#
#   (default)   the app: the release zip for this machine, a window, updates from releases
#   --headless  a venv under ~/.ml-stack, console scripts on PATH, no window
#   --dev       a git checkout with an editable install, following main
#   --system    --headless, per machine: starts at boot, no login, as the user who ran it
#   --uninstall takes it off, and leaves the model cache alone
#
# Every step past the install is an ml-stack command, not shell: `ml-stack-serve build`,
# `ml-stack-setup`, `ml-stack-models fetch`, `ml-stack-fleet join`, `ml-stack-doctor`.
# Nothing here reimplements what one of those already does.
#
# Answer every prompt with the environment and it runs unattended (a machine with no
# terminal is never prompted at all):
#   ML_STACK_MODE=app|headless|dev|system   ML_STACK_NAME=<this machine>
#   ML_STACK_PASSPHRASE=<the words every machine shares>   ML_STACK_CLUSTER=<group>
#   ML_STACK_MODELS=auto|default|none|<word>   ML_STACK_ADOPT_CACHE=yes|no
#   ML_STACK_REF=main|v1.2.3   ML_STACK_BUILD=release|source
#   ML_STACK_OFFLINE_ZIP=/path/to.zip   ML_STACK_OFFLINE_MODELS=/dir   (no network at all)
set -eu

REPO="${ML_STACK_REPO:-adammikulis/ml-stack}"
API="https://api.github.com/repos/$REPO/releases/latest"
GIT_URL="https://github.com/$REPO"
EXTRAS="store,hub,web,plot"
MODE="${ML_STACK_MODE:-app}"
MODELS="${ML_STACK_MODELS:-}"
REF="${ML_STACK_REF:-}"
ADOPT="${ML_STACK_ADOPT_CACHE:-}"
OFFLINE_ZIP="${ML_STACK_OFFLINE_ZIP:-}"
OFFLINE_MODELS="${ML_STACK_OFFLINE_MODELS:-}"
UNINSTALL=no
PY=""
BIN=""
TRACK=""

say() { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
interactive() { [ -t 0 ] && [ -t 1 ]; }

while [ $# -gt 0 ]; do
  case "$1" in
    --headless) MODE=headless ;;
    --dev)      MODE=dev ;;
    --system)   MODE=system ;;
    --app)      MODE=app ;;
    --uninstall) UNINSTALL=yes ;;
    --adopt-cache) ADOPT=yes ;;
    --no-adopt-cache) ADOPT=no ;;
    --models) shift; MODELS="${1:-none}" ;;
    --ref)    shift; REF="${1:-}" ;;
    --help|-h) sed -n '2,25p' "$0" 2>/dev/null || say "see the header of install.sh"; exit 0 ;;
    *) die "unknown option $1 (--headless, --dev, --system, --uninstall)" ;;
  esac
  shift
done

case "$(uname -s)" in
  Darwin) OS=macos ;;
  Linux)  OS=linux ;;
  *) die "unsupported system: $(uname -s). Download it from $GIT_URL/releases/latest" ;;
esac
case "$(uname -m)" in
  arm64|aarch64) ARCH=arm64 ;;
  x86_64|amd64)  ARCH=x86_64 ;;
  *) die "unsupported processor: $(uname -m)" ;;
esac
[ "$OS" = linux ] && ARCH=x86_64
KEY="ml-stack-$OS-$ARCH"

# -- python -------------------------------------------------------------------
# Say how to get one; never install a system Python behind somebody's back.
find_python() {
  for candidate in python3.13 python3.12 python3.11 python3; do
    if have "$candidate" && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
      PY="$(command -v "$candidate")"
      return 0
    fi
  done
  if [ "$OS" = macos ]; then
    die "ml-stack needs Python 3.11 or newer. Install it with:  brew install python@3.13
     (or from https://www.python.org/downloads/macos/), then run this again."
  fi
  die "ml-stack needs Python 3.11 or newer. Install it with:
       sudo apt install python3 python3-venv   (Debian, Ubuntu)
       sudo dnf install python3                (Fedora, RHEL)
     then run this again."
}

# -- the app (default) --------------------------------------------------------
install_app() {
  step "the app"
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  if [ -n "$OFFLINE_ZIP" ]; then
    say "installing from $OFFLINE_ZIP; no network step will run"
    cp "$OFFLINE_ZIP" "$TMP/pkg.zip"
  else
    have curl || die "this needs curl"
    say "looking for the newest ml-stack for $OS $ARCH"
    JSON=$(curl -fsSL -H 'Accept: application/vnd.github+json' "$API") \
      || die "could not reach GitHub"
    TAG=$(printf '%s' "$JSON" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
    URL=$(printf '%s' "$JSON" \
      | tr ',' '\n' | grep 'browser_download_url' | grep "$KEY" \
      | sed -n 's/.*"\(https[^"]*\)".*/\1/p' | head -1)
    [ -n "${URL:-}" ] || die "release ${TAG:-latest} has no download for $KEY"
    say "downloading $TAG"
    curl -fL# -o "$TMP/pkg.zip" "$URL" || die "download failed"
  fi
  have unzip || die "this needs unzip"
  unzip -q "$TMP/pkg.zip" -d "$TMP/out" || die "the download could not be unpacked"

  if [ "$OS" = macos ] && [ -d "$TMP/out/ml-stack.app" ]; then
    DEST="${ML_STACK_DEST:-/Applications}"
    [ -w "$DEST" ] || DEST="$HOME/Applications"
    mkdir -p "$DEST"
    rm -rf "$DEST/ml-stack.app"
    cp -R "$TMP/out/ml-stack.app" "$DEST/ml-stack.app"
    # Downloads are quarantined; without this macOS refuses to open it at all.
    xattr -dr com.apple.quarantine "$DEST/ml-stack.app" 2>/dev/null || true
    say ""
    say "Installed to $DEST/ml-stack.app"
    say "Open it, and type the same passphrase you used on your other machines."
    say "It downloads gemma-4-E2B on first run (2.6G, about 1.5s a question) and offers"
    say "the bigger models this machine has room for."
    open "$DEST/ml-stack.app" 2>/dev/null || true
  else
    DEST="${ML_STACK_DEST:-$HOME/.local/bin}"
    mkdir -p "$DEST"
    for name in ml-stack ml-stack-headless; do
      [ -f "$TMP/out/$name" ] || continue
      install -m 0755 "$TMP/out/$name" "$DEST/$name"
    done
    say ""
    say "Installed to $DEST"
    case ":$PATH:" in
      *":$DEST:"*) say "Run: ml-stack" ;;
      *)           say "Run: $DEST/ml-stack   (or add $DEST to your PATH)" ;;
    esac
  fi
}

# -- headless: a venv and the console scripts ---------------------------------
venv_root() {
  if [ "$MODE" = system ]; then
    printf '%s' "${ML_STACK_PREFIX:-/opt/ml-stack}/venv"
  else
    printf '%s' "${ML_STACK_PREFIX:-$HOME/.ml-stack}/venv"
  fi
}

make_venv() {
  VENV="$1"
  find_python
  say "python: $PY"
  [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" \
    || die "could not make a virtualenv at $VENV (on Debian: sudo apt install python3-venv)"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  BIN="$VENV/bin"
}

install_headless() {
  step "headless"
  make_venv "$(venv_root)"
  if [ -n "$OFFLINE_ZIP" ]; then
    say "installing from $OFFLINE_ZIP; no network step will run"
    "$BIN/pip" install --quiet "$OFFLINE_ZIP" || die "could not install $OFFLINE_ZIP"
  else
    WANT="$REF"
    if [ -z "$WANT" ]; then
      WANT=$(curl -fsSL -H 'Accept: application/vnd.github+json' "$API" 2>/dev/null \
        | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
      [ -n "$WANT" ] || WANT=main
    fi
    say "installing ml-stack[$EXTRAS] at $WANT"
    "$BIN/pip" install --quiet --upgrade \
      "ml-stack[$EXTRAS] @ git+$GIT_URL@$WANT" || die "pip could not install ml-stack"
    # The ref decides how it keeps itself current: a tag follows releases, main follows main.
    case "$WANT" in
      main|master) TRACK="$WANT" ;;
      *)           TRACK="" ;;
    esac
  fi
  link_scripts "$BIN"
}

link_scripts() {
  FROM="$1"
  TO="${ML_STACK_DEST:-$HOME/.local/bin}"
  [ "$MODE" = system ] && TO="${ML_STACK_DEST:-/usr/local/bin}"
  mkdir -p "$TO" 2>/dev/null || true
  for name in ml-stack ml-stack-traind ml-stack-fleet ml-stack-peers ml-stack-serve \
              ml-stack-models ml-stack-bench ml-stack-setup ml-stack-doctor ml-stack-mcp; do
    [ -x "$FROM/$name" ] || continue
    ln -sf "$FROM/$name" "$TO/$name" 2>/dev/null || true
  done
  say "console scripts in $TO"
  case ":$PATH:" in
    *":$TO:"*) : ;;
    *) say "add $TO to your PATH to run them by name" ;;
  esac
}

# -- dev: a checkout that follows main ----------------------------------------
install_dev() {
  step "developer"
  have git || die "this needs git"
  SRC="${ML_STACK_SRC:-$HOME/.local/share/ml-stack/src}"
  if [ -d "$SRC/.git" ]; then
    say "updating $SRC"
    git -C "$SRC" pull --ff-only || say "  it has commits main does not; left alone"
  else
    say "cloning into $SRC"
    mkdir -p "$(dirname "$SRC")"
    git clone "$GIT_URL" "$SRC" || die "could not clone $GIT_URL"
  fi
  make_venv "$(venv_root)"
  say "editable install of $SRC"
  (cd "$SRC" && "$BIN/pip" install --quiet -e ".[$EXTRAS]") \
    || die "pip could not install $SRC"
  link_scripts "$BIN"
  TRACK="${ML_STACK_TRACK:-main}"
}

# -- per machine: at boot, as the user who installed it -----------------------
install_system() {
  step "per machine"
  [ "$(id -u)" = 0 ] || die "--system installs for the whole machine, so it needs root:
     curl -fsSL .../install.sh | sudo sh -s -- --system"
  RUNAS="${SUDO_USER:-$(id -un)}"
  [ "$RUNAS" = root ] && die "run this with sudo from your own account, not as root:
     the service is installed to run as you, so it reads the models you already have."
  HOME_DIR=$(eval echo "~$RUNAS")
  say "the service will run as $RUNAS ($HOME_DIR)"

  # The models already on this disk. Running as the installing user means the service opens
  # that cache where it is -- nothing moved, nothing linked, nothing downloaded twice.
  CACHE_ARGS="--same-user"
  if [ "$ADOPT" = yes ]; then CACHE_ARGS="--adopt"; fi
  # shellcheck disable=SC2086
  "$BIN/python" -m ml_stack.fleet.autostart cache \
      --user-cache "$HOME_DIR/.cache/huggingface" $CACHE_ARGS || true

  "$BIN/python" -m ml_stack.fleet.autostart system --user "$RUNAS" --home "$HOME_DIR" \
    || die "could not install the boot service"
  if [ "$OS" = macos ]; then
    # How much memory a model may wire down survives a reboot only as a LaunchDaemon plist,
    # and writing it needs root -- which this has, right now. Asking again later costs
    # another password. Windows has no equivalent: its VRAM is dedicated.
    "$BIN/ml-stack-serve" memory --persist \
      || say "  (the wired limit was not raised; 'ml-stack-serve memory' prints the line)"
  fi
}

# -- after any install --------------------------------------------------------
llama_build() {
  step "llama.cpp"
  [ -x "$BIN/ml-stack-serve" ] || { say "skipped: no ml-stack-serve"; return 0; }
  if [ -n "$OFFLINE_ZIP" ]; then say "offline: skipping the llama.cpp build"; return 0; fi
  FROM=release
  if [ "${ML_STACK_BUILD:-}" = source ] && { have cc || have clang || have gcc; }; then
    FROM=source
  fi
  "$BIN/ml-stack-serve" build --from "$FROM" \
    || say "  the build did not finish; 'ml-stack-serve build' retries"
  # A model whose measured profile names a fork needs that fork; mainline will not load it.
  case "${CHOSEN_BUILD:-}" in
    unsloth)
      "$BIN/ml-stack-serve" build --repo unslothai/llama.cpp --from release --name unsloth \
        || say "  the named fork did not build, and the chosen model needs it" ;;
  esac
}

sizing() {
  step "what this machine can do"
  if [ -x "$BIN/ml-stack-setup" ]; then "$BIN/ml-stack-setup" || true; else say "skipped"; fi
}

fetch_models() {
  step "models"
  WANT="$MODELS"
  if [ -z "$WANT" ]; then
    case "$MODE" in
      app) WANT=default ;;   # gemma-4-E2B: the smallest that still answers
      *)   WANT=auto ;;      # headless and system are power users: the best that fits
    esac
  fi
  if [ "$WANT" = none ]; then say "none asked for"; return 0; fi
  if [ -n "$OFFLINE_MODELS" ]; then
    say "offline: using the models in $OFFLINE_MODELS; nothing is downloaded"
    return 0
  fi
  ROOM=$("$BIN/python" -c \
    'from ml_stack.fleet.bench import machine_room; print(machine_room())' 2>/dev/null || echo 0)
  PICK=$("$BIN/python" -m ml_stack.fleet.autostart choose --room "$ROOM" --want "$WANT" 2>/dev/null || true)
  if [ -z "$PICK" ]; then
    say "no measured model fits this machine's $ROOM bytes; none fetched"
    return 0
  fi
  say "fetching $PICK into the one cache on this machine"
  # `ml-stack-models fetch` checks every download's sha256 and refuses a mismatch.
  # shellcheck disable=SC2086
  "$BIN/ml-stack-models" fetch $PICK \
    || say "  the fetch did not finish; 'ml-stack-models fetch' retries"
}

join_fleet() {
  step "joining the fleet"
  [ -x "$BIN/ml-stack-fleet" ] || { say "skipped: no ml-stack-fleet"; return 0; }
  set -- join --persist
  [ -n "${ML_STACK_NAME:-}" ] && set -- "$@" --name "$ML_STACK_NAME"
  [ -n "${ML_STACK_CLUSTER:-}" ] && set -- "$@" --group "$ML_STACK_CLUSTER"
  [ -n "$TRACK" ] && set -- "$@" --track "$TRACK"
  if [ -n "${ML_STACK_PASSPHRASE:-}" ]; then
    set -- "$@" --passphrase "$ML_STACK_PASSPHRASE"
  elif ! interactive; then
    say "no passphrase, and no terminal to ask at. Set ML_STACK_PASSPHRASE and re-run,"
    say "or run:  ml-stack-fleet join --persist"
    return 0
  fi
  "$BIN/ml-stack-fleet" "$@" || say "  join did not finish; 'ml-stack-fleet join' retries"
}

check_over() {
  step "checking it over"
  if [ -x "$BIN/ml-stack-doctor" ]; then "$BIN/ml-stack-doctor" || true; else say "skipped"; fi
}

last_screen() {
  step "done"
  say "  machine     ${ML_STACK_NAME:-$(hostname)}"
  say "  cluster     ${ML_STACK_CLUSTER:-ml-stack}"
  say "  open        http://127.0.0.1:8770/ui/"
  if [ -x "$BIN/python" ]; then
    "$BIN/python" - <<'PYEOF' 2>/dev/null || true
from ml_stack.fleet import updates

said = updates.state()
print(f"  running     {said['version'] or '?'}  {said['commit'] or '?'}")
PYEOF
  fi
  if [ -n "$TRACK" ]; then
    say "  updates     follows $TRACK, whenever nothing is running here"
  else
    say "  updates     releases, whenever nothing is running here"
  fi
  say ""
  say "  next        ml-stack-fleet status    -- who else is in the fleet"
}

do_uninstall() {
  step "removing ml-stack"
  VENV="$(venv_root)"
  [ -x "$VENV/bin/python" ] || VENV="${ML_STACK_PREFIX:-/opt/ml-stack}/venv"
  if [ -x "$VENV/bin/python" ]; then
    # `uninstall.plan` ticks everything ml-stack made for itself and leaves unticked what
    # the person made -- their models and their datasets. Only the ticked ones go.
    "$VENV/bin/python" - <<'PYEOF' || say "  (the uninstall plan did not run)"
from pathlib import Path

from ml_stack.fleet import uninstall

root = Path("~/.ml-stack/traind").expanduser()
items = uninstall.plan(root)
went = uninstall.remove(root, [i.key for i in items if i.default])
for name in went.get("removed", []):
    print("  removed", name)
for name in (i.name for i in items if not i.default):
    print("  kept   ", name)
PYEOF
  fi
  rm -rf "$VENV"
  say ""
  say "The model cache is left where it is, so coming back downloads nothing again."
  say "Remove it yourself with:  rm -rf ~/.cache/huggingface/hub"
  exit 0
}

# -- go -----------------------------------------------------------------------
if [ "$UNINSTALL" = yes ]; then do_uninstall; fi

case "$MODE" in
  app)      install_app ;;
  headless) install_headless ;;
  dev)      install_dev ;;
  system)   install_headless; install_system ;;
  *) die "unknown mode '$MODE' (app, headless, dev, system)" ;;
esac

if [ "$MODE" != app ]; then
  sizing
  llama_build
  fetch_models
  join_fleet
  check_over
  last_screen
fi
