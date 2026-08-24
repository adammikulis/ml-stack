#!/bin/sh
# Network installer for ml-stack. Downloads the current release for this machine.
#
#   curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh
#
# Everything it fetches comes from the GitHub release for the repository below.
set -eu

REPO="${ML_STACK_REPO:-adammikulis/ml-stack}"
API="https://api.github.com/repos/$REPO/releases/latest"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "this needs $1"; }
need curl
need unzip

case "$(uname -s)" in
  Darwin) OS=macos ;;
  Linux)  OS=linux ;;
  *)      die "unsupported system: $(uname -s). Download it by hand from https://github.com/$REPO/releases/latest" ;;
esac
case "$(uname -m)" in
  arm64|aarch64) ARCH=arm64 ;;
  x86_64|amd64)  ARCH=x86_64 ;;
  *)             die "unsupported processor: $(uname -m)" ;;
esac
if [ "$OS" = macos ] && [ "$ARCH" != arm64 ]; then
  die "ml-stack needs an Apple silicon Mac (M1 or later)."
fi
[ "$OS" = linux ] && ARCH=x86_64
KEY="ml-stack-$OS-$ARCH"

say "Looking for the newest ml-stack for $OS $ARCH..."
JSON=$(curl -fsSL -H 'Accept: application/vnd.github+json' "$API") \
  || die "could not reach GitHub"

TAG=$(printf '%s' "$JSON" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
URL=$(printf '%s' "$JSON" \
  | tr ',' '\n' | grep 'browser_download_url' | grep "$KEY" \
  | sed -n 's/.*"\(https[^"]*\)".*/\1/p' | head -1)
[ -n "${URL:-}" ] || die "release ${TAG:-latest} has no download for $KEY"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
say "Downloading $TAG..."
curl -fL# -o "$TMP/pkg.zip" "$URL" || die "download failed"
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
