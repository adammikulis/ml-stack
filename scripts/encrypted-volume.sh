#!/bin/sh
# A directory inside an encrypted disk image, unlocked only while something needs it. macOS.
#
#   scripts/encrypted-volume.sh NAME MOUNT_DIR setup [--source DIR] [--size 2g]
#   scripts/encrypted-volume.sh NAME MOUNT_DIR mount
#   scripts/encrypted-volume.sh NAME MOUNT_DIR unmount
#
# The image is MOUNT_DIR.sparsebundle, APFS under AES-256, its passphrase made once and kept
# in the login keychain under the service NAME-data. `setup` makes the image and mounts it;
# with --source DIR it moves that directory into the volume and leaves DIR as a symlink to
# MOUNT_DIR, so whatever read DIR reads the volume. `mount` attaches the image unless it is
# attached already; `unmount` detaches it.
#
# FileVault covers a machine that is off. This covers one that is on: the directory is
# unreadable to anything running as you while the volume is not attached.
set -eu

usage() {
  echo "usage: $0 NAME MOUNT_DIR {setup [--source DIR] [--size SIZE]|mount|unmount}" >&2
  exit 2
}

[ $# -ge 3 ] || usage
NAME=$1
MOUNT=$2
ACTION=$3
shift 3
IMAGE="$MOUNT.sparsebundle"
SERVICE="$NAME-data"
SOURCE=""
SIZE=2g
while [ $# -gt 0 ]; do
  case $1 in
    --source) [ $# -ge 2 ] || usage; SOURCE=$2; shift 2 ;;
    --size) [ $# -ge 2 ] || usage; SIZE=$2; shift 2 ;;
    *) usage ;;
  esac
done

password() { security find-generic-password -a "$USER" -s "$SERVICE" -w 2>/dev/null; }
# the directory exists whether or not the image is attached; ask the kernel, not the filesystem
attached() { mount | grep -q " on $MOUNT "; }

case $ACTION in
setup)
  if [ -e "$IMAGE" ]; then echo "already set up: $IMAGE"; exit 0; fi
  if ! password >/dev/null; then
    PW=$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 40)
    security add-generic-password -a "$USER" -s "$SERVICE" -w "$PW"
    echo "made a passphrase and stored it in your login keychain as '$SERVICE'"
  fi
  mkdir -p "$(dirname "$MOUNT")"
  PW=$(password)
  printf '%s' "$PW" | hdiutil create -size "$SIZE" -type SPARSEBUNDLE -fs APFS \
      -encryption AES-256 -stdinpass -volname "$SERVICE" "$IMAGE" >/dev/null
  "$0" "$NAME" "$MOUNT" mount
  if [ -n "$SOURCE" ]; then
    if [ -d "$SOURCE" ] && [ ! -L "$SOURCE" ]; then
      rsync -a "$SOURCE/" "$MOUNT/"
      rm -rf "$SOURCE"
    fi
    ln -sfn "$MOUNT" "$SOURCE"
    echo "$SOURCE points at the volume at $MOUNT"
  fi
  echo "the volume is at $MOUNT; the image is $IMAGE"
  ;;
mount)
  attached && exit 0
  mkdir -p "$MOUNT"
  PW=$(password)
  printf '%s' "$PW" | hdiutil attach "$IMAGE" -stdinpass -mountpoint "$MOUNT" -nobrowse -quiet
  ;;
unmount)
  hdiutil detach "$MOUNT" -quiet 2>/dev/null || true
  ;;
*)
  usage ;;
esac
