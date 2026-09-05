"""``ml-stack-speech``: what this machine can hear and say, and doing either.

    ml-stack-speech providers
    ml-stack-speech transcribe clip.wav --language en
    ml-stack-speech say "the model is up" --out said.wav
    ml-stack-speech regions clip.wav

Every subcommand takes ``--provider NAME`` to pick one engine rather than the first that
works, and ``--json`` to print what a program would read.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
from pathlib import Path

from ml_stack.speech.protocols import ProviderError
from ml_stack.speech.service import providers, regions, say, transcribe

__all__ = ["main"]


def _audio(path: str) -> Path:
    """The named audio file, or an error naming it rather than one from inside an engine."""
    found = Path(path).expanduser()
    if not found.is_file():
        raise FileNotFoundError(errno.ENOENT, "no such audio file", str(found))
    return found


def _providers(args: argparse.Namespace) -> int:
    found = providers()
    if args.json:
        print(json.dumps(found, ensure_ascii=False))
        return 0
    for kind, table in found.items():
        chosen = table["auto"] or "nothing available"
        print(f"{kind}  (auto: {chosen})")
        for one in table["providers"]:
            mark = "ok  " if one["available"] else "  ! "
            detail = one["model"] or "" if one["available"] else one["detail"]
            print(f"  {mark}{one['name']}: {detail}")
    return 0


def _transcribe(args: argparse.Namespace) -> int:
    got = transcribe(_audio(args.file), provider=args.provider,
                     language=args.language)
    if args.json:
        print(json.dumps({
            "text": got.text, "language": got.language, "duration_s": got.duration_s,
            "model": got.model,
            "segments": [{"text": s.text, "start_s": s.start_s, "end_s": s.end_s}
                         for s in got.segments],
        }, ensure_ascii=False))
    else:
        print(got.text)
    return 0


def _say(args: argparse.Namespace) -> int:
    spoken = say(args.text, provider=args.provider, voice=args.voice)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(spoken.to_wav())
    print(f"{out}  {spoken.duration_s:.1f}s  {spoken.sample_rate} Hz  "
          f"voice {spoken.voice or 'default'}")
    return 0


def _regions(args: argparse.Namespace) -> int:
    found = regions(_audio(args.file), provider=args.provider)
    if args.json:
        print(json.dumps({
            "speech": found.speech, "confidence": found.confidence,
            "regions": [{"start_s": r.start_s, "end_s": r.end_s} for r in found.regions],
        }, ensure_ascii=False))
        return 0
    if not found.regions:
        print("no speech")
        return 0
    for one in found.regions:
        print(f"{one.start_s:7.2f} -> {one.end_s:7.2f}  ({one.duration_s:.2f}s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """``ml-stack-speech`` -- transcribe a file, say a line, find the speech in a recording."""
    parser = argparse.ArgumentParser(
        prog="ml-stack-speech",
        description="Speech recognition, synthesis and voice activity on this machine, "
                    "whichever engine is installed.")
    subs = parser.add_subparsers(dest="command", required=True)

    listed = subs.add_parser("providers", help="every engine, and whether it could run here")
    listed.add_argument("--json", action="store_true", help="print it as JSON")
    listed.set_defaults(run=_providers)

    heard = subs.add_parser("transcribe", help="an audio file as text")
    heard.add_argument("file", help="anything ffmpeg reads: wav, mp3, m4a, a video")
    heard.add_argument("--provider", default="", help="one engine rather than the first that works")
    heard.add_argument("--language", default="", help="the spoken language, e.g. en; guessed if left out")
    heard.add_argument("--json", action="store_true", help="the segments with their times, as JSON")
    heard.set_defaults(run=_transcribe)

    spoke = subs.add_parser("say", help="text spoken into a WAV file")
    spoke.add_argument("text", help="what to say")
    spoke.add_argument("--out", required=True, help="the WAV file to write")
    spoke.add_argument("--provider", default="", help="one engine rather than the first that works")
    spoke.add_argument("--voice", default="", help="the voice, as that engine names it")
    spoke.set_defaults(run=_say)

    where = subs.add_parser("regions", help="where in a recording somebody is speaking")
    where.add_argument("file", help="anything ffmpeg reads")
    where.add_argument("--provider", default="", help="one engine rather than the first that works")
    where.add_argument("--json", action="store_true", help="print it as JSON")
    where.set_defaults(run=_regions)

    args = parser.parse_args(argv)
    for empty_is_unset in ("provider", "language", "voice"):
        if getattr(args, empty_is_unset, None) == "":
            setattr(args, empty_is_unset, None)
    try:
        return int(args.run(args))
    except ProviderError as why:
        print(f"ml-stack-speech: {why}", file=sys.stderr)
        return 1
    except OSError as why:
        print(f"ml-stack-speech: {why}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
