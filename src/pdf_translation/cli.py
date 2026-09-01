from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import Profile, ProfileError
from .extractor import ExtractionError, extract_document
from .pipeline import (
    PipelineError,
    create_release,
    create_shard_plan,
    merge_shards,
    translate_all_shards,
    translate_shard,
    validate_merged,
)


def _profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to a version 1.0 translation profile",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-translate",
        description="Fail-closed local translation pipeline for structured PDFs",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="Extract PDF checkpoints")
    _profile_argument(extract)
    extract.add_argument("--pages", help="1-based selector: all, 1-5, 1,3,8-10")
    extract.add_argument("--force", action="store_true")

    plan = commands.add_parser("plan", help="Create a balanced shard plan")
    _profile_argument(plan)
    plan.add_argument("--shards", type=int, required=True)

    translate = commands.add_parser(
        "translate", help="Translate one shard or the complete shard plan"
    )
    _profile_argument(translate)
    selection = translate.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shard", type=int)
    selection.add_argument("--all-shards", action="store_true")
    translate.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel shard workers used with --all-shards",
    )
    translate.add_argument("--force", action="store_true")

    merge = commands.add_parser("merge", help="Merge current shard outputs")
    _profile_argument(merge)

    validate = commands.add_parser(
        "validate", help="Run global release-blocking QA"
    )
    _profile_argument(validate)

    release = commands.add_parser(
        "release", help="Create a release catalog after successful QA"
    )
    _profile_argument(release)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    profile = Profile.load(args.profile)
    if args.command == "extract":
        result = extract_document(
            profile, pages=args.pages, force=bool(args.force)
        )
    elif args.command == "plan":
        result = create_shard_plan(profile, args.shards)
    elif args.command == "translate":
        result = (
            translate_all_shards(
                profile, workers=args.workers, force=bool(args.force)
            )
            if args.all_shards
            else translate_shard(profile, args.shard, force=bool(args.force))
        )
    elif args.command == "merge":
        result = merge_shards(profile)
    elif args.command == "validate":
        result = validate_merged(profile)
        _print(result)
        return 0 if result["status"] == "PASS" else 2
    elif args.command == "release":
        result = create_release(profile)
    else:  # pragma: no cover - argparse enforces a command
        raise AssertionError(args.command)
    _print(result)
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except (ProfileError, ExtractionError, PipelineError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("interrupted; completed checkpoints remain reusable", file=sys.stderr)
        raise SystemExit(130)
