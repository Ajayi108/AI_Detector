# Command-line interface for local detector analysis.
from __future__ import annotations

import argparse
import json
import sys

from ai_detector.detectors import create_detectors, detector_catalog
from ai_detector.engine import DetectorEngine
from ai_detector.formatting import report_to_text
from ai_detector.text import TextExtractionError, load_text_from_path


# Parse CLI arguments and dispatch to the selected command.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-detector", description="Local multi-engine AI text detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-detectors", help="Show available detector adapters")
    list_parser.set_defaults(func=_list_detectors)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze text or a file")
    analyze_parser.add_argument("path", nargs="?", help="Text file, Markdown, DOCX, or PDF to analyze")
    analyze_parser.add_argument("--text", help="Raw text to analyze instead of a file")
    analyze_parser.add_argument(
        "--detectors",
        default="desklib,adal,heuristic",
        help="Comma-separated detector keys. Use list-detectors to see keys.",
    )
    analyze_parser.add_argument("--json", action="store_true", help="Print JSON output")
    analyze_parser.set_defaults(func=_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


# Print the available detector adapters.
def _list_detectors(_args: argparse.Namespace) -> int:
    for info in detector_catalog():
        default = "default" if info.default_enabled else "optional"
        advisory = ", advisory" if info.advisory_only else ""
        print(f"{info.key:12} {info.name} ({default}{advisory})")
        print(f"             {info.description}")
    return 0


# Analyze raw text or a local file and print the report.
def _analyze(args: argparse.Namespace) -> int:
    if not args.text and not args.path:
        print("Provide either --text or a path.", file=sys.stderr)
        return 2

    try:
        text = args.text if args.text is not None else load_text_from_path(args.path)
    except TextExtractionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    detector_keys = [key.strip() for key in args.detectors.split(",") if key.strip()]
    engine = DetectorEngine(create_detectors(detector_keys))
    report = engine.analyze(text)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report_to_text(report))
    return 0
