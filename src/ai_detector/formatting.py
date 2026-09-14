# Formatting helpers for CLI and UI detector reports.
from __future__ import annotations

from ai_detector.models import AnalysisReport


LABEL_TEXT = {
    "likely_ai": "Likely AI",
    "likely_human": "Likely human",
    "unclear": "Inconclusive",
    "too_short": "Too short",
    "unavailable": "Unavailable",
    "error": "Error",
}


# Convert internal labels into user-facing wording.
def label_text(label: str) -> str:
    return LABEL_TEXT.get(label, label.replace("_", " ").title())


# Format a probability as a percentage, or a dash when absent.
def percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


# Render a full report as readable plain text for the CLI.
def report_to_text(report: AnalysisReport) -> str:
    lines = [
        f"Verdict: {label_text(report.verdict)}",
        f"Detail: {report.verdict_detail}",
        f"Words: {report.word_count}",
        f"Chunks: {report.chunk_count}",
        f"Consensus AI likelihood: {percent(report.consensus_ai_probability)}",
        "",
        "Detector results:",
    ]
    for result in report.detector_results:
        lines.append(
            f"- {result.name}: {label_text(result.label)} "
            f"({percent(result.ai_probability)}, {result.status}) - {result.detail}"
        )
    if report.notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines)
