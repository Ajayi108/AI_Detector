# Shared result models used by the engine, CLI, and desktop UI.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Label = str

LIKELY_AI: Label = "likely_ai"
LIKELY_HUMAN: Label = "likely_human"
UNCLEAR: Label = "unclear"
TOO_SHORT: Label = "too_short"
UNAVAILABLE: Label = "unavailable"
ERROR: Label = "error"


@dataclass(slots=True)
class DetectorResult:
    key: str
    name: str
    status: str
    label: Label
    ai_probability: float | None = None
    confidence: float | None = None
    detail: str = ""
    elapsed_ms: int = 0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tell the consensus engine whether this detector produced a usable score.
    @property
    def usable(self) -> bool:
        return self.status == "ok" and self.ai_probability is not None


@dataclass(slots=True)
class AnalysisReport:
    text: str
    word_count: int
    chunk_count: int
    verdict: Label
    verdict_detail: str
    agreement: str
    consensus_ai_probability: float | None
    detector_results: list[DetectorResult]
    notes: list[str] = field(default_factory=list)

    # Convert the report into plain JSON-safe data for exports and CLI output.
    def as_dict(self) -> dict[str, Any]:
        return {
            "word_count": self.word_count,
            "chunk_count": self.chunk_count,
            "verdict": self.verdict,
            "verdict_detail": self.verdict_detail,
            "agreement": self.agreement,
            "consensus_ai_probability": self.consensus_ai_probability,
            "detectors": [
                {
                    "key": result.key,
                    "name": result.name,
                    "status": result.status,
                    "label": result.label,
                    "ai_probability": result.ai_probability,
                    "confidence": result.confidence,
                    "detail": result.detail,
                    "elapsed_ms": result.elapsed_ms,
                    "weight": result.weight,
                    "metadata": result.metadata,
                }
                for result in self.detector_results
            ],
            "notes": self.notes,
        }
