# Consensus engine that runs detectors and combines their results.
from __future__ import annotations

from ai_detector.detectors import BaseDetector, create_detectors
from ai_detector.models import (
    AnalysisReport,
    DetectorResult,
    LIKELY_AI,
    LIKELY_HUMAN,
    TOO_SHORT,
    UNCLEAR,
    UNAVAILABLE,
)
from ai_detector.text import chunk_text, clean_text, count_words


class DetectorEngine:
    # Store detector choices and chunking settings for one analysis run.
    def __init__(
        self,
        detectors: list[BaseDetector] | None = None,
        target_chunk_words: int = 380,
        overlap_words: int = 60,
        max_chunks: int = 24,
    ) -> None:
        self.detectors = detectors if detectors is not None else create_detectors()
        self.target_chunk_words = target_chunk_words
        self.overlap_words = overlap_words
        self.max_chunks = max_chunks

    # Analyze text with every selected detector and return a report.
    def analyze(self, text: str) -> AnalysisReport:
        # Clean once, chunk once, then every detector sees the same input.
        prepared = clean_text(text)
        word_count = count_words(prepared)
        chunks = chunk_text(
            prepared,
            target_words=self.target_chunk_words,
            overlap_words=self.overlap_words,
            max_chunks=self.max_chunks,
        )

        if not prepared or word_count < 40:
            results = [
                DetectorResult(
                    key=detector.key,
                    name=detector.name,
                    status="skipped",
                    label=TOO_SHORT,
                    detail=f"Needs more text for reliable detection; got {word_count} words.",
                    weight=detector.weight,
                )
                for detector in self.detectors
            ]
            return AnalysisReport(
                text=prepared,
                word_count=word_count,
                chunk_count=0,
                verdict=TOO_SHORT,
                verdict_detail="Text is too short for a responsible AI-detection result.",
                agreement="none",
                consensus_ai_probability=None,
                detector_results=results,
                notes=_default_notes(word_count),
            )

        # Detector failures are returned as rows by each adapter, so one bad model does not stop the report.
        results = [detector.analyze_chunks(chunks) for detector in self.detectors]
        verdict, detail, agreement, probability = _consensus(results)
        return AnalysisReport(
            text=prepared,
            word_count=word_count,
            chunk_count=len(chunks),
            verdict=verdict,
            verdict_detail=detail,
            agreement=agreement,
            consensus_ai_probability=probability,
            detector_results=results,
            notes=_default_notes(word_count),
        )


# Combine detector rows into one cautious final verdict.
def _consensus(results: list[DetectorResult]) -> tuple[str, str, str, float | None]:
    # Advisory-only signals are shown, but they are not allowed to create a strong final verdict.
    usable = [result for result in results if result.usable]
    primary = [result for result in usable if not result.metadata.get("advisory_only")]
    considered = primary or usable

    if not considered:
        return (
            UNAVAILABLE,
            "No detector produced a usable result. Install model dependencies or enable a local model.",
            "none",
            None,
        )

    if not primary:
        total_weight = sum(max(result.weight, 0.01) for result in considered)
        weighted_probability = sum((result.ai_probability or 0.0) * max(result.weight, 0.01) for result in considered)
        weighted_probability = weighted_probability / total_weight
        return (
            UNCLEAR,
            "Only advisory fallback signals ran. Install at least one ML detector for a stronger result.",
            "advisory-only",
            weighted_probability,
        )

    total_weight = sum(max(result.weight, 0.01) for result in considered)
    weighted_probability = sum((result.ai_probability or 0.0) * max(result.weight, 0.01) for result in considered)
    weighted_probability = weighted_probability / total_weight

    ai_votes = sum(1 for result in considered if result.label == LIKELY_AI)
    human_votes = sum(1 for result in considered if result.label == LIKELY_HUMAN)
    unclear_votes = sum(1 for result in considered if result.label == UNCLEAR)

    if ai_votes and human_votes:
        return (
            UNCLEAR,
            "Detectors disagree. Treat this as inconclusive and review the text manually.",
            "mixed",
            weighted_probability,
        )

    if ai_votes >= 2 or (ai_votes == 1 and weighted_probability >= 0.72 and human_votes == 0):
        return (
            LIKELY_AI,
            "Available detectors lean toward AI-generated text.",
            "aligned" if ai_votes > 1 else "single-detector",
            weighted_probability,
        )

    if human_votes >= 2 or (human_votes == 1 and weighted_probability <= 0.28 and ai_votes == 0):
        return (
            LIKELY_HUMAN,
            "Available detectors lean toward human-written text.",
            "aligned" if human_votes > 1 else "single-detector",
            weighted_probability,
        )

    if unclear_votes or 0.38 < weighted_probability < 0.62:
        return (
            UNCLEAR,
            "The signal is weak or mixed; there is not enough evidence for a strong label.",
            "weak",
            weighted_probability,
        )

    if weighted_probability >= 0.62:
        return (LIKELY_AI, "Available detectors weakly lean toward AI-generated text.", "weak", weighted_probability)
    return (LIKELY_HUMAN, "Available detectors weakly lean toward human-written text.", "weak", weighted_probability)


# Add standard safety notes to every report.
def _default_notes(word_count: int) -> list[str]:
    notes = [
        "AI detection is probabilistic and should not be treated as proof of authorship.",
        "Use the per-detector results and disagreement as evidence, not as an accusation.",
    ]
    if word_count < 150:
        notes.append("Short samples are less reliable; 300+ words is strongly preferred.")
    return notes
