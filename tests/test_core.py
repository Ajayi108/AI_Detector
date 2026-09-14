# Core regression tests for text helpers and the detector engine.
import unittest

from ai_detector.detectors import HeuristicDetector
from ai_detector.engine import DetectorEngine
from ai_detector.models import TOO_SHORT
from ai_detector.text import chunk_text, clean_text, count_words


class CoreTests(unittest.TestCase):
    # Verify text cleanup produces predictable spacing.
    def test_clean_text_normalizes_whitespace(self):
        self.assertEqual(clean_text(" Hello\t\tworld \r\n\r\n\r\n again "), "Hello world\n\nagain")

    # Verify the word counter handles punctuation.
    def test_count_words_handles_basic_text(self):
        self.assertEqual(count_words("One, two, and three."), 4)

    # Verify overlapping chunks include shared boundary words.
    def test_chunk_text_uses_overlap(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(text, target_words=40, overlap_words=10, max_chunks=10)
        self.assertEqual(len(chunks), 3)
        self.assertIn("word30", chunks[0])
        self.assertIn("word30", chunks[1])

    # Verify very short text returns a safe no-verdict result.
    def test_engine_returns_too_short_for_short_text(self):
        report = DetectorEngine([HeuristicDetector()]).analyze("Too short.")
        self.assertEqual(report.verdict, TOO_SHORT)
        self.assertIsNone(report.consensus_ai_probability)

    # Verify the engine can run with the no-dependency advisory detector.
    def test_engine_runs_with_heuristic_detector(self):
        text = (
            "Additionally, this overview explores the landscape of modern tools. "
            "Furthermore, it is important to note that consistent phrasing can appear. "
            "Overall, the document provides a structured explanation with repeated patterns. "
        ) * 12
        report = DetectorEngine([HeuristicDetector()]).analyze(text)
        self.assertGreater(report.word_count, 60)
        self.assertEqual(report.detector_results[0].status, "ok")
        self.assertIsNotNone(report.consensus_ai_probability)


# Run tests directly when this file is executed.
if __name__ == "__main__":
    unittest.main()
