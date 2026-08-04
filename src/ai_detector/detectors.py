# Detector adapters and lightweight fallback signals for local analysis.
from __future__ import annotations

import math
import os
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from statistics import fmean

from ai_detector.models import DetectorResult, LIKELY_AI, LIKELY_HUMAN, TOO_SHORT, UNCLEAR
from ai_detector.text import WORD_RE, count_words


@dataclass(frozen=True, slots=True)
class DetectorInfo:
    key: str
    name: str
    description: str
    default_enabled: bool
    advisory_only: bool


class BaseDetector(ABC):
    key = "base"
    name = "Base detector"
    description = ""
    min_words = 50
    weight = 1.0
    default_enabled = True
    advisory_only = False

    # Run one detector on one text chunk with shared error handling.
    def analyze(self, text: str) -> DetectorResult:
        # Every detector gets the same short-text gate so the UI stays honest.
        word_count = count_words(text)
        if word_count < self.min_words:
            return DetectorResult(self.key, self.name, "skipped", TOO_SHORT, detail=f"Needs at least {self.min_words} words; got {word_count}.", weight=self.weight)

        started = time.perf_counter()
        try:
            result = self._analyze_text(text)
        except Exception as exc:  # Optional ML stacks should fail as a detector row, not crash the app.
            result = DetectorResult(self.key, self.name, "error", "error", detail=str(exc), weight=self.weight)
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    # Run one detector across all chunks and average usable chunk scores.
    def analyze_chunks(self, chunks: list[str]) -> DetectorResult:
        # Long documents are scored by chunks, then averaged into one detector result.
        results = [self.analyze(chunk) for chunk in chunks]
        usable = [result for result in results if result.usable]
        if not usable:
            return self._first_non_ok(results)

        ai_probability = float(fmean(result.ai_probability for result in usable if result.ai_probability is not None))
        confidence = abs(ai_probability - 0.5) * 2
        detail = usable[0].detail if len(usable) == 1 else f"Analyzed {len(usable)} chunks and averaged the detector scores."
        metadata = dict(usable[0].metadata) if len(usable) == 1 else {}
        metadata.update({"chunks_analyzed": len(usable), "chunks_total": len(chunks), "advisory_only": self.advisory_only})
        return DetectorResult(self.key, self.name, "ok", label_from_probability(ai_probability), ai_probability, confidence, detail, sum(result.elapsed_ms for result in results), self.weight, metadata)

    # Let each concrete detector implement its own scoring logic.
    @abstractmethod
    def _analyze_text(self, text: str) -> DetectorResult:
        raise NotImplementedError

    # Return the first skipped/error row when no chunk produced a score.
    def _first_non_ok(self, results: list[DetectorResult]) -> DetectorResult:
        if not results:
            return DetectorResult(self.key, self.name, "skipped", TOO_SHORT, detail="No text chunks to analyze.", weight=self.weight)
        result = results[0]
        result.elapsed_ms = sum(item.elapsed_ms for item in results)
        return result


class DesklibDetector(BaseDetector):
    key = "desklib"
    name = "Desklib DeBERTa Detector"
    description = "MIT-licensed DeBERTa-v3-large detector trained around RAID."
    min_words = 80
    weight = 1.0
    model_id = "desklib/ai-text-detector-v1.01"
    max_length = 768

    # Store the model id and lazy-loaded model objects.
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("AI_DETECTOR_DESKLIB_MODEL", self.model_id)
        self._tokenizer = None
        self._model = None
        self._device = None

    # Score text with the Desklib binary classifier.
    def _analyze_text(self, text: str) -> DetectorResult:
        self._ensure_loaded()
        import torch

        encoded = self._tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        self._model.eval()
        with torch.no_grad():
            probability = torch.sigmoid(self._model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])["logits"]).item()
        return DetectorResult(self.key, self.name, "ok", label_from_probability(probability), float(probability), abs(float(probability) - 0.5) * 2, "Transformer classifier probability for AI-generated text.", weight=self.weight, metadata={"model_id": self.model_id})

    # Load Desklib only when the user actually runs it.
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
        except ImportError as exc:
            raise RuntimeError("Desklib requires ML dependencies. Install with: pip install -r requirements.txt") from exc

        class DesklibAIDetectionModel(PreTrainedModel):
            config_class = AutoConfig
            all_tied_weights_keys = {}

            # Create the DeBERTa backbone and one-logit classifier head.
            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = nn.Linear(config.hidden_size, 1)
                self.init_weights()

            # Pool token embeddings and return an AI logit.
            def forward(self, input_ids, attention_mask=None, labels=None):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs[0]
                input_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                pooled = (last_hidden_state * input_mask).sum(dim=1) / torch.clamp(input_mask.sum(dim=1), min=1e-9)
                return {"logits": self.classifier(pooled)}

        local_only = os.getenv("AI_DETECTOR_OFFLINE", "0") == "1"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=local_only)
        self._model = DesklibAIDetectionModel.from_pretrained(self.model_id, local_files_only=local_only)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)


class AdalDetector(BaseDetector):
    key = "adal"
    name = "ADAL RoBERTa Detector"
    description = "Adversarial-learning detector based on the RADAR line of research."
    min_words = 80
    weight = 0.9
    model_id = "Shushant/ADAL_AI_Detector"
    max_length = 512

    # Store the model id and lazy-loaded model objects.
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("AI_DETECTOR_ADAL_MODEL", self.model_id)
        self._tokenizer = None
        self._model = None
        self._device = None

    # Score text with the ADAL RoBERTa classifier.
    def _analyze_text(self, text: str) -> DetectorResult:
        self._ensure_loaded()
        import torch

        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        self._model.eval()
        with torch.no_grad():
            ai_probability = float(torch.softmax(self._model(**encoded).logits, dim=-1)[0][0].item())
        return DetectorResult(self.key, self.name, "ok", label_from_probability(ai_probability), ai_probability, abs(ai_probability - 0.5) * 2, "RoBERTa classifier probability. Label 0 is AI, label 1 is human.", weight=self.weight, metadata={"model_id": self.model_id})

    # Load ADAL only when the user actually runs it.
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("ADAL requires ML dependencies. Install with: pip install -r requirements.txt") from exc

        local_only = os.getenv("AI_DETECTOR_OFFLINE", "0") == "1"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=local_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id, local_files_only=local_only)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)


class BinocularsDetector(BaseDetector):
    key = "binoculars"
    name = "Binoculars-Style Deep Scan"
    description = "Slower language-model comparison scan; uses the official package when installed."
    min_words = 120
    weight = 0.75
    default_enabled = False
    scorer_model_id = "distilgpt2"
    observer_model_id = "gpt2"
    max_length = 512

    # Keep all heavy model objects lazy.
    def __init__(self) -> None:
        self._bino = None
        self._mode = None
        self._tokenizer = None
        self._scorer = None
        self._observer = None
        self._device = None
        self.scorer_model_id = os.getenv("AI_DETECTOR_BINOCULARS_SCORER", self.scorer_model_id)
        self.observer_model_id = os.getenv("AI_DETECTOR_BINOCULARS_OBSERVER", self.observer_model_id)

    # Convert the Binoculars prediction into a comparable detector row.
    def _analyze_text(self, text: str) -> DetectorResult:
        self._ensure_loaded()
        if self._mode == "internal":
            return self._analyze_with_internal_models(text)

        raw_score = float(self._bino.compute_score(text))
        prediction = str(self._bino.predict(text))
        lowered = prediction.lower()
        if "ai" in lowered or "machine" in lowered:
            label, ai_probability = LIKELY_AI, 0.75
        elif "human" in lowered:
            label, ai_probability = LIKELY_HUMAN, 0.25
        else:
            label, ai_probability = UNCLEAR, 0.5
        return DetectorResult(self.key, self.name, "ok", label, ai_probability, abs(ai_probability - 0.5) * 2, f"Binoculars prediction: {prediction}. Raw score is not a probability.", weight=self.weight, metadata={"raw_score": raw_score, "raw_prediction": prediction})

    # Load official Binoculars if available, otherwise use a compatible internal scan.
    def _ensure_loaded(self) -> None:
        if self._mode is not None:
            return
        try:
            from binoculars import Binoculars
        except ImportError:
            self._load_internal_models()
            return
        self._bino = Binoculars()
        self._mode = "official"

    # Load two small causal language models for the internal comparison scan.
    def _load_internal_models(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Binoculars-style scan requires torch and transformers. Install with: pip install -r requirements.txt") from exc

        local_only = os.getenv("AI_DETECTOR_OFFLINE", "0") == "1"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.scorer_model_id, local_files_only=local_only)
        self._scorer = AutoModelForCausalLM.from_pretrained(self.scorer_model_id, local_files_only=local_only).to(self._device)
        self._observer = AutoModelForCausalLM.from_pretrained(self.observer_model_id, local_files_only=local_only).to(self._device)
        self._scorer.eval()
        self._observer.eval()
        self._mode = "internal"

    # Score text with a lightweight approximation of the Binoculars ratio.
    def _analyze_with_internal_models(self, text: str) -> DetectorResult:
        import torch
        import torch.nn.functional as F

        encoded = self._tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        attention_mask = attention_mask.to(self._device) if attention_mask is not None else None
        with torch.no_grad():
            scorer_logits = self._scorer(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
            observer_logits = self._observer(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_nll = F.cross_entropy(scorer_logits.reshape(-1, scorer_logits.size(-1)), labels.reshape(-1), reduction="mean")
        observer_probs = F.softmax(observer_logits, dim=-1)
        scorer_log_probs = F.log_softmax(scorer_logits, dim=-1)
        cross_entropy = -(observer_probs * scorer_log_probs).sum(dim=-1).mean()
        raw_score = float(torch.exp(token_nll - cross_entropy).item())
        ai_probability = 1 / (1 + math.exp((raw_score - 0.9) * 12))
        detail = f"Internal Binoculars-style ratio using {self.scorer_model_id}/{self.observer_model_id}; raw score {raw_score:.3f}."
        return DetectorResult(self.key, self.name, "ok", label_from_probability(ai_probability), ai_probability, abs(ai_probability - 0.5) * 2, detail, weight=self.weight, metadata={"raw_score": raw_score, "mode": "internal", "scorer": self.scorer_model_id, "observer": self.observer_model_id})


class HeuristicDetector(BaseDetector):
    key = "heuristic"
    name = "Local Style Signals"
    description = "Tiny no-dependency advisory baseline. Useful for testing, not a proof."
    min_words = 60
    weight = 0.2
    advisory_only = True

    # Score text with simple local style features only.
    def _analyze_text(self, text: str) -> DetectorResult:
        # This detector is intentionally weak; it keeps the app testable before ML models are installed.
        words = [word.lower() for word in WORD_RE.findall(text)]
        unique_ratio = len(set(words)) / max(1, len(words))
        burstiness = _coefficient_of_variation(_sentence_lengths(text))
        marker_hits = sum(1 for marker in AI_STYLE_MARKERS if marker in text.lower())
        repeat_score = _repeated_ngram_score(words)
        score = 0.5 + _scale(0.48 - unique_ratio, 0.0, 0.24) * 0.18 + _scale(0.42 - burstiness, 0.0, 0.42) * 0.18 + min(marker_hits / 5, 1) * 0.16 + min(repeat_score, 1) * 0.14
        score = max(0.18, min(0.82, score))
        features = {"unique_word_ratio": round(unique_ratio, 3), "sentence_length_variation": round(burstiness, 3), "style_marker_hits": marker_hits, "repeated_ngram_score": round(repeat_score, 3)}
        return DetectorResult(self.key, self.name, "ok", label_from_probability(score), score, abs(score - 0.5) * 2, "Advisory style signal only; install ML detectors for real analysis.", weight=self.weight, metadata={"features": features, "advisory_only": True})


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
AI_STYLE_MARKERS = {"additionally", "furthermore", "moreover", "in conclusion", "overall", "it is important to note", "in today's", "delve", "tapestry", "realm", "landscape"}
DETECTOR_TYPES: dict[str, type[BaseDetector]] = {"desklib": DesklibDetector, "adal": AdalDetector, "binoculars": BinocularsDetector, "heuristic": HeuristicDetector}


# Return metadata for the UI and CLI detector list.
def detector_catalog() -> list[DetectorInfo]:
    return [DetectorInfo(key, detector.name, detector.description, detector.default_enabled, detector.advisory_only) for key, detector in DETECTOR_TYPES.items()]


# Create detector instances from selected keys.
def create_detectors(keys: list[str] | None = None) -> list[BaseDetector]:
    # Defaults are the two real ML detectors plus the advisory fallback.
    selected = keys or [key for key, detector in DETECTOR_TYPES.items() if detector.default_enabled]
    detectors: list[BaseDetector] = []
    for key in selected:
        normalized = key.strip().lower()
        if normalized:
            if normalized not in DETECTOR_TYPES:
                raise ValueError(f"Unknown detector: {key}")
            detectors.append(DETECTOR_TYPES[normalized]())
    return detectors


# Convert an AI probability into the app's three main labels.
def label_from_probability(ai_probability: float) -> str:
    if ai_probability >= 0.68:
        return LIKELY_AI
    if ai_probability <= 0.32:
        return LIKELY_HUMAN
    return UNCLEAR


# Count words in each sentence for burstiness scoring.
def _sentence_lengths(text: str) -> list[int]:
    lengths = [len(WORD_RE.findall(sentence)) for sentence in SENTENCE_RE.findall(text)]
    return [length for length in lengths if length] or [0]


# Measure how uneven sentence lengths are.
def _coefficient_of_variation(values: list[int]) -> float:
    usable = [value for value in values if value > 0]
    if len(usable) < 2:
        return 0.0
    mean = sum(usable) / len(usable)
    variance = sum((value - mean) ** 2 for value in usable) / len(usable)
    return math.sqrt(variance) / max(mean, 1)


# Estimate repeated phrasing with repeated n-grams.
def _repeated_ngram_score(words: list[str], n: int = 3) -> float:
    if len(words) < n * 4:
        return 0.0
    counts = Counter(zip(*(words[index:] for index in range(n))))
    return sum(count - 1 for count in counts.values() if count > 1) / max(1, len(words) / 80)


# Map a bounded feature value into a zero-to-one range.
def _scale(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)
