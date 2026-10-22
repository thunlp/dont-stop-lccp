"""Offline LongReason dataset and dual-metric evaluator for OpenCompass."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


STRICT_ANSWER_RE = re.compile(r"(?i)\bthe answer is\s*([A-E])\b")
ANSWER_COLON_RE = re.compile(r"(?i)\banswer\s*:\s*([A-E])\b")
BARE_ANSWER_RE = re.compile(r"(?i)\b([A-E])\b")
SAFE_EXPLICIT_PATTERNS = (
    ("the_answer_is", STRICT_ANSWER_RE),
    (
        "the_correct_answer_is",
        re.compile(r"\bthe\s+correct\s+answer\s+is\s*([A-E])\b", re.I | re.ASCII),
    ),
    (
        "answer_colon",
        re.compile(r"\banswer\s*:\s*([A-E])\b", re.I | re.ASCII),
    ),
)
SAFE_BARE_ANSWER_RE = re.compile(r"\A\s*([A-E])\s*\Z", re.I | re.ASCII)


def extract_strict_answer(text: str) -> str:
    """Apply the paper's exact answer extraction rule."""
    match = STRICT_ANSWER_RE.search(str(text))
    return match.group(1).upper() if match else ""


def extract_lenient_answer(text: str) -> str:
    """Accept strict, ``Answer: X``, or the final bare A-E token."""
    value = str(text)
    strict = extract_strict_answer(value)
    if strict:
        return strict
    match = ANSWER_COLON_RE.search(value)
    if match:
        return match.group(1).upper()
    matches = BARE_ANSWER_RE.findall(value)
    return matches[-1].upper() if matches else ""


def extract_safe_explicit_answer(text: str) -> tuple[str, str]:
    """Extract only explicit, non-conflicting multiple-choice answer forms."""
    hits = [
        (name, match.group(1).upper())
        for name, pattern in SAFE_EXPLICIT_PATTERNS
        for match in pattern.finditer(str(text))
    ]
    choices = {choice for _, choice in hits}
    if len(choices) == 1:
        formats = {name for name, _ in hits}
        format_name = next(iter(formats)) if len(formats) == 1 else "multiple_explicit"
        return choices.pop(), format_name
    if len(choices) > 1:
        return "", "ambiguous_explicit"

    match = SAFE_BARE_ANSWER_RE.fullmatch(str(text))
    if match:
        return match.group(1).upper(), "bare_choice"
    return "", "unextractable"


def _parquet_path(path: str, split: str) -> Path:
    source = Path(path).expanduser()
    if source.is_file():
        return source
    candidate = source / f"{split}.parquet"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"LongReason split {split!r} is missing: {candidate}. "
            "Run prepare_data.py first."
        )
    return candidate


@LOAD_DATASET.register_module()
class LongReasonLocalDataset(BaseDataset):
    """Load one pinned LongReason split from local parquet only."""

    @staticmethod
    def load(
        path: str,
        split: str,
        num_samples: int | None = None,
        **kwargs: Any,
    ) -> Dataset:
        parquet = _parquet_path(path, split)
        dataset = load_dataset(
            "parquet",
            data_files=str(parquet),
            split="train",
        )
        required = {"prompt", "answer"}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(f"{parquet} is missing columns: {sorted(missing)}")
        if num_samples is not None:
            if num_samples < 1:
                raise ValueError("num_samples must be positive")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset


def _reference_answer(reference: Any) -> str:
    if isinstance(reference, dict):
        reference = reference.get("answer", "")
    return str(reference).strip().upper()


@ICL_EVALUATORS.register_module()
class LongReasonEvaluator(BaseEvaluator):
    """Report strict paper accuracy and a diagnostic lenient accuracy."""

    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("predictions and references have different lengths")

        strict_correct = 0
        lenient_correct = 0
        details = []
        for prediction, reference in zip(predictions, references):
            text = str(prediction)
            gold = _reference_answer(reference)
            strict = extract_strict_answer(text)
            lenient = extract_lenient_answer(text)
            strict_ok = strict == gold
            lenient_ok = lenient == gold
            strict_correct += int(strict_ok)
            lenient_correct += int(lenient_ok)
            details.append(
                {
                    "prediction_strict": strict,
                    "prediction_lenient": lenient,
                    "reference": gold,
                    "strict_correct": strict_ok,
                    "lenient_correct": lenient_ok,
                }
            )

        total = len(predictions)
        divisor = total or 1
        return {
            "accuracy_strict": round(100.0 * strict_correct / divisor, 2),
            "accuracy_lenient": round(100.0 * lenient_correct / divisor, 2),
            "strict_correct": strict_correct,
            "lenient_correct": lenient_correct,
            "total": total,
            "details": details,
        }


__all__ = [
    "LongReasonEvaluator",
    "LongReasonLocalDataset",
    "extract_lenient_answer",
    "extract_safe_explicit_answer",
    "extract_strict_answer",
]
