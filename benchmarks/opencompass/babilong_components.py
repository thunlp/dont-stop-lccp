"""Local BABILong dataset, prompt formatter, and dual-metric evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from opencompass.datasets.babilong.prompts import (
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    get_formatted_input,
)
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


TASKS = tuple(f"qa{index}" for index in range(1, 11))
SPLITS = ("8k", "16k", "32k")

# booydar/babilong commit 7a6efee29f5cac03c3c410e6799c80fd2ffe3610,
# babilong/metrics.py (blob 4c112c4cf1c6e8dcec2018a2c33807f024e69b48).
TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa4": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa5": ["Bill", "Fred", "Jeff", "Mary", "apple", "football", "milk"],
    "qa6": ["no", "yes"],
    "qa7": ["none", "one", "three", "two"],
    "qa8": ["apple", "football", "milk", "nothing"],
    "qa9": ["no", "yes"],
    "qa10": ["maybe", "no", "yes"],
}


def format_prompt(sample: dict[str, Any], task: str) -> str:
    """Render the official instruction, examples, post-prompt, and sample."""
    prompt = DEFAULT_PROMPTS[task]
    return get_formatted_input(
        sample["input"],
        sample["question"],
        prompt["examples"],
        prompt["instruction"],
        prompt["post_prompt"],
        template=DEFAULT_TEMPLATE,
    )


def preprocess_official_output(output: str) -> str:
    """Apply current official BABILong output preprocessing exactly."""
    output = output.lower()
    output = output.split(".")[0]
    output = output.split("<context>")[0]
    output = output.split("<example>")[0]
    output = output.split("Question")[0]
    return output


def compare_official(
    target: str, output: str, question: str, task_labels: list[str]
) -> bool:
    """Apply current official label-aware BABILong comparison semantics."""
    output = preprocess_official_output(output)
    target = target.lower()
    labels = {label.lower() for label in task_labels}
    labels_in_question = {label for label in labels if label in question.lower()}
    labels_in_output = {label for label in labels if label in output}
    labels_in_output -= labels_in_question

    if "," in target and len(target) > 3:
        subtargets = target.split(",")
        return all(item in labels_in_output for item in subtargets) and len(
            labels_in_output
        ) == len(subtargets)
    return target in labels_in_output and len(labels_in_output) == 1


def compare_legacy(target: str, output: str) -> bool:
    """Preserve the current OpenCompass BABILong substring scorer."""
    target = target.lower()
    output = output.lower().split(".")[0]
    output = output.split("<context>")[0]
    output = output.split("<example>")[0]
    return target in output


def load_samples(path: str | Path, task: str, split_name: str) -> list[dict[str, Any]]:
    """Load one task/split and attach the fully formatted official prompt."""
    if task not in TASKS:
        raise ValueError(f"Unsupported BABILong task: {task}")
    if split_name not in SPLITS:
        raise ValueError(f"Unsupported BABILong split: {split_name}")
    source = Path(path).expanduser() / task / f"{split_name}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing BABILong data file: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list in {source}")

    rows = []
    for index, sample in enumerate(payload):
        target = str(sample["target"])
        question = str(sample["question"])
        rows.append(
            {
                "prompt": format_prompt(sample, task),
                "reference": {"target": target, "question": question},
                "target": target,
                "question": question,
                "task": task,
                "split": split_name,
                "sample_index": index,
            }
        )
    return rows


@LOAD_DATASET.register_module()
class BabiLongLocalDataset(BaseDataset):
    """Load the local 100-sample BABILong JSON files."""

    @staticmethod
    def load(path: str, task: str, split_name: str, **kwargs) -> Dataset:
        return Dataset.from_list(load_samples(path, task, split_name))


def _reference_fields(reference: Any) -> tuple[str, str]:
    if isinstance(reference, dict):
        return str(reference["target"]), str(reference["question"])
    if isinstance(reference, (list, tuple)) and len(reference) == 1:
        return _reference_fields(reference[0])
    raise TypeError(f"Invalid BABILong reference: {reference!r}")


@ICL_EVALUATORS.register_module()
class BabiLongDualEvaluator(BaseEvaluator):
    """Report official label-aware and legacy substring accuracy."""

    def __init__(self, task: str):
        if task not in TASK_LABELS:
            raise ValueError(f"Unsupported BABILong task: {task}")
        self.task = task

    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("predictions and references have different lengths")
        if not predictions:
            return {"score_official": 0.0, "score_legacy": 0.0}

        official = 0
        legacy = 0
        for prediction, reference in zip(predictions, references):
            target, question = _reference_fields(reference)
            text = str(prediction)
            official += compare_official(
                target, text, question, TASK_LABELS[self.task]
            )
            legacy += compare_legacy(target, text)
        count = len(predictions)
        return {
            "score_official": round(100 * official / count, 2),
            "score_legacy": round(100 * legacy / count, 2),
        }


__all__ = [
    "BabiLongDualEvaluator",
    "BabiLongLocalDataset",
    "SPLITS",
    "TASKS",
    "TASK_LABELS",
    "compare_legacy",
    "compare_official",
    "format_prompt",
    "load_samples",
    "preprocess_official_output",
]
