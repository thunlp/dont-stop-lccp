"""Local OpenCompass dataset and evaluator for GraphWalks."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET, TEXT_POSTPROCESSORS


def _parquet_files(path: Path, data_file: str | None) -> list[str]:
    if path.is_file():
        return [str(path)]
    if data_file:
        matches = sorted(path.rglob(data_file))
    else:
        matches = sorted(path.rglob("*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No GraphWalks parquet files found under {path}")
    return [str(match) for match in matches]


@LOAD_DATASET.register_module()
class GraphWalks32KDataset(BaseDataset):
    """Load GraphWalks and retain prompts fitting the evaluated chat template."""

    @staticmethod
    def load(
        path: str,
        tokenizer_path: str,
        problem_type: str | None = None,
        data_file: str | None = "graphwalks_128k_and_shorter.parquet",
        max_rendered_tokens: int = 32000,
        **kwargs,
    ) -> Dataset:
        source = Path(path).expanduser()
        if source.exists():
            raw = load_dataset(
                "parquet",
                data_files=_parquet_files(source, data_file),
                split="train",
            )
        else:
            raw = load_dataset(path, data_files=data_file, split="train")

        if problem_type is not None:
            raw = raw.filter(lambda row: row["problem_type"] == problem_type)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True,
        )

        def rendered_length(prompt: str) -> int:
            messages = [{"role": "user", "content": prompt}]
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                )
            except TypeError:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            return len(token_ids)

        raw = raw.filter(
            lambda row: rendered_length(row["prompt"]) <= max_rendered_tokens,
            desc=f"Filtering GraphWalks to <= {max_rendered_tokens} rendered tokens",
        )

        samples = []
        for item in raw:
            answer_nodes = item["answer_nodes"]
            if isinstance(answer_nodes, str):
                answer_nodes = ast.literal_eval(answer_nodes)
            samples.append(
                {
                    "prompt": item["prompt"],
                    "answer": list(answer_nodes),
                    "problem_type": item["problem_type"],
                }
            )
        if not samples:
            raise ValueError(
                f"No GraphWalks samples fit max_rendered_tokens={max_rendered_tokens}"
            )
        return Dataset.from_list(samples)


_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*\[([^\]]*)\]")


@TEXT_POSTPROCESSORS.register_module("graphwalks_strict")
def graphwalks_postprocess(text: str) -> str:
    """Keep an answer only when the official format occurs on the last line."""
    last_line = text.strip().split("\n")[-1] if text.strip() else ""
    match = _FINAL_ANSWER_RE.search(last_line)
    return match.group(0) if match else text


def _parse_node_list(text: str) -> set[str]:
    last_line = text.strip().split("\n")[-1] if text.strip() else ""
    match = _FINAL_ANSWER_RE.search(last_line)
    if not match:
        return set()
    return {
        item.strip().strip('"').strip("'")
        for item in match.group(1).split(",")
        if item.strip()
    }


@ICL_EVALUATORS.register_module()
class GraphWalksEvaluator(BaseEvaluator):
    """Official set precision/recall/F1, reported on a 0–100 scale."""

    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("predictions and references have different lengths")

        f1_scores = []
        precision_scores = []
        recall_scores = []
        for prediction, reference in zip(predictions, references):
            if isinstance(reference, dict):
                gold = reference.get("answer", [])
            elif isinstance(reference, (list, tuple)):
                gold = reference
            else:
                gold = []
            gold_set = set(gold)
            pred_set = _parse_node_list(str(prediction))
            overlap = len(pred_set & gold_set)

            if not pred_set and not gold_set:
                precision = recall = f1 = 1.0
            else:
                precision = overlap / len(pred_set) if pred_set else 0.0
                recall = overlap / len(gold_set) if gold_set else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        count = len(f1_scores)
        if not count:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "score": 0.0}
        metrics = {
            "f1": 100 * sum(f1_scores) / count,
            "precision": 100 * sum(precision_scores) / count,
            "recall": 100 * sum(recall_scores) / count,
        }
        metrics = {key: round(value, 2) for key, value in metrics.items()}
        metrics["score"] = metrics["f1"]
        return metrics
