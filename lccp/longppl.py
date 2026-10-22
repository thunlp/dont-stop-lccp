"""Evaluate fixed annotations; no reference-model selection at evaluation time."""

import argparse
import json
from pathlib import Path

from .metrics import HistoricalPerplexityAccumulator, selected_loss_indices


def token_losses(model, input_ids, chunk_size=256, backend="full"):
    import torch
    from torch.nn import functional

    if chunk_size < 1 or input_ids.shape[1] < 2:
        raise ValueError("Need a positive chunk size and at least two tokens")
    with torch.inference_mode():
        if backend == "minicpm-head":
            if model.config.model_type != "minicpm":
                raise ValueError("Chunked head is verified only for the release MiniCPM implementation")
            hidden = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                 use_cache=False, return_dict=True)[0]
            logits = None
        elif backend == "full":
            logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, return_dict=True).logits
            hidden = None
        else:
            raise ValueError("Unknown loss backend")
        losses = []
        for start in range(0, input_ids.shape[1] - 1, chunk_size):
            end = min(start + chunk_size, input_ids.shape[1] - 1)
            scores = model.lm_head(hidden[:, start:end]) if hidden is not None else logits[:, start:end]
            labels = input_ids[:, start + 1:end + 1]
            loss = functional.cross_entropy(scores[0].float(), labels[0], reduction="none")
            losses.extend(loss.cpu().tolist())
    return losses


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code,
        torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attention,
    ).to(args.device).eval()
    if args.attention == "flash_attention_2" and getattr(model.config, "attention_sink_enabled", False):
        from .backend import require_native_sink
        require_native_sink(model)
    return model, tokenizer


def model_arguments(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attention", default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--loss-backend", choices=["full", "minicpm-head"], default="full")
    parser.add_argument("--loss-chunk-size", type=int, default=256)


def evaluate(model, tokenizer, records, max_length, chunk_size, backend, *, min_key_tokens=10):
    if not isinstance(min_key_tokens, int) or min_key_tokens < 1:
        raise ValueError("Minimum selected token count must be a positive integer")
    historical = {name: HistoricalPerplexityAccumulator() for name in ["ppl", "longppl", "shortppl"]}
    sample_count = 0
    seen = set()
    for record in records:
        if record["id"] in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(record["id"])
        text = record["text"]
        for key in ["long_spans", "short_spans"]:
            if any(end > len(text) for start, end in record[key]):
                raise ValueError("Annotation extends beyond text")
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False,
                            truncation=True, max_length=max_length)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        losses = token_losses(model, encoded["input_ids"].to(model.device), chunk_size, backend)
        historical["ppl"].add(losses, weight=len(offsets))
        for name, key in [("longppl", "long_spans"), ("shortppl", "short_spans")]:
            selected = selected_loss_indices(offsets, record[key])
            historical[name].add([losses[index] for index in selected], min_count=min_key_tokens)
        sample_count += 1
    if not sample_count:
        raise ValueError("Empty evaluation dataset")
    return {"samples": sample_count, "max_length": max_length, "min_key_tokens": min_key_tokens,
            "aggregation": "historical",
            "metrics": {name: accumulator.result() for name, accumulator in historical.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--min-key-tokens", type=int, default=10)
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("max-length must be at least two")
    model, tokenizer = load_model(args)
    if args.data.is_dir():
        from .data import read_fixed_dataset
        result = evaluate(model, tokenizer, read_fixed_dataset(args.data),
                          args.max_length, args.loss_chunk_size, args.loss_backend,
                          min_key_tokens=args.min_key_tokens)
    else:
        with args.data.open() as stream:
            result = evaluate(model, tokenizer, (json.loads(line) for line in stream if line.strip()),
                              args.max_length, args.loss_chunk_size, args.loss_backend,
                              min_key_tokens=args.min_key_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
