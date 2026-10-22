"""Fixed character-span selection and token-weighted perplexity."""

import math


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError("Expected nonempty half-open character intervals")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def selected_loss_indices(offsets, intervals):
    spans = intervals
    for start, end in spans:
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError("Invalid historical character interval")
    indices = []
    token_index = 0
    span_index = 0
    while token_index < len(offsets) and span_index < len(spans):
        start, end = offsets[token_index]
        span_start, span_end = spans[span_index]
        if token_index == 0 or end <= start:
            token_index += 1
            continue
        if start >= span_start and end <= span_end:
            indices.append(token_index - 1)
            token_index += 1
        elif start < span_start:
            token_index += 1
        else:
            span_index += 1
    return indices


class HistoricalPerplexityAccumulator:
    def __init__(self):
        self.weighted_sum = 0.0
        self.token_count = 0
        self.samples = 0

    def add(self, losses, weight=None, min_count=1):
        values = [float(value) for value in losses]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("NLL values must be finite and nonnegative")
        count = len(values) if weight is None else weight
        if not values or count < min_count:
            return
        if not isinstance(count, int) or count < 1:
            raise ValueError("Historical weight must be a positive integer")
        self.weighted_sum += math.exp(math.fsum(values) / len(values)) * count
        self.token_count += count
        self.samples += 1

    def result(self):
        return {"perplexity": self.weighted_sum / self.token_count if self.token_count else None,
                "token_count": self.token_count, "samples": self.samples}
