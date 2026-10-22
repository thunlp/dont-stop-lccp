"""Equal-weight average of the FOUR benchmark-level scores on a 0-100 scale."""

import argparse
import json
import math
from pathlib import Path


def aggregate(scores):
    expected = {"longbench", "graphwalks", "longreason", "babilong"}
    if set(scores) != expected:
        raise ValueError("Require exactly four benchmark-level scores")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100
           for value in scores.values()):
        raise ValueError("Scores must be finite and on a 0-100 scale")
    return sum(scores.values()) / 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    args = parser.parse_args()
    scores = json.loads(args.scores.read_text())
    print(json.dumps({"scores": scores, "four_benchmark_mean": aggregate(scores)}, indent=2))
