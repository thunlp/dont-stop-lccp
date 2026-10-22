"""Equal-weight Huber scaling fit with explicit one-sided exponent regularization."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def predict_log(theta, counts):
    counts = np.asarray(counts, dtype=float)
    if np.any(~np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("Counts must be positive and finite")
    return np.exp(theta[0] - theta[1] * np.log(counts[:, 0])) + np.exp(
        theta[2] - theta[3] * np.log(counts[:, 1])) + np.exp(
        theta[4] - theta[5] * np.log(counts[:, 2])) + theta[6]


def slice_exponent(counts, observed, axis):
    other_axes = [index for index in range(3) if index != axis]
    groups = {}
    for index, row in enumerate(counts):
        groups.setdefault(tuple(row[other_axes]), []).append(index)
    exponents = []
    for indices in groups.values():
        values = counts[indices, axis]
        if len(np.unique(values)) < 4:
            continue
        scaled = values / values.min()
        targets = observed[indices]
        best = None
        for initial in [0.1, 0.25, 0.5, 1.0]:
            result = least_squares(
                lambda parameters: parameters[0] * scaled ** (-parameters[1]) + parameters[2] - targets,
                [max(float(np.ptp(targets)), 0.1), initial, float(targets.min())],
                bounds=([0, 0.001, -20], [100, 3, 20]), max_nfev=5000,
            )
            if result.success and (best is None or result.cost < best.cost):
                best = result
        if best is not None:
            exponents.append(float(best.x[1]))
    if not exponents:
        raise ValueError("Need training-only slices with at least four distinct axis values")
    return float(np.median(exponents)), len(exponents)


def objective(theta, counts, observed, thresholds, f_scale=0.05, penalties=(1.0, 1.0)):
    residuals = predict_log(theta, counts) - observed
    absolute = np.abs(residuals)
    huber = np.where(absolute <= f_scale, 0.5 * residuals ** 2,
                     f_scale * (absolute - 0.5 * f_scale))
    excess = np.maximum(theta[[3, 5]] - thresholds, 0)
    return float(huber.sum() + (np.asarray(penalties) * np.square(excess)).sum())


def mixed_loss(data_count, loss_name="huber", f_scale=0.05):
    def loss(squared):
        result = np.vstack([squared.copy(), np.ones_like(squared), np.zeros_like(squared)])
        if loss_name == "huber":
            mask = (np.arange(len(squared)) < data_count) & (squared > f_scale ** 2)
            root = np.sqrt(squared[mask])
            result[0, mask] = 2 * f_scale * root - f_scale ** 2
            result[1, mask] = f_scale / root
            result[2, mask] = -0.5 * f_scale / (squared[mask] * root)
        return result
    return loss


def fit(counts, observed, restarts=32, seed=0, threshold_mode="fixed", thresholds=(0.2, 0.2),
        penalties=(1.0, 1.0), loss_name="huber", f_scale=0.05,
        sample_weights=None, historical_penalty_convention=False):
    counts = np.asarray(counts, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if (counts.ndim != 2 or counts.shape[1] != 3 or observed.ndim != 1
            or len(observed) == 0 or restarts < 1 or len(counts) != len(observed)
            or np.any(~np.isfinite(observed))):
        raise ValueError("Invalid training data or restart count")
    predict_log(np.zeros(7), counts)
    if loss_name not in {"huber", "linear"} or threshold_mode not in {"fixed", "slice"}:
        raise ValueError("Unknown loss or threshold mode")
    if not np.isfinite(f_scale) or f_scale <= 0:
        raise ValueError("f_scale must be positive")
    penalties = np.asarray(penalties, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if (penalties.shape != (2,) or thresholds.shape != (2,)
            or np.any(~np.isfinite(penalties)) or np.any(~np.isfinite(thresholds))
            or np.any(penalties < 0) or np.any(thresholds <= 0)):
        raise ValueError("Invalid exponent penalties or thresholds")
    short, short_slices = slice_exponent(counts, observed, 1)
    long, long_slices = slice_exponent(counts, observed, 2)
    if threshold_mode == "slice":
        thresholds = np.array([short, long])
    weights = np.ones(len(observed)) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if weights.shape != observed.shape or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Sample weights must be finite and positive")
    reference = np.exp(np.mean(np.log(counts), axis=0))
    logarithms = np.log(counts / reference)
    coefficient_indices = np.array([0, 2, 4])
    exponent_indices = np.array([1, 3, 5])
    lower = np.array([-20, 0.001, -20, 0.001, -20, 0.001, -5.0])
    upper = np.array([20, 3, 20, 3, 20, 3, 5.0])
    penalty_scale = np.sqrt(penalties * (1 if historical_penalty_convention else 2))
    def residuals(parameters):
        components = np.exp(parameters[coefficient_indices] - logarithms * parameters[exponent_indices])
        data = (components.sum(axis=1) + parameters[6] - observed) * np.sqrt(weights)
        return np.r_[data, penalty_scale * np.maximum(parameters[[3, 5]] - thresholds, 0)]
    def jacobian(parameters):
        components = np.exp(parameters[coefficient_indices] - logarithms * parameters[exponent_indices])
        result = np.zeros((len(observed) + 2, 7))
        result[:len(observed), coefficient_indices] = components * np.sqrt(weights)[:, None]
        result[:len(observed), exponent_indices] = -components * logarithms * np.sqrt(weights)[:, None]
        result[:len(observed), 6] = np.sqrt(weights)
        result[len(observed) + np.arange(2), [3, 5]] = penalty_scale[:, None] * (parameters[[3, 5]] > thresholds)[:, None]
        return result
    random = np.random.default_rng(seed)
    best = None
    successes = 0
    for restart in range(restarts):
        exponents = np.array([0.23, min(short, 1), min(long, 1)])
        amplitudes = np.full(3, max(float(np.mean(observed)), 0.1) / 3)
        start = np.array([np.log(amplitudes[0]), exponents[0], np.log(amplitudes[1]), exponents[1], np.log(amplitudes[2]), exponents[2], 0.0])
        if restart:
            start += random.normal(size=7) * [0.7, 0.15, 0.7, 0.15, 0.7, 0.15, 0.5]
        start = np.clip(start, lower + 1e-8, upper - 1e-8)
        result = least_squares(residuals, start, jac=jacobian, bounds=(lower, upper),
                               loss=mixed_loss(len(observed), loss_name, f_scale),
                               max_nfev=10000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if result.success and np.isfinite(result.cost):
            successes += 1
        if result.success and np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        raise RuntimeError("No optimization restart converged")
    theta = best.x.copy()
    theta[coefficient_indices] += theta[exponent_indices] * np.log(reference)
    return theta, {"beta_s_slice_median": short, "beta_l_slice_median": long,
                   "beta_s_threshold": float(thresholds[0]), "beta_l_threshold": float(thresholds[1]),
                   "threshold_mode": threshold_mode, "short_slices": short_slices,
                   "long_slices": long_slices, "objective": float(best.cost),
                   "converged_restarts": successes, "reference_counts": reference.tolist(),
                   "loss": loss_name, "f_scale": f_scale, "penalties_short_long": penalties.tolist(),
                   "penalty_convention": "half-lambda-squared" if historical_penalty_convention else "lambda-squared",
                   "regularizer_is_huberized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-scales", nargs="+", required=True)
    parser.add_argument("--holdout-scale", required=True)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--threshold-mode", choices=["fixed", "slice"], default="fixed")
    parser.add_argument("--beta-s-threshold", type=float, default=0.2)
    parser.add_argument("--beta-l-threshold", type=float, default=0.2)
    parser.add_argument("--lambda-s", type=float, default=1.0)
    parser.add_argument("--lambda-l", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.holdout_scale in args.train_scales:
        parser.error("Held-out scale must not appear in training")
    with args.data.open() as stream:
        rows = list(csv.DictReader(stream))
    train = [row for row in rows if row["scale"] in args.train_scales]
    holdout = [row for row in rows if row["scale"] == args.holdout_scale]
    if not train or not holdout or set(args.train_scales) != {row["scale"] for row in train}:
        parser.error("Requested split is missing from data")
    def arrays(subset):
        counts = np.array([[float(row[key]) for key in ["N", "D_short", "D_long"]] for row in subset])
        ppl = np.array([float(row["longppl"]) for row in subset])
        if np.any(~np.isfinite(ppl)) or np.any(ppl < 1):
            raise ValueError("Perplexity must be finite and at least one")
        return counts, np.log(ppl)
    counts, observed = arrays(train)
    theta, diagnostics = fit(counts, observed, args.restarts, args.seed, args.threshold_mode,
                             (args.beta_s_threshold, args.beta_l_threshold),
                             (args.lambda_s, args.lambda_l))
    params = dict(zip(["A", "alpha", "B_s", "beta_s", "B_l", "beta_l", "C"],
                      [float(np.exp(theta[0])), float(theta[1]), float(np.exp(theta[2])), float(theta[3]),
                       float(np.exp(theta[4])), float(theta[5]), float(theta[6])]))
    results = {"implementation": "huber-with-beta-regularization", "parameters": params,
               "diagnostics": diagnostics, "train_scales": args.train_scales, "holdout_scale": args.holdout_scale,
               "seed": args.seed, "restarts": args.restarts, "f_scale": 0.05,
               "lambda_s": args.lambda_s, "lambda_l": args.lambda_l}
    for name, subset in [("train", train), ("holdout", holdout)]:
        split_counts, target = arrays(subset)
        predicted = predict_log(theta, split_counts)
        results[name] = {"samples": len(subset), "rmse_log": float(np.sqrt(np.mean((predicted - target) ** 2))),
                         "predicted_longppl": np.exp(predicted).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
