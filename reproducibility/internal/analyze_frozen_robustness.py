"""Post hoc map-level sensitivity analysis; never runs or tunes an optimizer.

Usage: py -3.12 analyze_frozen_robustness.py [--data PATH] [--output PATH]
Requires numpy only. The analysis protocol is stored beside this script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import sys

import numpy as np

ZERO_TOLERANCE = 1e-12
BOOTSTRAPS = 10000
RANDOM_SEED = 20261014


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def signed_ranks(differences):
    values = [float(x) for x in differences if abs(float(x)) > ZERO_TOLERANCE]
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    doubled_ranks = [0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]]) == abs(values[order[start]]):
            end += 1
        for i in order[start:end]:
            doubled_ranks[i] = start + 1 + end
        start = end
    return values, doubled_ranks


def exact_signed_rank(differences):
    """Exact conditional sign distribution, with ties and zeros as in source."""
    values, ranks = signed_ranks(differences)
    if not values:
        return 1.0
    total = sum(ranks)
    observed = sum(rank for value, rank in zip(values, ranks) if value > 0)
    counts = [0] * (total + 1)
    counts[0] = 1
    reachable = 0
    for rank in ranks:
        for score in range(reachable, -1, -1):
            counts[score + rank] += counts[score]
        reachable += rank
    assert sum(counts) == 2 ** len(values)
    distance = abs(2 * observed - total)
    return sum(count for score, count in enumerate(counts)
               if abs(2 * score - total) >= distance) / (2 ** len(values))


def original_signed_rank(differences):
    values, ranks2 = signed_ranks(differences)
    if len(values) <= 20:
        return exact_signed_rank(values)
    ranks = [x / 2 for x in ranks2]
    observed = sum(rank for value, rank in zip(values, ranks) if value > 0)
    center = sum(ranks) / 2
    variance = sum(rank * rank for rank in ranks) / 4
    direction = 1 if observed >= center else -1
    z = (observed - center - direction * 0.5) / math.sqrt(variance)
    return math.erfc(abs(z) / math.sqrt(2))


def validate_exact_distribution():
    cases = ([0, 0, 0], [1], [1, -1], [1, 1, -1],
             [1, -2, 3, -4, 5], [0, 1e-13, -2, 2, 3, -3, 3])
    for differences in cases:
        values, ranks = signed_ranks(differences)
        total = sum(ranks)
        observed = sum(rank for value, rank in zip(values, ranks) if value > 0)
        scores = [sum(r for r, s in zip(ranks, signs) if s)
                  for signs in itertools.product((0, 1), repeat=len(values))]
        brute = sum(abs(2 * score - total) >= abs(2 * observed - total)
                    for score in scores) / len(scores)
        assert exact_signed_rank(differences) == brute
    return len(cases)


def holm(values):
    output = [0.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=lambda i: values[i])):
        previous = max(previous, min(1.0, values[index] * (len(values) - rank)))
        output[index] = previous
    return output


def map_family(scene):
    if "clutter" in scene:
        return "clutter"
    if "highrise" in scene or "urban" in scene:
        return "urban"
    if "barrier" in scene:
        return "barrier"
    raise ValueError(f"Unknown map family: {scene}")


def effect(pairs, metric):
    a = statistics.fmean(x[0] for x in pairs)
    b = statistics.fmean(x[1] for x in pairs)
    return 100 * (a / b - 1) if metric == "length" else 100 * (a - b)


def make_index(rows):
    index = {}
    for row in rows:
        key = (row["map_root"], row["scene"], int(row["seed"]), row["method"])
        assert key not in index, f"Duplicate run key: {key}"
        assert row["nominal_success"] in ("True", "False")
        if row["nominal_success"] == "False":
            assert float(row["cfr_0.6"]) == 0.0
        index[key] = row
    return index


def reconstruct(index, contrast):
    groups = defaultdict(list)
    total = excluded = 0
    for key, first in sorted(index.items()):
        if key[3] != contrast["reference"]:
            continue
        if contrast.get("map_root") and key[0] != contrast["map_root"]:
            continue
        other = index[key[:3] + (contrast["comparator"],)]
        total += 1
        if contrast["metric"] == "length" and (
            first["nominal_success"] == "False" or other["nominal_success"] == "False"
        ):
            excluded += 1
            continue
        metric = contrast["metric"]
        groups[key[:2]].append((float(first[metric]), float(other[metric])))
    keys = sorted(groups)
    pairs = [(statistics.fmean(x[0] for x in groups[key]),
              statistics.fmean(x[1] for x in groups[key])) for key in keys]
    assert total == int(contrast["matched_runs"])
    assert excluded == int(contrast["excluded_failed_pairs"])
    assert len(keys) == int(contrast["maps"])
    return keys, pairs, groups


def stratified_interval(keys, pairs, metric, identity):
    # Keep the synthetic family mixture fixed; complete maps are the units.
    rng_seed = RANDOM_SEED ^ int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    strata = defaultdict(list)
    for i, (root, scene) in enumerate(keys):
        strata[(root, map_family(scene))].append(i)
    draws = np.concatenate([rng.choice(indices, size=(BOOTSTRAPS, len(indices)), replace=True)
                            for _, indices in sorted(strata.items())], axis=1)
    means = np.asarray(pairs)[draws].mean(axis=1)
    sample_effects = (100 * (means[:, 0] / means[:, 1] - 1)
                      if metric == "length" else 100 * (means[:, 0] - means[:, 1]))
    return tuple(float(x) for x in np.quantile(sample_effects, [0.025, 0.975]))


def wilson_interval(successes, trials=500, z=1.959963984540054):
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return center - half, center + half


def threshold_analysis(cohort, rows):
    selected = [row for row in rows if row["method"] == "full"]
    n = len(selected)
    empirical = overlaps = lower_above = upper_below = empirical_overlap = 0
    path_rows = []
    for row in selected:
        rate = float(row["cfr_0.6"])
        count = round(500 * rate)
        assert abs(rate - count / 500) < 1e-12
        feasible = row["nominal_success"] == "True"
        passed = feasible and count >= 413
        assert passed == (row["pass_0.6"] == "True")
        low, high = wilson_interval(count)
        overlap = low <= 0.825 <= high
        empirical += passed
        overlaps += overlap
        empirical_overlap += passed and overlap
        lower_above += low > 0.825
        upper_below += high < 0.825
        path_rows.append({"cohort": cohort, "map_root": row["map_root"], "scene": row["scene"],
                          "seed": row["seed"], "nominal_success": feasible,
                          "collision_free_trials": count, "trials": 500,
                          "empirical_pass": passed, "wilson_95_low": low,
                          "wilson_95_high": high, "interval_covers_target": overlap})
    return ({"cohort": cohort, "method": "full", "paths": n, "external_trials_per_path": 500,
             "target_cfr": 0.825, "empirical_pass_paths": empirical,
             "empirical_pass_percent": 100 * empirical / n,
             "wilson_interval_covers_target_paths": overlaps,
             "wilson_interval_covers_target_percent": 100 * overlaps / n,
             "empirical_pass_with_interval_covering_target_paths": empirical_overlap,
             "wilson_lower_bound_above_target_paths": lower_above,
             "wilson_upper_bound_below_target_paths": upper_below}, path_rows)


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=here.parents[1] / "工作源稿" / "main" / "data")
    parser.add_argument("--output", type=Path, default=here)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    exact_test_cases = validate_exact_distribution()
    summary_rows, map_rows, deletion_rows, threshold_rows, threshold_paths = [], [], [], [], []
    source_hashes = {}
    validations = {"exact_distribution_brute_force_cases_passed": exact_test_cases,
                   "effect_max_absolute_difference": 0.0,
                   "original_p_max_absolute_difference": 0.0,
                   "original_holm_max_absolute_difference": 0.0,
                   "cohorts": {}}
    for cohort, expected in (("E", 92), ("P6", 14)):
        run_path = args.data / f"{cohort}_run_rows.csv"
        contrast_path = args.data / f"{cohort}_paired_statistics.csv"
        for path in (run_path, contrast_path):
            source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        runs, contrasts = read_csv(run_path), read_csv(contrast_path)
        assert len(contrasts) == expected
        index = make_index(runs)
        prepared, cohort_summary = [], []
        for contrast in contrasts:
            keys, pairs, groups = reconstruct(index, contrast)
            differences = [a - b for a, b in pairs]
            estimate = effect(pairs, contrast["metric"])
            identity = "/".join([cohort, contrast.get("map_root", "all"), contrast.get("category", "P6"),
                                  contrast["reference"], contrast["comparator"], contrast["metric"]])
            error = abs(estimate - float(contrast["effect_reference_minus_comparator"]))
            p_error = abs(original_signed_rank(differences) - float(contrast["p_value"]))
            validations["effect_max_absolute_difference"] = max(validations["effect_max_absolute_difference"], error)
            validations["original_p_max_absolute_difference"] = max(validations["original_p_max_absolute_difference"], p_error)
            assert error < 1e-10, (identity, "effect", error)
            assert p_error < 1e-10, (identity, "p", p_error)
            intervals = stratified_interval(keys, pairs, contrast["metric"], identity)
            leave_one_out = [effect(pairs[:i] + pairs[i + 1:], contrast["metric"]) for i in range(len(pairs))]
            same_sign = sum(value * estimate > 0 for value in leave_one_out)
            row = {"cohort": cohort, "map_root": contrast.get("map_root", "all"),
                   "category": contrast.get("category", "P6"), "reference": contrast["reference"],
                   "comparator": contrast["comparator"], "metric": contrast["metric"],
                   "maps": len(pairs), "matched_runs": contrast["matched_runs"],
                   "excluded_failed_pairs": contrast["excluded_failed_pairs"],
                   "effect": estimate, "effect_unit": contrast["effect_unit"],
                   "primary_p": float(contrast["p_value"]), "primary_p_holm": float(contrast["p_holm"]),
                   "exact_p": exact_signed_rank(differences),
                   "stratified_bootstrap_ci_low": intervals[0], "stratified_bootstrap_ci_high": intervals[1],
                   "bootstrap_resamples": BOOTSTRAPS, "loo_effect_min": min(leave_one_out),
                   "loo_effect_max": max(leave_one_out), "loo_same_effect_sign": same_sign,
                   "maps_reference_better": sum(d < -ZERO_TOLERANCE if contrast["metric"] == "length"
                                                else d > ZERO_TOLERANCE for d in differences),
                   "nonzero_maps": len(signed_ranks(differences)[0]),
                   "correction_family_size": expected}
            cohort_summary.append(row)
            prepared.append((keys, pairs, contrast))
            for i, (key, pair) in enumerate(zip(keys, pairs)):
                map_rows.append({"contrast_id": identity, "map_root": key[0], "scene": key[1],
                                 "family": map_family(key[1]), "paired_seeds": len(groups[key]),
                                 "reference_map_mean": pair[0], "comparator_map_mean": pair[1],
                                 "difference": differences[i], "effect_after_deleting_this_map": leave_one_out[i]})
        exact_adjusted = holm([row["exact_p"] for row in cohort_summary])
        primary_adjusted = holm([row["primary_p"] for row in cohort_summary])
        for row, exact_p_holm, primary_p_holm in zip(cohort_summary, exact_adjusted, primary_adjusted):
            row["exact_p_holm"] = exact_p_holm
            row["significance_changes_with_exact_test"] = (row["primary_p_holm"] < 0.05) != (exact_p_holm < 0.05)
            h_error = abs(primary_p_holm - row["primary_p_holm"])
            assert h_error < 1e-10
            validations["original_holm_max_absolute_difference"] = max(validations["original_holm_max_absolute_difference"], h_error)
        if cohort == "P6":
            all_keys = prepared[0][0]
            assert all(keys == all_keys for keys, _, _ in prepared)
            for key in all_keys:
                deleted = []
                for keys, pairs, contrast in prepared:
                    i = keys.index(key)
                    remaining = pairs[:i] + pairs[i + 1:]
                    deleted.append({"deleted_map_root": key[0], "deleted_scene": key[1],
                                    "comparator": contrast["comparator"], "metric": contrast["metric"],
                                    "remaining_maps": len(remaining), "effect": effect(remaining, contrast["metric"]),
                                    "exact_p": exact_signed_rank([a - b for a, b in remaining])})
                for row, value in zip(deleted, holm([row["exact_p"] for row in deleted])):
                    row["exact_p_holm_14"] = value
                deletion_rows.extend(deleted)
            for summary in cohort_summary:
                selected = [row for row in deletion_rows if row["comparator"] == summary["comparator"]
                            and row["metric"] == summary["metric"]]
                summary["loo_exact_p_holm_min"] = min(row["exact_p_holm_14"] for row in selected)
                summary["loo_exact_p_holm_max"] = max(row["exact_p_holm_14"] for row in selected)
                summary["loo_exact_holm_significant"] = sum(row["exact_p_holm_14"] < 0.05 for row in selected)
        summary_rows.extend(cohort_summary)
        threshold_summary, path_rows = threshold_analysis(cohort, runs)
        threshold_rows.append(threshold_summary)
        threshold_paths.extend(path_rows)
        validations["cohorts"][cohort] = {"run_rows": len(runs), "outcome_contrasts": len(contrasts),
                                              "reconstructed_all_effects": True,
                                              "reconstructed_all_primary_p_values": True,
                                              "failure_exclusion_counts_match": True,
                                              "holm_decisions_changed_with_exact_test": sum(r["significance_changes_with_exact_test"] for r in cohort_summary)}
    write_csv(args.output / "robustness_all_contrasts.csv", summary_rows)
    write_csv(args.output / "paired_map_rows.csv", map_rows)
    write_csv(args.output / "P6_leave_one_map_out.csv", deletion_rows)
    write_csv(args.output / "target_threshold_sensitivity.csv", threshold_rows)
    write_csv(args.output / "target_threshold_path_rows.csv", threshold_paths)
    validations["all_checks_passed"] = True
    write_json(args.output / "validation.json", validations)
    write_json(args.output / "provenance.json", {
        "analysis": "post hoc sensitivity analysis; frozen primary results are unchanged",
        "utc_completed": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "bootstrap_seed": RANDOM_SEED, "bootstrap_resamples": BOOTSTRAPS,
        "data_directory": str(args.data.resolve()), "source_sha256": source_hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((here / "analysis_protocol.json").read_bytes()).hexdigest()})
    print(json.dumps({"validation": validations, "threshold": threshold_rows,
                      "P6": [r for r in summary_rows if r["cohort"] == "P6"]}, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
