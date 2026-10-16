import csv
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from uav_path_planning import (
    Bounds3D,
    Environment3D,
    ExperimentRecord,
    Path3D,
    bootstrap_ci,
    compare_paired,
    evaluate_path,
    holm_adjust,
    metric_higher_is_better,
    summarize_by_group,
    wilcoxon_signed_rank,
    write_statistical_report,
)


def _record(algorithm: str, scenario: str, seed: int, length: float, feasible: bool = True) -> ExperimentRecord:
    environment = Environment3D(Bounds3D((0.0, 0.0, 0.0), (20.0, 20.0, 20.0)))
    path = Path3D([(0.0, 0.0, 0.0), (length, 0.0, 0.0)])
    metrics = evaluate_path(path, environment)
    # Keep the fixture deterministic while allowing an explicit infeasible row.
    if not feasible:
        metrics = replace(metrics, feasible=False, objective=metrics.objective + 100.0)
    return ExperimentRecord(algorithm, scenario, seed, 0.1 + seed, 10 + seed, metrics)


class StatisticalReportingTests(unittest.TestCase):
    def test_descriptive_statistics_are_grouped_by_scenario(self):
        records = [
            _record("de", "flat", 0, 10.0),
            _record("de", "flat", 1, 14.0),
            _record("de", "flat", 2, 12.0, feasible=False),
            _record("de", "city", 0, 20.0),
        ]
        rows = summarize_by_group(records, metrics=("length", "feasible", "runtime_seconds"))
        flat_length = next(row for row in rows if row.scenario == "flat" and row.metric == "length")
        self.assertEqual(flat_length.n, 3)
        self.assertEqual(flat_length.valid_n, 2)
        self.assertEqual(flat_length.failure_n, 1)
        self.assertAlmostEqual(flat_length.mean, 12.0)
        self.assertAlmostEqual(flat_length.median, 12.0)
        self.assertAlmostEqual(flat_length.std, math.sqrt(8.0))
        self.assertAlmostEqual(flat_length.iqr, 2.0)
        self.assertAlmostEqual(flat_length.feasible_rate, 2.0 / 3.0)

    def test_exact_wilcoxon_and_zero_pairs(self):
        result = wilcoxon_signed_rank([1.0, 2.0, 3.0, 0.0])
        self.assertEqual(result.n, 3)
        self.assertEqual(result.method, "exact sign enumeration")
        self.assertAlmostEqual(result.p_value, 0.25)
        all_zero = wilcoxon_signed_rank([0.0, 0.0])
        self.assertEqual(all_zero.n, 0)
        self.assertEqual(all_zero.p_value, 1.0)

    def test_empty_and_nonfinite_wilcoxon_inputs_are_not_zero_differences(self):
        for differences in ([], [math.nan, math.inf]):
            with self.subTest(differences=differences):
                result = wilcoxon_signed_rank(differences)
                self.assertEqual(result.n, 0)
                self.assertTrue(math.isnan(result.p_value))
                self.assertTrue(math.isnan(result.statistic))
                self.assertIn("no finite", result.method)

    def test_large_wilcoxon_uses_normal_approximation(self):
        result = wilcoxon_signed_rank([1.0] * 21)
        self.assertEqual(result.n, 21)
        self.assertIn("normal approximation", result.method)
        self.assertLess(result.p_value, 0.01)

    def test_wilcoxon_cross_checks_scipy_when_available(self):
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            self.skipTest("SciPy is optional")
        differences = [1.0, 2.0, 3.0, -4.0, 5.0]
        ours = wilcoxon_signed_rank(differences)
        reference = wilcoxon(differences, zero_method="wilcox", method="exact")
        self.assertAlmostEqual(ours.statistic, float(reference.statistic))
        self.assertAlmostEqual(ours.p_value, float(reference.pvalue))

    def test_bootstrap_is_reproducible_and_handles_singleton(self):
        first = bootstrap_ci([1.0, 2.0, 4.0], n_resamples=1000, seed=11)
        second = bootstrap_ci([1.0, 2.0, 4.0], n_resamples=1000, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(bootstrap_ci([3.0], n_resamples=20), (3.0, 3.0))

    def test_paired_comparison_matches_scenario_and_seed(self):
        records = []
        for scenario in ("flat", "city"):
            for seed in (0, 1, 2):
                records.append(_record("de", scenario, seed, 10.0 + seed))
                records.append(_record("adaptive_de", scenario, seed, 8.0 + seed))
        comparison = compare_paired(records, proposed="adaptive_de", baseline="de", metric="length", bootstrap_resamples=1000)
        self.assertEqual(comparison.n_pairs, 6)
        expected = ((2.0 / 10.0) + (2.0 / 11.0) + (2.0 / 12.0)) / 3.0 * 100.0
        self.assertAlmostEqual(comparison.mean_relative_improvement_pct, expected, places=4)
        self.assertAlmostEqual(comparison.win_rate_pct, 100.0)
        self.assertGreater(comparison.paired_cohens_d, 0.0)
        self.assertAlmostEqual(comparison.rank_biserial_effect, 1.0)
        self.assertLessEqual(comparison.bootstrap_ci_low_pct, comparison.mean_relative_improvement_pct)
        self.assertGreaterEqual(comparison.bootstrap_ci_high_pct, comparison.mean_relative_improvement_pct)

    def test_report_writes_markdown_and_csv(self):
        records = []
        for seed in (0, 1):
            records.extend([_record("de", "flat", seed, 10.0 + seed), _record("adaptive_de", "flat", seed, 8.0 + seed)])
        with tempfile.TemporaryDirectory() as directory:
            descriptive, paired = write_statistical_report(
                records,
                directory,
                proposed="adaptive_de",
                baselines=("de",),
                metrics=("length",),
                bootstrap_resamples=100,
            )
            self.assertTrue(descriptive)
            self.assertEqual(len(paired), 1)
            report = Path(directory) / "statistical_report.md"
            paired_csv = Path(directory) / "statistical_report_paired.csv"
            self.assertIn("Wilcoxon", report.read_text(encoding="utf-8"))
            with paired_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_pairs"], "2")

    def test_failed_length_pairs_are_excluded_but_failures_are_reported(self):
        records = [
            _record("de", "scene", 0, 10.0, feasible=True),
            _record("adaptive_de", "scene", 0, 8.0, feasible=True),
            _record("de", "scene", 1, 10.0, feasible=False),
            _record("adaptive_de", "scene", 1, 1.0, feasible=True),
        ]
        comparison = compare_paired(records, proposed="adaptive_de", baseline="de", metric="length")
        self.assertEqual(comparison.n_pairs, 2)
        self.assertEqual(comparison.n_compared, 1)
        self.assertEqual(comparison.n_valid_pairs, 1)
        self.assertEqual(comparison.n_excluded_pairs, 1)
        self.assertEqual(comparison.either_failed_pairs, 1)
        self.assertEqual(comparison.proposed_failures, 0)
        self.assertEqual(comparison.baseline_failures, 1)
        self.assertAlmostEqual(comparison.mean_relative_improvement_pct, 20.0)

    def test_scenario_unit_avoids_pseudoreplication(self):
        records = []
        for seed in range(3):
            records.extend([_record("de", "A", seed, 10.0 + seed), _record("adaptive_de", "A", seed, 8.0 + seed)])
        records.extend([_record("de", "B", 0, 20.0), _record("adaptive_de", "B", 0, 10.0)])
        seed_result = compare_paired(records, proposed="adaptive_de", baseline="de", metric="length", unit="seed")
        scene_result = compare_paired(records, proposed="adaptive_de", baseline="de", metric="length", unit="scenario")
        self.assertEqual(seed_result.n_units, 4)
        self.assertEqual(scene_result.n_units, 2)
        self.assertEqual(scene_result.n_scenarios, 2)
        self.assertEqual(scene_result.n_pairs, 4)
        self.assertEqual(scene_result.n_valid_pairs, 4)
        self.assertEqual(scene_result.n_excluded_pairs, 0)

    def test_scenario_aggregation_does_not_count_seeds_as_excluded_pairs(self):
        records = []
        for scenario in range(8):
            for seed in range(3):
                records.extend([
                    _record("de", str(scenario), seed, 10.0),
                    _record("adaptive_de", str(scenario), seed, 8.0),
                ])
        with tempfile.TemporaryDirectory() as directory:
            _, comparisons = write_statistical_report(
                records, directory, proposed="adaptive_de", baselines=("de",),
                metrics=("objective",), bootstrap_resamples=10,
            )
            result = comparisons[0]
            self.assertFalse(result.feasible_only)
            self.assertEqual(result.n_pairs, 24)
            self.assertEqual(result.n_valid_pairs, 24)
            self.assertEqual(result.n_excluded_pairs, 0)
            self.assertEqual(result.n_units, 8)
            report = (Path(directory) / "statistical_report.md").read_text(encoding="utf-8")
            self.assertIn("| objective | 24 | 24 | 8 | 0 |", report)

    def test_no_path_fallback_is_failure_in_summaries_and_comparisons(self):
        baseline = _record("de", "scene", 0, 10.0)
        fallback = replace(_record("adaptive_de", "scene", 0, 8.0), status="no_path")
        self.assertTrue(fallback.metrics.feasible)
        rows = summarize_by_group([fallback], metrics=("length", "smoothness", "feasible"))
        for row in rows:
            self.assertEqual(row.failure_n, 1)
            self.assertEqual(row.feasible_n, 0)
            self.assertEqual(row.feasible_rate, 0.0)
            if row.metric in {"length", "smoothness"}:
                self.assertEqual(row.valid_n, 0)
                self.assertTrue(math.isnan(row.mean))
            else:
                self.assertEqual(row.mean, 0.0)
        result = compare_paired([baseline, fallback], proposed="adaptive_de", baseline="de", metric="length")
        self.assertEqual(result.proposed_failures, 1)
        self.assertEqual(result.n_valid_pairs, 0)
        self.assertEqual(result.n_excluded_pairs, 1)
        self.assertEqual(result.feasible_rate_proposed, 0.0)
        self.assertTrue(math.isnan(result.wilcoxon_p_value))
        self.assertIn("after filtering", result.wilcoxon_method)

    def test_nonfinite_pairs_are_counted_before_scenario_aggregation(self):
        baseline = _record("de", "scene", 1, 10.0)
        nonfinite = replace(baseline, metrics=replace(baseline.metrics, objective=math.nan))
        records = [
            _record("de", "scene", 0, 10.0), _record("adaptive_de", "scene", 0, 8.0),
            nonfinite, _record("adaptive_de", "scene", 1, 8.0),
        ]
        result = compare_paired(records, proposed="adaptive_de", baseline="de", metric="objective", unit="scenario")
        self.assertEqual((result.n_pairs, result.n_valid_pairs, result.n_excluded_pairs), (2, 1, 1))
        self.assertEqual(result.n_units, 1)
        self.assertEqual(result.either_failed_pairs, 0)

    def test_no_matched_pairs_have_no_test_or_effect_estimate(self):
        records = [_record("de", "A", 0, 10.0), _record("adaptive_de", "B", 0, 8.0)]
        result = compare_paired(records, proposed="adaptive_de", baseline="de", metric="length")
        self.assertEqual(result.n_pairs, 0)
        self.assertEqual(result.n_valid_pairs, 0)
        self.assertEqual(result.n_excluded_pairs, 0)
        self.assertTrue(math.isnan(result.wilcoxon_p_value))
        self.assertTrue(math.isnan(result.paired_cohens_d))
        self.assertTrue(math.isnan(result.rank_biserial_effect))
        self.assertIn("no matched pairs", result.wilcoxon_method)

    def test_direction_and_holm_correction(self):
        self.assertTrue(metric_higher_is_better("clearance"))
        self.assertTrue(metric_higher_is_better("min_obstacle_distance"))
        self.assertFalse(metric_higher_is_better("length"))
        adjusted = holm_adjust([0.01, 0.04, 0.001])
        self.assertEqual(len(adjusted), 3)
        self.assertGreaterEqual(adjusted[0], 0.01)
        self.assertAlmostEqual(adjusted[2], 0.003)

    def test_duplicate_pair_is_rejected(self):
        records = [
            _record("de", "scene", 0, 10.0),
            _record("de", "scene", 0, 11.0),
            _record("adaptive_de", "scene", 0, 8.0),
        ]
        with self.assertRaises(ValueError):
            compare_paired(records, proposed="adaptive_de", baseline="de", metric="length")


if __name__ == "__main__":
    unittest.main()
