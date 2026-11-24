"""End-to-end integration test of the CLI pipeline on the committed tiny fixture dataset."""

from __future__ import annotations

import json
from pathlib import Path

from aml_graph.cli import main

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def test_end_to_end_cli_pipeline_on_fixture(tmp_path: Path) -> None:
    cfg_dir = str(FIXTURE_ROOT / "configs")
    data_dir = str(FIXTURE_ROOT / "data")
    out_dir = tmp_path / "outputs"

    assert (
        main(
            [
                "data",
                "--no-download",
                "--config-dir",
                cfg_dir,
                "--data-root",
                data_dir,
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "features",
                "--config-dir",
                cfg_dir,
                "--data-root",
                data_dir,
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "experiments",
                "--config-dir",
                cfg_dir,
                "--data-root",
                data_dir,
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "report",
                "--config-dir",
                cfg_dir,
                "--data-root",
                data_dir,
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    expected_files = [
        "ibm_features.csv",
        "elliptic2_features.csv",
        "prep_log.json",
        "results_raw.csv",
        "results_summary.csv",
        "results_lift.png",
        "unit4_selected_hyperparameters.csv",
        "unit5_holdout_comparison.csv",
        "unit5_archival_published_means.csv",
        "unit5_archival_model_comparison.png",
        "provenance_data.json",
        "provenance_features.json",
        "provenance_experiments.json",
        "provenance_report.json",
        "research_extension/common_bfs_model_results.csv",
        "research_extension/feature_ablation.csv",
        "research_extension/domain_shift_train_only.csv",
        "research_extension/IBM_activity_feature_experiment.csv",
        "research_extension/research_extension_summary.json",
        "phase2/label_blind_candidates.csv",
        "phase2/fold_metrics.csv",
        "phase2/ipw_appendix_audit.csv",
        "phase2/summary_by_regime.csv",
        "phase2/paired_comparisons.csv",
        "phase2/paired_comparisons_by_fold.csv",
        "phase2/uniform_seed_check.csv",
        "phase2/uniform_seed_check_by_fold.csv",
        "phase2/before_after_comparison.csv",
        "phase2/window_leak_audit_summary.csv",
        "phase2/before_vs_after_benchmark.png",
        "phase2/window_leak_audit.png",
        "phase2/rolling_vs_account_folds.png",
    ]
    for rel in expected_files:
        assert (out_dir / rel).exists(), f"Missing expected output: {rel}"

    prov = json.loads((out_dir / "provenance_experiments.json").read_text(encoding="utf-8"))
    assert "library_versions" in prov
    assert "config_hashes" in prov
    assert "combined_sha256" in prov["config_hashes"]
