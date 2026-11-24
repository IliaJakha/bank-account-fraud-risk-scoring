"""Command-line entrypoint for the AML subgraph structural analysis pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aml_graph.config import load_all_configs, save_run_metadata
from aml_graph.data import (
    ensure_sources,
    load_and_clean_ibm,
    load_and_prepare_elliptic2,
    parse_ibm_patterns,
)
from aml_graph.evaluate import run_extensions, run_legacy_benchmark, run_unit4_tuning
from aml_graph.features import build_feature_tables
from aml_graph.plots import generate_all_reports
from aml_graph.sampling import EdgeIndex, build_ibm_positives, sample_ibm_negatives


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    config_dir = Path(args.config_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data_root).resolve()
    return config_dir, output_dir, data_root


def cmd_data(args: argparse.Namespace) -> int:
    config_dir, output_dir, data_root = _resolve_paths(args)
    cfgs = load_all_configs(config_dir)
    collection = ensure_sources(
        cfgs["data"],
        data_root=data_root,
        download_missing=not args.no_download,
    )
    save_run_metadata(output_dir, "data", config_dir, extra={"collection": collection})
    print(f"Verified {len(collection) - 1} dataset files under {data_root}")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    config_dir, output_dir, data_root = _resolve_paths(args)
    cfgs = load_all_configs(config_dir)
    scfg = cfgs["sampling"]["ibm_sampling"]

    ibm_dir = data_root / "ibm"
    ell_dir = data_root / "elliptic2"

    collection = ensure_sources(cfgs["data"], data_root=data_root, download_missing=False)
    ibm, accounts, acc_id, ibm_stats = load_and_clean_ibm(ibm_dir)
    attempts, pat_stats = parse_ibm_patterns(ibm_dir, ibm, acc_id)
    ibm_stats.update(pat_stats)

    cc, nodes, edges_cc, ell_stats = load_and_prepare_elliptic2(ell_dir)

    idx = EdgeIndex(ibm, len(accounts))
    positives = build_ibm_positives(
        attempts,
        acc_id,
        idx,
        min_window=int(scfg.get("min_window_minutes", 60)),
    )
    rng = np.random.default_rng(int(scfg.get("seed", 42)))
    negatives, neg_stats = sample_ibm_negatives(
        positives,
        idx,
        rng,
        neg_per_pos=int(scfg.get("neg_per_pos", 20)),
        tries=int(scfg.get("bfs_tries", 10)),
        max_attempt_multiplier=int(scfg.get("max_attempt_multiplier", 20)),
    )

    ibm_feat, ell_feat, sets, feat_stats = build_feature_tables(
        positives, negatives, idx, cc, nodes, edges_cc
    )
    subgraph_stats = {**neg_stats, **feat_stats}
    original_end = {g["sg_id"]: int(g["t1"]) for g in positives + negatives}

    output_dir.mkdir(parents=True, exist_ok=True)
    ibm_feat.to_csv(output_dir / "ibm_features.csv", index=False)
    ell_feat.to_csv(output_dir / "elliptic2_features.csv", index=False)
    with open(output_dir / "subgraph_sets.pkl", "wb") as fh:
        pickle.dump({**sets, "ibm_end": original_end}, fh)

    prep_log = {
        "collection": collection,
        "ibm": ibm_stats,
        "elliptic2": ell_stats,
        "subgraphs": subgraph_stats,
    }
    with open(output_dir / "prep_log.json", "w", encoding="utf-8") as fh:
        json.dump(prep_log, fh, indent=2, default=str)

    save_run_metadata(output_dir, "features", config_dir, extra={"subgraphs": subgraph_stats})
    print(
        f"Saved IBM ({len(ibm_feat):,}) and Elliptic2 ({len(ell_feat):,}) "
        f"feature tables to {output_dir}"
    )
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    config_dir, output_dir, data_root = _resolve_paths(args)
    cfgs = load_all_configs(config_dir)

    ibm_feat = pd.read_csv(output_dir / "ibm_features.csv")
    ell_feat = pd.read_csv(output_dir / "elliptic2_features.csv")
    with open(output_dir / "subgraph_sets.pkl", "rb") as fh:
        sets = pickle.load(fh)

    results_raw = run_legacy_benchmark(ibm_feat, ell_feat, sets, cfgs["benchmark"], output_dir)
    unit4_bundle = run_unit4_tuning(
        ibm_feat, ell_feat, sets, cfgs["benchmark"], cfgs["tuning"], output_dir
    )

    if not args.skip_extensions:
        ibm_dir = data_root / "ibm"
        ibm, accounts, _, _ = load_and_clean_ibm(ibm_dir)
        idx = EdgeIndex(ibm, len(accounts))
        original_end = sets.get("ibm_end")
        if original_end is None:
            scfg = cfgs["sampling"]["ibm_sampling"]
            _, _, acc_id, _ = load_and_clean_ibm(ibm_dir)
            attempts, _ = parse_ibm_patterns(ibm_dir, ibm, acc_id)
            positives = build_ibm_positives(
                attempts, acc_id, idx, int(scfg.get("min_window_minutes", 60))
            )
            rng = np.random.default_rng(int(scfg.get("seed", 42)))
            negatives, _ = sample_ibm_negatives(
                positives, idx, rng, int(scfg.get("neg_per_pos", 20))
            )
            original_end = {g["sg_id"]: int(g["t1"]) for g in positives + negatives}
        run_extensions(
            ibm_feat=ibm_feat,
            ell_feat=ell_feat,
            sets=sets,
            idx=idx,
            original_end=original_end,
            unit4_bundle=unit4_bundle,
            sampling_cfg=cfgs["sampling"],
            benchmark_cfg=cfgs["benchmark"],
            tuning_cfg=cfgs["tuning"],
            ext_cfg=cfgs["extensions"],
            out_dir=output_dir,
        )

    save_run_metadata(
        output_dir,
        "experiments",
        config_dir,
        extra={
            "legacy_rows": len(results_raw),
            "unit5_rows": len(unit4_bundle["evaluation"]),
            "extensions_executed": not args.skip_extensions,
        },
    )
    print(f"Completed experiments; saved CSVs and provenance to {output_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config_dir, output_dir, _ = _resolve_paths(args)
    cfgs = load_all_configs(config_dir)
    generated = generate_all_reports(output_dir, cfgs.get("extensions", {}))
    save_run_metadata(output_dir, "report", config_dir, extra={"figures": generated})
    print(f"Generated {len(generated)} report figures in {output_dir}")
    return 0


def cmd_test(_args: argparse.Namespace) -> int:
    proc = subprocess.run([sys.executable, "-m", "pytest"], check=False)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aml_graph", description="AML Subgraph Structural Analysis CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config-dir", default="configs", help="Directory containing YAML configs")
    common.add_argument(
        "--output-dir", default="outputs", help="Directory for output tables and plots"
    )
    common.add_argument(
        "--data-root", default="data", help="Root directory containing ibm/ and elliptic2/"
    )

    p_data = sub.add_parser(
        "data", parents=[common], help="Download and/or SHA-256 verify dataset files"
    )
    p_data.add_argument(
        "--no-download", action="store_true", help="Fail instead of downloading missing files"
    )
    p_data.set_defaults(func=cmd_data)

    p_feat = sub.add_parser("features", parents=[common], help="Build subgraph feature tables")
    p_feat.set_defaults(func=cmd_features)

    p_exp = sub.add_parser(
        "experiments", parents=[common], help="Run benchmark, tuning, and extensions"
    )
    p_exp.add_argument("--skip-extensions", action="store_true", help="Skip Extensions 1-12")
    p_exp.set_defaults(func=cmd_experiments)

    p_rep = sub.add_parser(
        "report", parents=[common], help="Generate report plots from CSV outputs"
    )
    p_rep.set_defaults(func=cmd_report)

    p_test = sub.add_parser("test", help="Run pytest suite")
    p_test.set_defaults(func=cmd_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
