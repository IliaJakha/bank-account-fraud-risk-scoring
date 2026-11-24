"""Tests for SHA-256 verification and strict no-label-use guarantees in samplers/features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_graph.data import ChecksumMismatchError, compute_sha256, verify_file_checksum
from aml_graph.features import transaction_activity_features
from aml_graph.sampling import (
    EdgeIndex,
    LabelAccessForbiddenError,
    common_bfs_candidate,
)


def _synthetic_transactions() -> pd.DataFrame:
    base_ts = pd.Timestamp("2022-09-01 00:00:00")
    rows = []
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (0, 2), (2, 4)]
    for k, (s, d) in enumerate(edges):
        rows.append(
            {
                "s": s,
                "d": d,
                "is_laundering": int(k % 2 == 0),
                "timestamp": base_ts + pd.Timedelta(minutes=k * 10),
            }
        )
    return pd.DataFrame(rows)


def test_verify_file_checksum_fails_loudly_on_corruption_or_size_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "sample.csv"
    p.write_bytes(b"id,val\n1,100\n")
    expected_sha = compute_sha256(p)
    expected_size = p.stat().st_size

    assert verify_file_checksum(p, expected_sha, expected_size) == expected_sha

    with pytest.raises(ChecksumMismatchError, match="Size mismatch"):
        verify_file_checksum(p, expected_sha, expected_size + 1)

    p.write_bytes(b"id,val\n1,101\n")  # same byte length, 1-byte flip
    with pytest.raises(ChecksumMismatchError, match="SHA-256 mismatch"):
        verify_file_checksum(p, expected_sha, expected_size)


def test_label_blind_view_blocks_label_access() -> None:
    df = _synthetic_transactions()
    idx = EdgeIndex(df, n_accounts=6)
    blind = idx.label_blind_view()
    with pytest.raises(LabelAccessForbiddenError):
        _ = blind.y


def test_common_bfs_candidate_never_reads_labels() -> None:
    df_orig = _synthetic_transactions()
    df_all_zero = df_orig.assign(is_laundering=0)
    df_all_one = df_orig.assign(is_laundering=1)
    df_inverted = df_orig.assign(is_laundering=1 - df_orig["is_laundering"])

    idx_orig = EdgeIndex(df_orig, n_accounts=6)
    idx_zero = EdgeIndex(df_all_zero, n_accounts=6)
    idx_one = EdgeIndex(df_all_one, n_accounts=6)
    idx_inv = EdgeIndex(df_inverted, n_accounts=6)

    # Also verify running directly on a label-forbidden view succeeds without error
    g_blind = common_bfs_candidate(idx_orig.label_blind_view(), 2, 4, 120, np.random.default_rng(7))
    g_zero = common_bfs_candidate(idx_zero, 2, 4, 120, np.random.default_rng(7))
    g_one = common_bfs_candidate(idx_one, 2, 4, 120, np.random.default_rng(7))
    g_inv = common_bfs_candidate(idx_inv, 2, 4, 120, np.random.default_rng(7))

    assert g_blind is not None and g_zero is not None and g_one is not None and g_inv is not None
    for other in (g_zero, g_one, g_inv):
        assert g_blind["nodes"] == other["nodes"]
        assert np.array_equal(g_blind["edges"], other["edges"])
        assert g_blind["t0"] == other["t0"] and g_blind["t1"] == other["t1"]


def test_transaction_activity_features_never_reads_labels() -> None:
    df_a = _synthetic_transactions()
    df_b = df_a.assign(is_laundering=1 - df_a["is_laundering"])
    idx_a = EdgeIndex(df_a, n_accounts=6)
    idx_b = EdgeIndex(df_b, n_accounts=6)

    edges_map = {"sg_0": np.array([0, 1, 2, 6], dtype=np.int64)}
    starts = {"sg_0": int(idx_a.t[0])}
    ends = {"sg_0": int(idx_a.t[6]) + 60}

    feat_a = transaction_activity_features(
        "sg_0", edges_map, idx_a.label_blind_view(), starts, ends
    )
    feat_b = transaction_activity_features("sg_0", edges_map, idx_b, starts, ends)
    assert feat_a == feat_b
