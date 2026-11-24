"""Dataset acquisition, pinned revision verification, SHA-256 checks, and raw loading."""

from __future__ import annotations

import hashlib
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IBM_COLS: list[str] = [
    "timestamp",
    "from_bank",
    "from_acct",
    "to_bank",
    "to_acct",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

IBM_TEXT_COLS: list[str] = [
    "from_bank",
    "from_acct",
    "to_bank",
    "to_acct",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]

CC_ID, CC_LAB = "ccId", "ccLabel"
N_ID, N_CC = "clId", "ccId"
E_SRC, E_DST = "clId1", "clId2"


class ChecksumMismatchError(RuntimeError):
    """Raised when a dataset file fails SHA-256 or byte-size verification."""


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_checksum(
    path: Path,
    expected_sha256: str,
    expected_size: int | None = None,
) -> str:
    """Verify SHA-256 (and optional byte length) of path; fail loudly on mismatch."""
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file does not exist: {path}")
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != int(expected_size):
        raise ChecksumMismatchError(
            f"Size mismatch for {path.name}: expected {expected_size}, got {actual_size}"
        )
    actual_sha = compute_sha256(path)
    if actual_sha.lower() != expected_sha256.lower():
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual_sha}"
        )
    return actual_sha


def ensure_sources(
    data_cfg: dict[str, Any],
    data_root: Path,
    download_missing: bool = True,
) -> dict[str, Any]:
    """Ensure all configured source files exist at pinned revisions and pass SHA-256 checks."""
    template: str = data_cfg["url_template"]
    sources: dict[str, dict[str, Any]] = data_cfg["sources"]
    collection_log: dict[str, Any] = {"date": time.strftime("%Y-%m-%d")}

    for name, spec in sources.items():
        subdir = data_root / str(spec["subdir"])
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / name
        url = template.format(repo=spec["repo"], revision=spec["revision"], name=name)

        if path.exists():
            how = "already_present"
        elif download_missing:
            t0 = time.time()
            urllib.request.urlretrieve(url, path)
            how = f"downloaded_in_{time.time() - t0:.1f}s"
        else:
            raise FileNotFoundError(f"Missing dataset file {path} and download_missing=False")

        expected_sha = spec.get("sha256")
        expected_size = spec.get("size_bytes")
        actual_sha = (
            verify_file_checksum(
                path, str(expected_sha), int(expected_size) if expected_size is not None else None
            )
            if expected_sha
            else compute_sha256(path)
        )
        collection_log[name] = {
            "url": url,
            "revision": spec["revision"],
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha,
            "status": how,
        }
    return collection_log


def norm_bank(series: pd.Series) -> pd.Series:
    """Normalize bank identifiers by stripping whitespace and leading zeros."""
    s = series.astype(str).str.strip().str.lstrip("0")
    return s.where(s != "", "0")


def account_key(bank: pd.Series, acct: pd.Series) -> pd.Series:
    """Construct globally unique bank_account string keys."""
    return norm_bank(bank) + "_" + acct.astype(str).str.strip().str.upper()


def load_and_clean_ibm(ibm_dir: Path) -> tuple[pd.DataFrame, pd.Index, pd.Series, dict[str, Any]]:
    """Load HI-Small_Trans.csv, clean timestamps, and assign integer account IDs."""
    ibm = pd.read_csv(
        ibm_dir / "HI-Small_Trans.csv",
        header=0,
        names=IBM_COLS,
        dtype={c: str for c in IBM_TEXT_COLS},
    )
    stats: dict[str, Any] = {
        "raw_rows": len(ibm),
        "null_values_per_column": {k: int(v) for k, v in ibm.isna().sum().items() if v},
        "laundering_rows": int(ibm["is_laundering"].sum()),
        "laundering_rate_1_in": round(len(ibm) / max(1, int(ibm["is_laundering"].sum())), 1),
        "exact_duplicate_rows": int(ibm.duplicated().sum()),
        "cross_currency_rows": int((ibm["receiving_currency"] != ibm["payment_currency"]).sum()),
    }

    before = len(ibm)
    ibm = ibm.dropna(
        subset=["timestamp", "from_bank", "from_acct", "to_bank", "to_acct", "is_laundering"]
    )
    ibm["timestamp"] = pd.to_datetime(ibm["timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    ibm = ibm.dropna(subset=["timestamp"]).reset_index(drop=True)
    ibm["is_laundering"] = ibm["is_laundering"].astype(int)
    stats["rows_dropped_missing_or_bad_date"] = before - len(ibm)

    ibm["src"] = account_key(ibm["from_bank"], ibm["from_acct"])
    ibm["dst"] = account_key(ibm["to_bank"], ibm["to_acct"])
    codes, accounts = pd.factorize(pd.concat([ibm["src"], ibm["dst"]]))
    ibm["s"] = codes[: len(ibm)]
    ibm["d"] = codes[len(ibm) :]
    acc_id = pd.Series(np.arange(len(accounts)), index=accounts)

    pairs = ibm.groupby(["s", "d"]).size()
    stats.update(
        unique_accounts=len(accounts),
        self_loop_rows=int((ibm["s"] == ibm["d"]).sum()),
        time_range=[str(ibm["timestamp"].min()), str(ibm["timestamp"].max())],
        distinct_directed_account_pairs=len(pairs),
        share_of_pairs_with_repeat_transactions=round(float((pairs > 1).mean()), 4)
        if len(pairs)
        else 0.0,
    )
    return ibm, accounts, acc_id, stats


def parse_ibm_patterns(
    ibm_dir: Path,
    ibm: pd.DataFrame,
    acc_id: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse HI-Small_Patterns.txt into documented laundering attempts and audit coverage."""
    attempts: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    patterns_path = ibm_dir / "HI-Small_Patterns.txt"
    with open(patterns_path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                m = re.search(r"ATTEMPT\s*-\s*([A-Za-z\- ]+?)(?::|$)", line)
                cur = {"pattern": m.group(1).strip().upper() if m else "UNKNOWN", "rows": []}
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if cur and cur["rows"]:
                    attempts.append(cur)
                cur = None
            elif cur is not None and line:
                cur["rows"].append([p.strip() for p in line.split(",")][:11])

    stats: dict[str, Any] = {
        "laundering_attempts": len(attempts),
        "attempts_by_pattern": pd.Series([a["pattern"] for a in attempts], dtype="object")
        .value_counts()
        .to_dict(),
    }

    if attempts:
        pat = pd.DataFrame([r for a in attempts for r in a["rows"]], columns=IBM_COLS)
        pat["timestamp"] = pd.to_datetime(
            pat["timestamp"], format="%Y/%m/%d %H:%M", errors="coerce"
        )
        pat["src"] = account_key(pat["from_bank"], pat["from_acct"])
        pat["dst"] = account_key(pat["to_bank"], pat["to_acct"])
        key = ["timestamp", "src", "dst"]
        laund = ibm.loc[ibm["is_laundering"] == 1, key]
        if len(laund):
            merged = laund.merge(pat[key].drop_duplicates(), on=key, how="left", indicator=True)
            stats["pattern_coverage_of_laundering_rows"] = round(
                float((merged["_merge"] == "both").mean()), 4
            )
        else:
            stats["pattern_coverage_of_laundering_rows"] = 0.0
        stats["pattern_accounts_missing_from_transactions"] = int(
            (~pd.concat([pat["src"], pat["dst"]]).isin(acc_id.index)).sum()
        )
    else:
        stats["pattern_coverage_of_laundering_rows"] = 0.0
        stats["pattern_accounts_missing_from_transactions"] = 0

    return attempts, stats


def load_and_prepare_elliptic2(
    ell_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load Elliptic2 component, node, and edge CSVs and assign within-component edges."""
    cc = pd.read_csv(ell_dir / "connected_components.csv")
    nodes = pd.read_csv(ell_dir / "nodes.csv")
    edges = pd.read_csv(ell_dir / "edges.csv")

    cc["label"] = cc[CC_LAB].astype(str).str.strip().str.lower().map({"suspicious": 1, "licit": 0})
    unknown_dropped = int(cc["label"].isna().sum())
    cc = cc.dropna(subset=["label"]).copy()

    raw_node_rows = len(nodes)
    raw_edge_rows = len(edges)
    dup_nodes = int(nodes.duplicated([N_ID, N_CC]).sum())
    nodes = nodes.drop_duplicates([N_ID, N_CC]).copy()
    shared = nodes.groupby(N_ID)[N_CC].nunique()

    a = nodes[[N_ID, N_CC]].rename(columns={N_ID: E_SRC, N_CC: "cc"})
    b = nodes[[N_ID, N_CC]].rename(columns={N_ID: E_DST, N_CC: "cc2"})
    edges_cc = edges.merge(a, on=E_SRC).merge(b, on=E_DST)
    edges_cc = edges_cc[edges_cc["cc"] == edges_cc["cc2"]].drop(columns="cc2")

    stats: dict[str, Any] = {
        "labeled_subgraphs": len(cc),
        "licit": int((cc["label"] == 0).sum()),
        "suspicious": int((cc["label"] == 1).sum()),
        "unknown_labels_dropped": unknown_dropped,
        "node_rows": raw_node_rows,
        "edge_rows": raw_edge_rows,
        "duplicate_node_rows": dup_nodes,
        "clusters_in_more_than_one_subgraph": int((shared > 1).sum()),
        "max_subgraphs_sharing_one_cluster": int(shared.max()) if len(shared) else 0,
        "edges_assigned_to_a_subgraph": len(edges_cc),
        "edges_in_more_than_one_subgraph": int(
            (edges_cc.groupby([E_SRC, E_DST, "txId"])["cc"].nunique() > 1).sum()
        )
        if len(edges_cc)
        else 0,
        "original_edges_not_assigned": len(edges)
        - edges_cc.drop_duplicates([E_SRC, E_DST, "txId"]).shape[0],
        "self_loop_edges": int((edges_cc[E_SRC] == edges_cc[E_DST]).sum()),
        "parallel_edges_collapsed": int(edges_cc.duplicated(["cc", E_SRC, E_DST]).sum()),
    }
    return cc, nodes, edges_cc, stats
