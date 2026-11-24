"""Failing-by-design test documenting the legacy IBM positive vs negative window leak."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aml_graph.data import IBM_COLS
from aml_graph.sampling import EdgeIndex, build_ibm_positives, sample_ibm_negatives


def test_ibm_positive_and_negative_windows_follow_same_rule() -> None:
    """Verify positive and negative IBM candidate windows follow the same placement rule.

    In Phase 1 (legacy notebook logic):
      - `build_ibm_positives` sets `t0 = min(attempt_tx_time)` and
        `t1 = max(max(attempt_tx_time), t0 + 60)`. Thus every multi-hour positive has
        zero leading idle margin (`first_tx - t0 == 0`) and `active_span == t1 - t0`.
      - `sample_ibm_negatives` draws `t0` uniformly across `[t_min, t_max - dur]`,
        yielding nonzero idle padding inside `[t0, t1]`.
    This asymmetry leaks window construction into `active_span_minutes` and
    `transactions_per_hour`. Phase 2 replaces both with one label-blind window rule.
    """
    base = pd.Timestamp("2022-09-01 00:00:00")
    tx_rows: list[list[object]] = []
    attempts: list[dict[str, object]] = []

    # 6 multi-hour laundering attempts (each spanning 180 minutes > min_window=60)
    for k in range(6):
        day_offset = k * 1440
        t_first = (base + pd.Timedelta(minutes=day_offset + 120)).strftime("%Y/%m/%d %H:%M")
        t_mid = (base + pd.Timedelta(minutes=day_offset + 210)).strftime("%Y/%m/%d %H:%M")
        t_last = (base + pd.Timedelta(minutes=day_offset + 300)).strftime("%Y/%m/%d %H:%M")
        a, b, c = f"P{k}A", f"P{k}B", f"P{k}C"
        r1 = [t_first, "1", a, "1", b, 500.0, "US Dollar", 500.0, "US Dollar", "ACH", 1]
        r2 = [t_mid, "1", b, "1", c, 500.0, "US Dollar", 500.0, "US Dollar", "ACH", 1]
        r3 = [t_last, "1", c, "1", a, 500.0, "US Dollar", 500.0, "US Dollar", "ACH", 1]
        tx_rows.extend([r1, r2, r3])
        attempts.append({"pattern": "CYCLE", "rows": [r1, r2, r3]})

    # Dense licit background bursts (each burst lasts only 20 minutes inside a multi-day span)
    for d in range(12):
        for burst in range(6):
            burst_start = d * 720 + burst * 110
            t1_s = (base + pd.Timedelta(minutes=burst_start)).strftime("%Y/%m/%d %H:%M")
            t2_s = (base + pd.Timedelta(minutes=burst_start + 20)).strftime("%Y/%m/%d %H:%M")
            u, v, w = f"L{d}_{burst}A", f"L{d}_{burst}B", f"L{d}_{burst}C"
            tx_rows.append(
                [t1_s, "2", u, "2", v, 100.0, "US Dollar", 100.0, "US Dollar", "Cheque", 0]
            )
            tx_rows.append(
                [t2_s, "2", v, "2", w, 100.0, "US Dollar", 100.0, "US Dollar", "Cheque", 0]
            )

    ibm = pd.DataFrame(tx_rows, columns=IBM_COLS)
    ibm["timestamp"] = pd.to_datetime(ibm["timestamp"], format="%Y/%m/%d %H:%M")
    ibm["is_laundering"] = ibm["is_laundering"].astype(int)
    ibm["src"] = ibm["from_bank"] + "_" + ibm["from_acct"]
    ibm["dst"] = ibm["to_bank"] + "_" + ibm["to_acct"]
    codes, accounts = pd.factorize(pd.concat([ibm["src"], ibm["dst"]]))
    ibm["s"] = codes[: len(ibm)]
    ibm["d"] = codes[len(ibm) :]
    acc_id = pd.Series(np.arange(len(accounts)), index=accounts)

    idx = EdgeIndex(ibm, len(accounts))
    positives = build_ibm_positives(attempts, acc_id, idx, min_window=60)
    negatives, _ = sample_ibm_negatives(positives, idx, np.random.default_rng(42), neg_per_pos=5)
    assert len(positives) == 6 and len(negatives) > 0

    def span_ratio(g: dict[str, object]) -> float:
        e = np.asarray(g["edges"], dtype=np.int64)
        times = idx.t[e]
        dur = max(1, int(g["t1"]) - int(g["t0"]))
        return float((times.max() - times.min()) / dur)

    pos_ratios = [span_ratio(g) for g in positives]
    neg_ratios = [span_ratio(g) for g in negatives]

    # Under a symmetric, label-blind window rule, positives must NOT deterministically
    # clamp both endpoints to 100% of the window while negatives have < 50% active span.
    assert abs(float(np.mean(pos_ratios)) - float(np.mean(neg_ratios))) < 0.25
