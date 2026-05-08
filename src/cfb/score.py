"""Score a trajectory file (or in-memory list).

Reports Brier and Brier index in two weightings:

  event-weighted    mean over all events    (current default everywhere)
  source-weighted   mean of per-source means
                    — equalises the contribution of each source so a small
                    market like 'infer' counts the same as the dominant
                    polymarket bucket. Matches BLF Table 17 methodology.
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict


def _read_traj(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def score(rows: list[dict]) -> dict:
    """Compute event- and source-weighted Brier and Brier index from rows
    that contain at least `source`, `p`, `o`."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    sum_b = 0.0
    by_src_n: dict[str, int] = defaultdict(int)
    by_src_b: dict[str, float] = defaultdict(float)
    for r in rows:
        b = (float(r["p"]) - float(r["o"])) ** 2
        sum_b += b
        by_src_n[r["source"]] += 1
        by_src_b[r["source"]] += b
    ev_brier = sum_b / n
    src_means = [by_src_b[s] / by_src_n[s] for s in by_src_n]
    src_brier = sum(src_means) / len(src_means)
    return {
        "n": n,
        "n_sources": len(by_src_n),
        "brier_event": ev_brier,
        "brier_source": src_brier,
        "bi_event": 1.0 - 4.0 * ev_brier,
        "bi_source": 1.0 - 4.0 * src_brier,
        "by_source": {s: {"n": by_src_n[s],
                          "brier": by_src_b[s] / by_src_n[s],
                          "bi": 1.0 - 4.0 * by_src_b[s] / by_src_n[s]}
                       for s in sorted(by_src_n)},
    }


def format_score(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "<empty>"
    lines = [
        f"n={s['n']}  sources={s['n_sources']}",
        f"  event-weighted   Brier={s['brier_event']:.4f}   BI={s['bi_event']:.4f}",
        f"  source-weighted  Brier={s['brier_source']:.4f}   BI={s['bi_source']:.4f}",
        f"  per-source:",
    ]
    for src, d in s["by_source"].items():
        lines.append(f"    {src:12s}  n={d['n']:4d}  Brier={d['brier']:.4f}  BI={d['bi']:.4f}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("traj", help="path to trajectory.jsonl OR an xid (looked up under data/cfb/runs/<xid>/)")
    args = p.parse_args()
    import os
    path = args.traj
    if not os.path.exists(path):
        candidate = os.path.join("data", "cfb", "runs", args.traj, "trajectory.jsonl")
        if os.path.exists(candidate):
            path = candidate
        else:
            raise SystemExit(f"could not find {args.traj} or {candidate}")
    rows = _read_traj(path)
    print(format_score(score(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
