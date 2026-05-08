"""Apply post-processing layers (shrink, platt) to a cached run trajectory.

Reads:  data/cfb/runs/<in_xid>/trajectory.jsonl
Writes: data/cfb/runs/<out_xid>/trajectory.jsonl  + manifest.json

Layers are applied in **strict-online** order: when scoring event i (with
forecast date f_i), the layer's state is fitted only on events resolved
before f_i — never peeking at the future.

Most users won't invoke this CLI directly — `run.py` chains base + layers
from a string config. This module is exposed in case you want to apply a
layer ad-hoc to an existing run.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import date, datetime

from .agents.online_shrinker import OnlineShrinker
from .agents.online_platt import OnlinePlatt
from .score import _read_traj, score, format_score


def _to_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def _walk_chronological(rows: list[dict],
                        on_event,  # (row) -> p_post
                        on_observe,  # (row) -> None, called when r_j < f_i becomes true
                        ):
    """Walk rows in chronological resolution order, deferring `on_observe`
    until the row's r is strictly before some future event's f.

    Concretely: for each event i (resolution at r_i, forecast at f_i), call
    on_event AFTER promoting any pending past events with r_j < f_i to
    observed state via on_observe. Returns p_post for each event.
    """
    rows_sorted = sorted(rows, key=lambda r: (r["r"], r["i"]))
    pending: list[dict] = []
    out: list[dict] = []
    for row in rows_sorted:
        f_i = row["f"]
        # Promote all pending events with r_j < f_i into observed state
        keep = []
        for ev in pending:
            if ev["r"] < f_i:
                on_observe(ev)
            else:
                keep.append(ev)
        pending = keep

        # Predict for this event with the layer's current state
        p_post = on_event(row)
        out_row = {**row, "p_pre": row["p"], "p": p_post}
        out.append(out_row)
        pending.append(row)
    # remaining pending events were never used as history for any future event
    return out


def apply_shrink(rows: list[dict], ridge: float = 1.0) -> list[dict]:
    sh = OnlineShrinker(ridge=ridge)
    def on_event(row):
        return sh.predict(row["source"], float(row["p"]), row.get("p_crowd"))
    def on_observe(row):
        sh.update(row["source"], float(row["p"]), row.get("p_crowd"),
                  float(row["o"]))
    return _walk_chronological(rows, on_event, on_observe)


def apply_platt(rows: list[dict], ridge: float = 1.0,
                lr: float = 0.5) -> list[dict]:
    pl = OnlinePlatt(ridge=ridge, lr=lr)
    def on_event(row):
        return pl.predict(float(row["p"]))
    def on_observe(row):
        pl.update(float(row["p"]), float(row["o"]))
    return _walk_chronological(rows, on_event, on_observe)


_LAYERS = {
    "shrink": apply_shrink,
    "platt":  apply_platt,
}


def post(in_xid: str, out_xid: str, layer: str,
         ridge: float = 1.0, lr: float = 0.5,
         runs_dir: str | None = None) -> dict:
    """Apply a single post-processing layer to a cached run, in-process.
    Returns the score dict for the new run."""
    runs_dir = runs_dir or os.path.join(
        os.path.expanduser("~/BLF"), "data", "cfb", "runs")
    in_path = os.path.join(runs_dir, in_xid, "trajectory.jsonl")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"missing input trajectory: {in_path}")
    rows = _read_traj(in_path)

    if layer == "shrink":
        out_rows = apply_shrink(rows, ridge=ridge)
    elif layer == "platt":
        out_rows = apply_platt(rows, ridge=ridge, lr=lr)
    else:
        raise ValueError(f"unknown layer {layer}")

    out_dir = os.path.join(runs_dir, out_xid)
    os.makedirs(out_dir, exist_ok=True)
    out_traj = os.path.join(out_dir, "trajectory.jsonl")
    with open(out_traj, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({
            "xid": out_xid,
            "parent_xid": in_xid,
            "layer": layer,
            "ridge": ridge,
            "lr": lr if layer == "platt" else None,
            "n_events": len(out_rows),
        }, fh, indent=2, default=str)
    return score(out_rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_xid",  required=True)
    p.add_argument("--out", dest="out_xid", required=True)
    p.add_argument("--layer", required=True, choices=sorted(_LAYERS))
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--runs-dir", default=None)
    args = p.parse_args()

    s = post(in_xid=args.in_xid, out_xid=args.out_xid, layer=args.layer,
             ridge=args.ridge, lr=args.lr, runs_dir=args.runs_dir)
    print(format_score(s))
    print(f"\n[post] wrote data/cfb/runs/{args.out_xid}/trajectory.jsonl",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
