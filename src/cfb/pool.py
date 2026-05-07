"""Pool builder.

Reads per-question JSON from data/questions/{source}/{file_id}.json and emits
a frozen list of (u, f, r, o, text, meta) PoolEntry records, one per
resolution event. Multi-resolution dataset questions are flattened so that
each entry has exactly one (f, r, o) triple.

Filters:
  - asked-day window:  t0 <= f <= t_max
  - censoring:         drop r > t_max  (we will not see the outcome)
  - resolved only:     drop entries with o is None

Frozen output:
  data/cfb/pool-<sha8>.jsonl  — one PoolEntry per line as JSON
  data/cfb/pool-<sha8>.meta.json  — build params + source counts
"""

from __future__ import annotations
import json
import os
import re
import hashlib
import glob
from dataclasses import asdict
from datetime import date
from typing import Iterable

from .types import PoolEntry


_MARKET_SOURCES = ("infer", "manifold", "metaculus", "polymarket")
_DATASET_SOURCES = ("acled", "dbnomics", "fred", "wikipedia", "yfinance")
DEFAULT_SOURCES = _MARKET_SOURCES + _DATASET_SOURCES


def _to_date(s) -> date | None:
    if not s:
        return None
    s = str(s)[:10]
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def _base_id(file_id: str) -> str:
    """Strip a trailing _YYYY-MM-DD suffix from a per-question file id, leaving
    the underlying question identity (used for time-shift duplication stats)."""
    return re.sub(r"_\d{4}-\d{2}-\d{2}$", "", file_id)


def _coerce_binary(v) -> int | None:
    """Outcomes must be exactly 0 or 1. Float crowd-estimates / unresolved
    placeholders are dropped (return None)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f == 0.0:
        return 0
    if f == 1.0:
        return 1
    return None  # fractional => not a real resolution


def _expand_question(q: dict) -> list[tuple[date, date, int]]:
    """Return list of (f, r, o) triples for a question. Multi-resolution
    dataset questions become multiple triples, one per resolved date."""
    f = _to_date(q.get("forecast_due_date"))
    if f is None:
        return []

    rdates = q.get("resolution_dates")
    rto = q.get("resolved_to")

    triples: list[tuple[date, date, int]] = []
    if isinstance(rdates, list) and rdates:
        outs = rto if isinstance(rto, list) else [rto] * len(rdates)
        for r_raw, o_raw in zip(rdates, outs):
            r = _to_date(r_raw)
            o = _coerce_binary(o_raw)
            if r is None or o is None:
                continue
            triples.append((f, r, o))
    else:
        r = _to_date(q.get("resolution_date"))
        o = _coerce_binary(rto)
        if r is None or o is None:
            return []
        triples.append((f, r, o))
    return triples


def build_pool(
    questions_dir: str,
    t0: date,
    t_max: date,
    sources: Iterable[str] = DEFAULT_SOURCES,
    dedupe_base: bool = False,
) -> list[PoolEntry]:
    """Build a frozen pool.

    dedupe_base=True: for each (source, base_id) — i.e. the same underlying
    question text up to a forecast_due_date suffix — keep only the entries from
    the *earliest* forecast_due_date. This drops time-shifted re-asks that share
    text but differ only in reference value / asked-date, while preserving all
    of that one variant's resolution dates.
    """
    entries: list[PoolEntry] = []
    for source in sources:
        src_dir = os.path.join(questions_dir, source)
        if not os.path.isdir(src_dir):
            continue
        for fp in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
            try:
                with open(fp) as fh:
                    q = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            file_id = os.path.basename(fp)[:-5]
            for f, r, o in _expand_question(q):
                if not (t0 <= f and r <= t_max):
                    continue
                # Unique id: source-prefixed file id + resolution date marker.
                # Two resolution events of the same multi-res question get
                # distinct u values via the #r{date} suffix.
                u = f"{source}/{file_id}#r{r.isoformat()}"
                entries.append(PoolEntry(
                    u=u,
                    source=source,
                    f=f,
                    r=r,
                    o=o,
                    text=q.get("question", ""),
                    meta={
                        "id": q.get("id", file_id),
                        "base_id": _base_id(file_id),
                        "source": source,
                        "background": q.get("background", ""),
                        "resolution_criteria": q.get("resolution_criteria", ""),
                        "url": q.get("url", ""),
                        "market_value": q.get("market_value"),
                        "market_value_explanation": q.get("market_value_explanation", ""),
                        "market_date": q.get("market_date", ""),
                    },
                ))
    if dedupe_base:
        # earliest f per (source, base_id)
        earliest: dict[tuple[str, str], date] = {}
        for e in entries:
            k = (e.source, e.meta["base_id"])
            if k not in earliest or e.f < earliest[k]:
                earliest[k] = e.f
        entries = [e for e in entries
                   if e.f == earliest[(e.source, e.meta["base_id"])]]

    entries.sort(key=lambda e: (e.f, e.r, e.u))
    return entries


# --- frozen-pool I/O ----------------------------------------------------------

def _entry_to_jsonable(e: PoolEntry) -> dict:
    d = asdict(e)
    d["f"] = e.f.isoformat()
    d["r"] = e.r.isoformat()
    return d


def _entry_from_json(d: dict) -> PoolEntry:
    return PoolEntry(
        u=d["u"], source=d["source"],
        f=_to_date(d["f"]), r=_to_date(d["r"]),
        o=int(d["o"]), text=d.get("text", ""), meta=d.get("meta", {}),
    )


def freeze(entries: list[PoolEntry], out_dir: str,
           build_params: dict) -> tuple[str, str]:
    """Write entries to data/cfb/pool-<sha>.jsonl + .meta.json. Hash is over
    the canonical JSONL bytes so two builds with the same data produce the
    same filename."""
    os.makedirs(out_dir, exist_ok=True)
    payload = "\n".join(
        json.dumps(_entry_to_jsonable(e), sort_keys=True) for e in entries
    ).encode()
    sha = hashlib.sha256(payload).hexdigest()[:8]
    pool_path = os.path.join(out_dir, f"pool-{sha}.jsonl")
    meta_path = os.path.join(out_dir, f"pool-{sha}.meta.json")
    with open(pool_path, "wb") as fh:
        fh.write(payload)
    with open(meta_path, "w") as fh:
        json.dump({**build_params, "n_entries": len(entries), "sha": sha},
                  fh, indent=2, default=str)
    return pool_path, meta_path


def load_pool(pool_path: str) -> list[PoolEntry]:
    out: list[PoolEntry] = []
    with open(pool_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(_entry_from_json(json.loads(line)))
    return out
