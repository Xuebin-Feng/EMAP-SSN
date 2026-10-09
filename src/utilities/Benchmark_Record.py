# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Opt-in record of the hardware decisions the tools make on Auto.

Before they pick a device or a plan, six places time the hardware: embedding
generation, the all-vs-all alignment, injection, the database search, the
embedding MSA and the layout. They print what they measured and what they
chose. The benchmark also needs those choices in its report, so while
SSN_BENCHMARK_RECORD names a file, each decision is appended to that file as
one JSON line. Without the variable nothing is recorded, nothing is printed
and nothing changes, so a normal run behaves exactly as before.

Each line holds the decision's kind, a fixed identifier such as
"alignment_plan", the benchmark stage from SSN_BENCHMARK_STAGE when it is
set, and the decision's own fields. For a ranking those are the candidates,
with their device, variant, lanes, measured value, error and peak memory, the
value's unit and direction, the order the candidates were ranked in and the
winner. The fields are data, never display text: the benchmark words its
report itself. Recording never raises: a decision is only ever lost, never
allowed to stop the tool.
"""

from __future__ import annotations

import json
import math
import os
import time

RECORD_VARIABLE = "SSN_BENCHMARK_RECORD"
STAGE_VARIABLE = "SSN_BENCHMARK_STAGE"


def record_path():
    """The file decisions go to, or None when recording is off."""
    return os.environ.get(RECORD_VARIABLE) or None


def _plain(value):
    """``value`` as JSON-ready data: no NaN, numpy scalars as numbers, objects as text."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def record(kind, **fields):
    """Append one decision of ``kind``; return whether it was written."""
    path = record_path()
    if not path:
        return False
    entry = {
        "kind": str(kind),
        "stage": os.environ.get(STAGE_VARIABLE) or None,
        "time": round(time.time(), 3),
    }
    entry.update(fields)
    try:
        line = json.dumps(_plain(entry), ensure_ascii=False, allow_nan=False)
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError):
        return False
    return True


def candidate_fields(candidate):
    """The device fields of a Hardware_Acceleration.DeviceCandidate."""
    return {
        "device": getattr(candidate, "label", None) or str(candidate),
        "spec": getattr(candidate, "spec", None),
        "backend": getattr(candidate, "backend", None),
    }


def record_ranking(
    results,
    ranked,
    *,
    higher_is_better,
    tie_fraction,
    kind=None,
    unit=None,
    **context,
):
    """Record one Hardware_Acceleration.rank_benchmark_results decision."""
    if not record_path():
        return False
    try:
        results = list(results)
        position = {id(result): index for index, result in enumerate(results)}
        ranking = [position[id(result)] for result in ranked if id(result) in position]
        succeeded = [result for result in results if result.succeeded]
        fastest = None
        if succeeded:
            best = (max if higher_is_better else min)(succeeded, key=lambda result: float(result.value))
            fastest = position[id(best)]
        candidates = []
        for result in results:
            plan = getattr(result, "execution_plan", None)
            candidates.append({
                **candidate_fields(result.candidate),
                "variant": result.variant,
                "lanes": result.lanes,
                "value": result.value,
                "error": result.error,
                "peak_memory_bytes": result.peak_memory_bytes,
                "profile": getattr(plan, "profile_name", None),
            })
    except Exception:
        return False
    return record(
        kind or "device_ranking",
        direction="higher" if higher_is_better else "lower",
        unit=unit,
        tie_fraction=tie_fraction,
        candidates=candidates,
        ranking=ranking,
        winner=ranking[0] if ranking else None,
        fastest=fastest,
        **context,
    )


def record_search_plans(plans, ranked, costs, *, notes=(), **context):
    """Record the database search's final plan ranking (Tool_Pipeline.SearchPlan)."""
    if not record_path():
        return False
    try:
        plans = list(plans)
        for plan in ranked:
            if not any(plan is known for known in plans):
                plans.append(plan)
        position = {id(plan): index for index, plan in enumerate(plans)}
        ranking = [position[id(plan)] for plan in ranked]
        candidates = []
        for plan in plans:
            predicted = plan.predicted(costs) if plan.observations else None
            candidates.append({
                **candidate_fields(plan.candidate),
                "variant": plan.variant,
                "precision": plan.precision,
                "lanes": plan.lanes,
                "value": predicted,
                "trials": len(plan.observations),
                "error": None if plan.observations else "not measured",
            })
    except Exception:
        return False
    return record(
        "search_plan",
        direction="lower",
        unit="s",
        candidates=candidates,
        ranking=ranking,
        winner=ranking[0] if ranking else None,
        notes=list(notes),
        **context,
    )


def record_host_cache(store, setting, **context):
    """Record whether an EmbeddingTileStore packed every embedding in RAM or reads tiles."""
    if not record_path():
        return False
    try:
        fields = {
            "setting": setting,
            "choice": "packed" if store.fully_cached else "tiles",
            "limit_bytes": store.host_cache_bytes,
            "embedding_bytes": store.total_source_bytes,
        }
    except Exception:
        return False
    return record("host_cache", **fields, **context)


def read_records(path):
    """Every decision recorded in ``path``, skipping lines a killed write cut short."""
    decisions = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict):
                    decisions.append(entry)
    except OSError:
        return []
    return decisions


__all__ = [
    "RECORD_VARIABLE",
    "STAGE_VARIABLE",
    "record_path",
    "record",
    "candidate_fields",
    "record_ranking",
    "record_search_plans",
    "record_host_cache",
    "read_records",
]
