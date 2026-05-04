#!/usr/bin/env python3
"""Load-aware SVD scheduling algorithms from algorithm.pdf chapter 5.

The module intentionally keeps the scheduler independent from llama.cpp.  It
consumes a small JSON layer profile, solves the local DP problem, optionally
searches a suffix offload split point, and writes the per-layer SVD rate file
that `decode_svd_test` already accepts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


INF = float("inf")


@dataclass(frozen=True)
class Candidate:
    rate: float
    main_ms: float
    loss: float
    weight: float
    tail_ms: float = 0.0

    @property
    def clipped(self) -> int:
        return 1 if self.rate > 0.0 else 0


@dataclass(frozen=True)
class LayerProfile:
    layer: int
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class CoreSplitProfile:
    split_id: str
    p: int
    major_cpus: tuple[int, ...]
    minor_cpus: tuple[int, ...]
    layers: tuple[LayerProfile, ...]


@dataclass(frozen=True)
class OffloadCandidate:
    """End-to-end layer offload candidate.

    `m` follows the exp6 layer-coop runtime convention:

      PC    = [0, m)
      Phone = [m, n_layers)

    This is intentionally different from the older internal `split_m` field,
    which represented the last PC layer index.
    """

    m: int
    total_ms: float
    pc_ms: float = 0.0
    phone_ms: float = 0.0
    network_ms: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class MajorOnlyCandidate:
    p: int
    major_cpus: tuple[int, ...]
    minor_cpus: tuple[int, ...]
    total_ms: float
    source: str = ""


@dataclass
class LayerDecision:
    layer: int
    rate: float
    main_ms: float
    loss: float
    weight: float
    tail_ms: float = 0.0
    timeout_ms: float = 0.0


@dataclass
class LocalScheduleResult:
    feasible: bool
    deadline_ms: float
    split_id: str | None = None
    p: int | None = None
    major_cpus: list[int] | None = None
    minor_cpus: list[int] | None = None
    total_main_ms: float = INF
    total_loss: float = INF
    decisions: list[LayerDecision] | None = None
    reason: str = ""

    def rates(self, n_layers: int | None = None) -> list[float]:
        if not self.decisions:
            return []
        if n_layers is None:
            n_layers = max(d.layer for d in self.decisions) + 1
        rates = [0.0] * n_layers
        for decision in self.decisions:
            rates[decision.layer] = decision.rate
        return rates

    def timeouts(self, n_layers: int | None = None) -> list[float]:
        if not self.decisions:
            return []
        if n_layers is None:
            n_layers = max(d.layer for d in self.decisions) + 1
        timeouts = [0.0] * n_layers
        for decision in self.decisions:
            timeouts[decision.layer] = decision.timeout_ms
        return timeouts


@dataclass
class JointScheduleResult:
    mode: str
    feasible: bool
    split_m: int | None
    local: LocalScheduleResult | None
    offload_m: int | None = None
    tx_ms: float = 0.0
    end_ms: float = 0.0
    network_ms: float = 0.0
    total_ms: float = INF
    offloaded_layers: list[int] | None = None
    reason: str = ""


def _time_bucket(ms: float, quantum_ms: float) -> int:
    return int(math.ceil(ms / quantum_ms - 1e-12))


def _bucket_ms(bucket: int, quantum_ms: float) -> float:
    return bucket * quantum_ms


def load_profile(path: Path) -> tuple[list[LayerProfile], dict[str, Any]]:
    data = json.loads(path.read_text())
    if "core_splits" in data:
        splits, meta = load_core_split_profile(path)
        if not splits:
            return [], meta
        # Backward-compatible view: expose the first split to old callers.
        return list(splits[0].layers), meta
    layers = []
    for layer_obj in data["layers"]:
        candidates = tuple(
            Candidate(
                rate=float(c["rate"]),
                main_ms=float(c["main_ms"]),
                loss=float(c.get("loss", 0.0)),
                weight=float(c.get("weight", c["rate"])),
                tail_ms=float(c.get("tail_ms", 0.0)),
            )
            for c in layer_obj["candidates"]
        )
        layers.append(LayerProfile(layer=int(layer_obj["layer"]), candidates=candidates))
    layers.sort(key=lambda item: item.layer)
    meta = {k: v for k, v in data.items() if k != "layers"}
    return layers, meta


def _parse_cpus(value: Any) -> tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        cpus: list[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                cpus.extend(range(int(lo_s), int(hi_s) + 1))
            else:
                cpus.append(int(part))
        return tuple(sorted(set(cpus)))
    return tuple(int(x) for x in value)


def _load_layers(layer_objs: Iterable[dict[str, Any]]) -> tuple[LayerProfile, ...]:
    layers = []
    for layer_obj in layer_objs:
        candidates = tuple(
            Candidate(
                rate=float(c["rate"]),
                main_ms=float(c["main_ms"]),
                loss=float(c.get("loss", 0.0)),
                weight=float(c.get("weight", c["rate"])),
                tail_ms=float(c.get("tail_ms", 0.0)),
            )
            for c in layer_obj["candidates"]
        )
        layers.append(LayerProfile(layer=int(layer_obj["layer"]), candidates=candidates))
    return tuple(sorted(layers, key=lambda item: item.layer))


def load_core_split_profile(path: Path) -> tuple[list[CoreSplitProfile], dict[str, Any]]:
    data = json.loads(path.read_text())
    if "core_splits" not in data:
        layers, meta = load_profile(path)
        return [
            CoreSplitProfile(
                split_id="legacy",
                p=int(meta.get("p", 0)),
                major_cpus=_parse_cpus(meta.get("major_cpus")),
                minor_cpus=_parse_cpus(meta.get("minor_cpus")),
                layers=tuple(layers),
            )
        ], meta

    splits = []
    for index, split_obj in enumerate(data["core_splits"]):
        major_cpus = _parse_cpus(split_obj.get("major_cpus"))
        minor_cpus = _parse_cpus(split_obj.get("minor_cpus"))
        split_id = str(split_obj.get("split_id") or f"p{len(major_cpus)}_{index}")
        splits.append(
            CoreSplitProfile(
                split_id=split_id,
                p=int(split_obj.get("p", len(major_cpus))),
                major_cpus=major_cpus,
                minor_cpus=minor_cpus,
                layers=_load_layers(split_obj["layers"]),
            )
        )
    meta = {k: v for k, v in data.items() if k != "core_splits"}
    return splits, meta


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def solve_local_dp(
    layers: Iterable[LayerProfile],
    deadline_ms: float,
    timeout_budget_ms: float = 0.0,
    quantum_ms: float = 1.0,
    split: CoreSplitProfile | None = None,
) -> LocalScheduleResult:
    """Solve chapter 5.2 DP with adjacent-clipped-layer exclusion."""

    layer_list = list(layers)
    if not layer_list:
        return LocalScheduleResult(
            feasible=True,
            deadline_ms=deadline_ms,
            split_id=split.split_id if split else None,
            p=split.p if split else None,
            major_cpus=list(split.major_cpus) if split else None,
            minor_cpus=list(split.minor_cpus) if split else None,
            total_main_ms=0.0,
            total_loss=0.0,
            decisions=[],
        )

    max_bucket = _time_bucket(deadline_ms, quantum_ms)
    # state: (time_bucket, current_clipped) -> (loss, prev_key, candidate)
    states: dict[tuple[int, int], tuple[float, tuple[int, int] | None, Candidate | None]] = {
        (0, 0): (0.0, None, None)
    }
    parents: list[dict[tuple[int, int], tuple[tuple[int, int], Candidate]]] = []

    for layer in layer_list:
        next_states: dict[tuple[int, int], tuple[float, tuple[int, int] | None, Candidate | None]] = {}
        layer_parent: dict[tuple[int, int], tuple[tuple[int, int], Candidate]] = {}
        for prev_key, (prev_loss, _, _) in states.items():
            prev_bucket, prev_clipped = prev_key
            for candidate in layer.candidates:
                clipped = candidate.clipped
                if prev_clipped + clipped > 1:
                    continue
                bucket = prev_bucket + _time_bucket(candidate.main_ms, quantum_ms)
                if bucket > max_bucket:
                    continue
                key = (bucket, clipped)
                loss = prev_loss + candidate.loss
                old = next_states.get(key)
                if old is None or loss < old[0] - 1e-12:
                    next_states[key] = (loss, prev_key, candidate)
                    layer_parent[key] = (prev_key, candidate)
        parents.append(layer_parent)
        states = next_states
        if not states:
            return LocalScheduleResult(
                feasible=False,
                deadline_ms=deadline_ms,
                split_id=split.split_id if split else None,
                p=split.p if split else None,
                major_cpus=list(split.major_cpus) if split else None,
                minor_cpus=list(split.minor_cpus) if split else None,
                reason=f"no state survives after layer {layer.layer}",
            )

    best_key: tuple[int, int] | None = None
    best_loss = INF
    best_bucket = math.inf
    for key, (loss, _, _) in states.items():
        bucket, _ = key
        if loss < best_loss - 1e-12 or (abs(loss - best_loss) <= 1e-12 and bucket < best_bucket):
            best_key = key
            best_loss = loss
            best_bucket = bucket

    if best_key is None:
        return LocalScheduleResult(
            feasible=False,
            deadline_ms=deadline_ms,
            split_id=split.split_id if split else None,
            p=split.p if split else None,
            major_cpus=list(split.major_cpus) if split else None,
            minor_cpus=list(split.minor_cpus) if split else None,
            reason="no terminal state",
        )

    decisions_rev: list[LayerDecision] = []
    key = best_key
    for layer, layer_parent in zip(reversed(layer_list), reversed(parents)):
        prev_key, candidate = layer_parent[key]
        decisions_rev.append(
            LayerDecision(
                layer=layer.layer,
                rate=candidate.rate,
                main_ms=candidate.main_ms,
                loss=candidate.loss,
                weight=candidate.weight,
                tail_ms=candidate.tail_ms,
            )
        )
        key = prev_key
    decisions = list(reversed(decisions_rev))
    allocate_timeouts(decisions, timeout_budget_ms)
    return LocalScheduleResult(
        feasible=True,
        deadline_ms=deadline_ms,
        split_id=split.split_id if split else None,
        p=split.p if split else None,
        major_cpus=list(split.major_cpus) if split else None,
        minor_cpus=list(split.minor_cpus) if split else None,
        total_main_ms=sum(item.main_ms for item in decisions),
        total_loss=sum(item.loss for item in decisions),
        decisions=decisions,
    )


def allocate_timeouts(decisions: list[LayerDecision], timeout_budget_ms: float) -> None:
    total_weight = sum(max(0.0, item.weight) for item in decisions)
    if timeout_budget_ms <= 0.0 or total_weight <= 0.0:
        for item in decisions:
            item.timeout_ms = 0.0
        return
    for item in decisions:
        item.timeout_ms = timeout_budget_ms * max(0.0, item.weight) / total_weight


def _metric_series(meta: dict[str, Any], name: str, n_layers: int) -> list[float]:
    value = meta.get(name)
    if value is None:
        return [0.0] * (n_layers + 1)
    if isinstance(value, (int, float)):
        return [float(value)] * (n_layers + 1)
    series = [float(x) for x in value]
    if len(series) < n_layers + 1:
        series.extend([series[-1] if series else 0.0] * (n_layers + 1 - len(series)))
    return series


def _load_offload_candidates(meta: dict[str, Any], n_layers: int) -> list[OffloadCandidate]:
    """Read real or modeled layer-coop candidates from profile metadata.

    Preferred schema:

      "offload_candidates": [
        {
          "m": 8,
          "total_ms": 50.9,
          "pc_ms": 12.4,
          "phone_ms": 24.2,
          "network_ms": 14.3,
          "source": "layer_coop_sweep"
        }
      ]

    Legacy `tx_ms_by_split_m/end_ms_by_split_m` arrays are deliberately not
    returned here because they do not include the PC prefix compute time.  The
    older modeled edge/end search below still uses them together with a prefix
    DP, but they are not safe as direct no-SVD offload candidates.
    """

    candidates: list[OffloadCandidate] = []
    for obj in meta.get("offload_candidates") or []:
        try:
            m = int(obj.get("m", obj.get("offload_m")))
            total_ms = float(obj.get("total_ms", obj.get("steady_ms_per_token")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(total_ms) or total_ms < 0.0:
            continue
        m = max(0, min(n_layers, m))
        candidates.append(
            OffloadCandidate(
                m=m,
                total_ms=total_ms,
                pc_ms=float(obj.get("pc_ms", obj.get("prefix_ms", obj.get("prefix_decode_ms", 0.0))) or 0.0),
                phone_ms=float(obj.get("phone_ms", obj.get("server_ms", obj.get("server_decode_ms", 0.0))) or 0.0),
                network_ms=float(obj.get("network_ms", obj.get("network_wait_ms", 0.0)) or 0.0),
                source=str(obj.get("source", "offload_candidates")),
            )
        )
    if candidates:
        return sorted(candidates, key=lambda item: item.m)
    return []


def _nonzero_rate_count(result: LocalScheduleResult | None) -> int:
    if result is None or not result.decisions:
        return 0
    return sum(1 for item in result.decisions if item.rate > 0.0)


def _local_choice_key(result: LocalScheduleResult) -> tuple[float, int, int, float]:
    # The DP objective is minimum loss. Ties should naturally prefer the
    # no-SVD/no-minor case when all cores are idle and full local execution fits.
    p = result.p if result.p is not None else 0
    return (
        result.total_loss,
        _nonzero_rate_count(result),
        -p,
        result.total_main_ms,
    )


def _schedule_choice_key(result: JointScheduleResult, *, prefer_loss_within_ms: float = 0.0) -> tuple[float, float, int, int, float]:
    """Global choice: minimize estimated latency, then preserve accuracy.

    The old implementation made local/SVD win as soon as it was feasible.  The
    Algorithmv2 decision point needs a real comparison: first compute the best
    SVD/local plan, then compute the best layer-offload plan, then choose the
    lower estimated end-to-end latency.  `prefer_loss_within_ms` can optionally
    keep a lower-loss plan when two plans are practically tied.
    """

    local_loss = result.local.total_loss if result.local else 0.0
    clipped = _nonzero_rate_count(result.local)
    offloaded = len(result.offloaded_layers or [])
    # Quantize the first objective only if an explicit tie band is requested.
    if prefer_loss_within_ms > 0.0 and math.isfinite(result.total_ms):
        latency_key = math.floor(result.total_ms / prefer_loss_within_ms)
    else:
        latency_key = result.total_ms
    mode_penalty = 0 if result.mode in ("baseline_no_svd", "major_only_no_svd", "local") else 1
    return (latency_key, local_loss, clipped, mode_penalty, float(offloaded))


def _empty_local_result(
    n_layers: int,
    deadline_ms: float,
    *,
    split_id: str = "phone_full",
) -> LocalScheduleResult:
    decisions = [
        LayerDecision(
            layer=layer,
            rate=0.0,
            main_ms=0.0,
            loss=0.0,
            weight=0.0,
        )
        for layer in range(n_layers)
    ]
    return LocalScheduleResult(
        feasible=True,
        deadline_ms=deadline_ms,
        split_id=split_id,
        p=0,
        major_cpus=[],
        minor_cpus=[],
        total_main_ms=0.0,
        total_loss=0.0,
        decisions=decisions,
    )


def _baseline_ms_from_meta(meta: dict[str, Any], layers_or_n: int | Iterable[LayerProfile]) -> float:
    for key in ("baseline_no_svd_ms", "baseline_ms", "full_local_ms", "deadline_base_ms", "model_pred_full_base_ms"):
        value = meta.get(key)
        if value is None:
            continue
        try:
            ms = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(ms) and ms > 0.0:
            return ms

    if isinstance(layers_or_n, int):
        return INF
    total = 0.0
    found = False
    for layer in layers_or_n:
        zero = [candidate.main_ms for candidate in layer.candidates if candidate.rate <= 0.0]
        if not zero:
            return INF
        total += min(zero)
        found = True
    return total if found and total > 0.0 else INF


def _make_baseline_result(
    n_layers: int,
    baseline_ms: float,
    request_deadline_ms: float,
) -> JointScheduleResult | None:
    if not math.isfinite(baseline_ms) or baseline_ms <= 0.0:
        return None
    local = _empty_local_result(n_layers, request_deadline_ms, split_id="baseline_no_svd")
    local.total_main_ms = baseline_ms
    return JointScheduleResult(
        mode="baseline_no_svd",
        feasible=True,
        split_m=n_layers - 1,
        offload_m=n_layers,
        local=local,
        total_ms=baseline_ms,
        offloaded_layers=[],
        reason="baseline guard",
    )


def _make_major_only_result(
    split: CoreSplitProfile,
    n_layers: int,
    request_deadline_ms: float,
) -> JointScheduleResult | None:
    """Run the full no-SVD model only on the low-utilization major cores.

    This candidate is different from local SVD: it uses no truncation, no
    minor/tail work, and therefore has zero quality loss.  It exists because
    under heterogeneous load, excluding highly loaded cores can be faster than
    using all cores or trying to split SVD work across bad minor cores.
    """

    total_ms = 0.0
    decisions: list[LayerDecision] = []
    for layer in split.layers:
        zero_candidates = [candidate for candidate in layer.candidates if candidate.rate <= 0.0]
        if not zero_candidates:
            return None
        candidate = min(zero_candidates, key=lambda item: item.main_ms)
        total_ms += candidate.main_ms
        decisions.append(
            LayerDecision(
                layer=layer.layer,
                rate=0.0,
                main_ms=candidate.main_ms,
                loss=0.0,
                weight=0.0,
                tail_ms=0.0,
            )
        )
    local = LocalScheduleResult(
        feasible=True,
        deadline_ms=request_deadline_ms,
        split_id=f"{split.split_id}_major_only_no_svd",
        p=split.p,
        major_cpus=list(split.major_cpus),
        minor_cpus=list(split.minor_cpus),
        total_main_ms=total_ms,
        total_loss=0.0,
        decisions=decisions,
    )
    return JointScheduleResult(
        mode="major_only_no_svd",
        feasible=True,
        split_m=n_layers - 1,
        offload_m=n_layers,
        local=local,
        total_ms=total_ms,
        offloaded_layers=[],
        reason="full no-SVD execution on major cores only",
    )


def _make_major_only_result_from_candidate(
    candidate: MajorOnlyCandidate,
    n_layers: int,
    request_deadline_ms: float,
) -> JointScheduleResult | None:
    if not math.isfinite(candidate.total_ms) or candidate.total_ms <= 0.0:
        return None
    decisions = [
        LayerDecision(layer=layer, rate=0.0, main_ms=candidate.total_ms / max(1, n_layers), loss=0.0, weight=0.0)
        for layer in range(n_layers)
    ]
    local = LocalScheduleResult(
        feasible=True,
        deadline_ms=request_deadline_ms,
        split_id=f"measured_major_only_p{candidate.p}",
        p=candidate.p,
        major_cpus=list(candidate.major_cpus),
        minor_cpus=list(candidate.minor_cpus),
        total_main_ms=candidate.total_ms,
        total_loss=0.0,
        decisions=decisions,
    )
    return JointScheduleResult(
        mode="major_only_no_svd",
        feasible=True,
        split_m=n_layers - 1,
        offload_m=n_layers,
        local=local,
        total_ms=candidate.total_ms,
        offloaded_layers=[],
        reason=candidate.source or "measured major-only candidate",
    )


def _load_major_only_candidates(meta: dict[str, Any]) -> list[MajorOnlyCandidate]:
    candidates: list[MajorOnlyCandidate] = []
    for obj in meta.get("major_only_candidates") or []:
        try:
            p = int(obj.get("p", obj.get("n_run_cpus")))
            total_ms = float(obj.get("total_ms", obj.get("decode_ms", obj.get("generation_decode_ms"))))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(total_ms) or total_ms <= 0.0:
            continue
        candidates.append(
            MajorOnlyCandidate(
                p=p,
                major_cpus=_parse_cpus(obj.get("major_cpus", obj.get("run_cpus"))),
                minor_cpus=_parse_cpus(obj.get("minor_cpus", ())),
                total_ms=total_ms,
                source=str(obj.get("source", "major_only_candidates")),
            )
        )
    return candidates


def _best_major_only_candidate(
    splits: Iterable[CoreSplitProfile],
    n_layers: int,
    request_deadline_ms: float,
    meta: dict[str, Any] | None = None,
) -> JointScheduleResult | None:
    best: JointScheduleResult | None = None
    best_key: tuple[float, float, int, int, float] | None = None
    for candidate in _load_major_only_candidates(meta or {}):
        result = _make_major_only_result_from_candidate(candidate, n_layers, request_deadline_ms)
        if result is None:
            continue
        key = _schedule_choice_key(result)
        if best_key is None or key < best_key:
            best = result
            best_key = key
    for split in splits:
        if not split.minor_cpus:
            continue
        result = _make_major_only_result(split, n_layers, request_deadline_ms)
        if result is None:
            continue
        key = _schedule_choice_key(result)
        if best_key is None or key < best_key:
            best = result
            best_key = key
    return best


def _speedup_vs_baseline(result: JointScheduleResult, baseline_ms: float) -> float:
    if not math.isfinite(baseline_ms) or baseline_ms <= 0.0:
        return 0.0
    if not math.isfinite(result.total_ms) or result.total_ms <= 0.0:
        return 0.0
    return baseline_ms / result.total_ms


def _select_with_baseline_guard(
    choices: list[JointScheduleResult],
    baseline: JointScheduleResult | None,
    *,
    min_speedup: float,
    max_speedup: float = 0.0,
) -> JointScheduleResult | None:
    """Choose a schedule while guaranteeing a no-SVD fallback candidate.

    Algorithmv2's DP finds the best local SVD solution under a deadline.  In
    the real system that is not enough: the estimated model can be wrong, and
    sometimes the best action is to do nothing.  This guard makes
    `baseline_no_svd` an explicit candidate and only enables a non-baseline
    plan when its estimated latency beats baseline by `min_speedup`.

    When `max_speedup` is positive, candidates inside the requested speedup
    band are preferred over overly aggressive SVD choices.  This is useful for
    controlled experiments that target a 10%-20% gain instead of maximum
    clipping.
    """

    if baseline is None:
        return min(choices, key=_schedule_choice_key) if choices else None

    baseline_ms = baseline.total_ms
    min_speedup = max(1.0, float(min_speedup))
    max_speedup = max(0.0, float(max_speedup))

    lossless_no_svd_choices: list[tuple[tuple[float, float, int, int], JointScheduleResult]] = []
    for result in choices:
        if result.mode not in ("major_only_no_svd", "edge_end_no_svd"):
            continue
        speedup = _speedup_vs_baseline(result, baseline_ms)
        if speedup > 1.0 + 1e-12:
            p = result.local.p if result.local and result.local.p is not None else 0
            # no-SVD candidates have no PPL loss.  Pick the fastest lossless
            # route first, whether it is major-only local execution or
            # layer-level phone offload.  This preserves the user's intended
            # major-only-before-SVD rule without accidentally hiding a faster
            # no-SVD offload candidate.
            mode_rank = 0 if result.mode == "major_only_no_svd" else 1
            lossless_no_svd_choices.append(((-speedup, result.total_ms, mode_rank, p), result))
    if lossless_no_svd_choices:
        return min(lossless_no_svd_choices, key=lambda item: item[0])[1]

    eligible: list[tuple[tuple[float, float, float, float, int, int], JointScheduleResult]] = []
    fallback: list[tuple[tuple[float, float, int, int, float], JointScheduleResult]] = []

    for result in choices:
        if result.mode == "baseline_no_svd":
            continue
        speedup = _speedup_vs_baseline(result, baseline_ms)
        if speedup + 1e-12 < min_speedup:
            continue
        local_loss = result.local.total_loss if result.local else 0.0
        clipped = _nonzero_rate_count(result.local)
        if max_speedup > 0.0 and speedup <= max_speedup + 1e-12:
            center = (min_speedup + max_speedup) / 2.0
            eligible.append(((local_loss, clipped, abs(speedup - center), result.total_ms, -speedup, 0), result))
        else:
            fallback.append((_schedule_choice_key(result), result))

    if eligible:
        return min(eligible, key=lambda item: item[0])[1]
    if fallback and max_speedup <= 0.0:
        return min(fallback, key=lambda item: item[0])[1]
    if fallback and max_speedup > 0.0:
        # No in-band candidate exists. Prefer a small overshoot over falling
        # back to baseline only when it still satisfies the lower guard.
        return min(fallback, key=lambda item: (_speedup_vs_baseline(item[1], baseline_ms) - max_speedup, item[0]))[1]
    return baseline


def _make_offload_result(
    candidate: OffloadCandidate,
    n_layers: int,
    request_deadline_ms: float,
) -> JointScheduleResult | None:
    if candidate.total_ms > request_deadline_ms + 1e-9:
        return None
    m = max(0, min(n_layers, candidate.m))
    old_split_m = m - 1 if m > 0 else None
    local = _empty_local_result(n_layers, request_deadline_ms, split_id=f"offload_m{m}")
    return JointScheduleResult(
        mode="edge_end_no_svd",
        feasible=True,
        split_m=old_split_m,
        offload_m=m,
        local=local,
        tx_ms=candidate.network_ms,
        end_ms=candidate.phone_ms,
        network_ms=candidate.network_ms,
        total_ms=candidate.total_ms,
        offloaded_layers=list(range(m, n_layers)),
        reason=candidate.source,
    )


def _best_offload_candidate(
    meta: dict[str, Any],
    n_layers: int,
    request_deadline_ms: float,
) -> JointScheduleResult | None:
    best: JointScheduleResult | None = None
    best_key: tuple[float, float, int, int, float] | None = None
    for candidate in _load_offload_candidates(meta, n_layers):
        result = _make_offload_result(candidate, n_layers, request_deadline_ms)
        if result is None:
            continue
        key = _schedule_choice_key(result)
        if best_key is None or key < best_key:
            best_key = key
            best = result
    return best


def solve_local_dp_over_splits(
    splits: Iterable[CoreSplitProfile],
    deadline_ms: float,
    timeout_budget_ms: float = 0.0,
    quantum_ms: float = 1.0,
) -> LocalScheduleResult:
    best: LocalScheduleResult | None = None
    failures = []
    for split in splits:
        result = solve_local_dp(
            split.layers,
            deadline_ms=deadline_ms,
            timeout_budget_ms=timeout_budget_ms,
            quantum_ms=quantum_ms,
            split=split,
        )
        if result.feasible:
            if best is None or _local_choice_key(result) < _local_choice_key(best):
                best = result
        else:
            failures.append(f"{split.split_id}: {result.reason}")
    if best is not None:
        return best
    return LocalScheduleResult(
        feasible=False,
        deadline_ms=deadline_ms,
        reason="; ".join(failures) if failures else "no core split candidates",
    )


def solve_joint_schedule(
    layers: list[LayerProfile],
    local_deadline_ms: float,
    request_deadline_ms: float,
    timeout_budget_ms: float = 0.0,
    quantum_ms: float = 1.0,
    meta: dict[str, Any] | None = None,
) -> JointScheduleResult:
    """Solve chapter 5.4 flow and compare local SVD with layer offload."""

    meta = meta or {}
    local = solve_local_dp(layers, local_deadline_ms, timeout_budget_ms, quantum_ms)
    n_layers = len(layers)
    choices: list[JointScheduleResult] = []
    baseline = _make_baseline_result(
        n_layers,
        baseline_ms=_baseline_ms_from_meta(meta, layers),
        request_deadline_ms=request_deadline_ms,
    )
    if baseline is not None:
        choices.append(baseline)
    if local.feasible:
        if not bool(meta.get("quality_first_no_svd", False)):
            choices.append(
                JointScheduleResult(
                    mode="local",
                    feasible=True,
                    split_m=n_layers - 1,
                    offload_m=n_layers,
                    local=local,
                    total_ms=local.total_main_ms,
                    offloaded_layers=[],
                )
            )

    offload = _best_offload_candidate(meta, n_layers, request_deadline_ms)
    if offload is not None:
        choices.append(offload)

    if choices:
        selected = _select_with_baseline_guard(
            choices,
            baseline,
            min_speedup=float(meta.get("scheduler_min_speedup_vs_baseline", meta.get("min_speedup_vs_baseline", 1.0))),
            max_speedup=float(meta.get("scheduler_max_speedup_vs_baseline", meta.get("max_speedup_vs_baseline", 0.0))),
        )
        if selected is not None:
            return selected

    tx_series = _metric_series(meta, "tx_ms_by_split_m", n_layers)
    end_series = _metric_series(meta, "end_ms_by_split_m", n_layers)
    best: JointScheduleResult | None = None
    best_key: tuple[float, float, int, int, float] | None = None

    for split_m in range(n_layers - 1, -1, -1):
        tx_ms = tx_series[split_m]
        end_ms = end_series[split_m]
        edge_deadline = request_deadline_ms - tx_ms - end_ms
        if edge_deadline < 0.0:
            continue
        prefix = layers[: split_m + 1]
        edge = solve_local_dp(prefix, edge_deadline, timeout_budget_ms, quantum_ms)
        if not edge.feasible:
            continue
        total_ms = edge.total_main_ms + tx_ms + end_ms
        if total_ms <= request_deadline_ms + 1e-9:
            result = JointScheduleResult(
                mode="edge_end",
                feasible=True,
                split_m=split_m,
                offload_m=split_m + 1,
                local=edge,
                tx_ms=tx_ms,
                end_ms=end_ms,
                network_ms=tx_ms,
                total_ms=total_ms,
                offloaded_layers=list(range(split_m + 1, n_layers)),
            )
            key = _schedule_choice_key(result)
            if best_key is None or key < best_key:
                best_key = key
                best = result

    if best is not None:
        return best

    return JointScheduleResult(
        mode="none",
        feasible=False,
        split_m=None,
        local=local,
        reason="local DP infeasible and no suffix offload split satisfies the request deadline",
    )


def solve_joint_schedule_over_splits(
    splits: list[CoreSplitProfile],
    local_deadline_ms: float,
    request_deadline_ms: float,
    timeout_budget_ms: float = 0.0,
    quantum_ms: float = 1.0,
    meta: dict[str, Any] | None = None,
) -> JointScheduleResult:
    """Solve Algorithmv2 5.2 with outer core-split enumeration.

    The scheduler now follows the intended final decision:
    1. compute the best local SVD/no-SVD DP solution over all core splits;
    2. compute the best layer-offload/no-SVD solution for the current hardware;
    3. choose the lower estimated end-to-end latency, using loss only as a
       tie-breaker.
    """

    meta = meta or {}
    if not splits:
        return JointScheduleResult(
            mode="none",
            feasible=False,
            split_m=None,
            local=None,
            reason="no core split candidates",
        )

    local = solve_local_dp_over_splits(
        splits,
        deadline_ms=local_deadline_ms,
        timeout_budget_ms=timeout_budget_ms,
        quantum_ms=quantum_ms,
    )
    n_layers = len(splits[0].layers)
    choices: list[JointScheduleResult] = []
    baseline = _make_baseline_result(
        n_layers,
        baseline_ms=_baseline_ms_from_meta(meta, n_layers),
        request_deadline_ms=request_deadline_ms,
    )
    if baseline is not None:
        choices.append(baseline)
    if local.feasible:
        if not bool(meta.get("quality_first_no_svd", False)):
            choices.append(
                JointScheduleResult(
                    mode="local",
                    feasible=True,
                    split_m=n_layers - 1,
                    offload_m=n_layers,
                    local=local,
                    total_ms=local.total_main_ms,
                    offloaded_layers=[],
                )
            )

    major_only = _best_major_only_candidate(splits, n_layers, request_deadline_ms, meta)
    if major_only is not None:
        choices.append(major_only)

    offload = _best_offload_candidate(meta, n_layers, request_deadline_ms)
    if offload is not None:
        choices.append(offload)

    if choices:
        selected = _select_with_baseline_guard(
            choices,
            baseline,
            min_speedup=float(meta.get("scheduler_min_speedup_vs_baseline", meta.get("min_speedup_vs_baseline", 1.0))),
            max_speedup=float(meta.get("scheduler_max_speedup_vs_baseline", meta.get("max_speedup_vs_baseline", 0.0))),
        )
        if selected is not None:
            return selected

    tx_series = _metric_series(meta, "tx_ms_by_split_m", n_layers)
    end_series = _metric_series(meta, "end_ms_by_split_m", n_layers)
    best: JointScheduleResult | None = None
    best_key: tuple[int, float, float, int] | None = None

    for split_m in range(n_layers - 1, -1, -1):
        tx_ms = tx_series[split_m]
        end_ms = end_series[split_m]
        edge_deadline = request_deadline_ms - tx_ms - end_ms
        if edge_deadline < 0.0:
            continue

        prefix_splits = [
            CoreSplitProfile(
                split_id=split.split_id,
                p=split.p,
                major_cpus=split.major_cpus,
                minor_cpus=split.minor_cpus,
                layers=tuple(layer for layer in split.layers if layer.layer <= split_m),
            )
            for split in splits
        ]
        edge = solve_local_dp_over_splits(
            prefix_splits,
            deadline_ms=edge_deadline,
            timeout_budget_ms=timeout_budget_ms,
            quantum_ms=quantum_ms,
        )
        if not edge.feasible:
            continue
        total_ms = edge.total_main_ms + tx_ms + end_ms
        if total_ms > request_deadline_ms + 1e-9:
            continue
        offloaded = n_layers - split_m - 1
        key = (offloaded, edge.total_loss, total_ms, _nonzero_rate_count(edge))
        if best_key is None or key < best_key:
            best_key = key
            best = JointScheduleResult(
                mode="edge_end",
                feasible=True,
                split_m=split_m,
                offload_m=split_m + 1,
                local=edge,
                tx_ms=tx_ms,
                end_ms=end_ms,
                network_ms=tx_ms,
                total_ms=total_ms,
                offloaded_layers=list(range(split_m + 1, n_layers)),
            )

    if best is not None:
        return best

    return JointScheduleResult(
        mode="none",
        feasible=False,
        split_m=None,
        local=local,
        reason="local DP infeasible and no suffix offload split satisfies the request deadline",
    )


def write_rate_file(path: Path, rates: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(f"{max(0.0, min(1.0, rate)):.6g}" for rate in rates) + "\n")


def write_timeout_file(path: Path, timeouts_ms: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(f"{max(0.0, timeout):.6g}" for timeout in timeouts_ms) + "\n")


def result_to_json(result: JointScheduleResult, n_layers: int) -> dict[str, Any]:
    obj = asdict(result)
    if result.local and result.local.decisions is not None:
        obj["rates"] = result.local.rates(n_layers)
        obj["timeouts_ms"] = result.local.timeouts(n_layers)
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve exp12 load-aware SVD schedules.")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--local-deadline-ms", required=True, type=float)
    parser.add_argument("--request-deadline-ms", type=float)
    parser.add_argument("--timeout-budget-ms", type=float, default=0.0)
    parser.add_argument("--quantum-ms", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-rates", type=Path)
    parser.add_argument("--out-timeouts", type=Path)
    args = parser.parse_args()

    splits, meta = load_core_split_profile(args.profile)
    request_deadline_ms = args.request_deadline_ms
    if request_deadline_ms is None:
        request_deadline_ms = args.local_deadline_ms

    result = solve_joint_schedule_over_splits(
        splits=splits,
        local_deadline_ms=args.local_deadline_ms,
        request_deadline_ms=request_deadline_ms,
        timeout_budget_ms=args.timeout_budget_ms,
        quantum_ms=args.quantum_ms,
        meta=meta,
    )
    n_layers = len(splits[0].layers) if splits else 0
    obj = result_to_json(result, n_layers)
    print(json.dumps(obj, ensure_ascii=False, indent=2))

    if args.out_json:
        save_json(args.out_json, obj)
    if args.out_rates and result.local is not None and result.local.decisions is not None:
        write_rate_file(args.out_rates, result.local.rates(n_layers))
    if args.out_timeouts and result.local is not None and result.local.decisions is not None:
        write_timeout_file(args.out_timeouts, result.local.timeouts(n_layers))
    return 0 if result.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
