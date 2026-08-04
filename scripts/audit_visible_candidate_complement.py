#!/usr/bin/env python3
"""Run the candidate and emit exact fit/validation firing censuses.

The runtime mechanism remains identity-free. This offline audit uses case IDs
only to join frozen predictions to labels and to partition the exact 800 fit
cases from the remaining 200 labeled validation cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mib_pipeline.rapid_recovery as rapid_recovery  # noqa: E402
from mib_pipeline.models import PredictionRow  # noqa: E402
from mib_pipeline.visible_candidate_complement import (  # noqa: E402
    apply_visible_candidate_complement,
    visible_candidate_complement_repairs,
)
import solution  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> dict[str, PredictionRow]:
    rows: dict[str, PredictionRow] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        row = PredictionRow.from_mapping(raw, fallback_case_id=raw["case_id"])
        if row.case_id in rows:
            raise ValueError(f"duplicate prediction case: {row.case_id}")
        rows[row.case_id] = row
    return rows


def _load_truth(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    if len(rows) != 1000:
        raise ValueError("candidate audit requires the canonical 1000 labels")
    return rows


def _census(
    events: list[dict[str, Any]],
    truth: dict[str, dict[str, str]],
    *,
    use_frozen: bool,
) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    by_field: Counter[str] = Counter()
    by_mechanism: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: item["case_id"]):
        before = event["frozen_before"] if use_frozen else event["runtime_before"]
        repairs = event["frozen_repairs"] if use_frozen else event["runtime_repairs"]
        for repair in repairs:
            field_name = repair["field_name"]
            value = repair["value"]
            expected = truth[event["case_id"]][field_name]
            before_correct = before[field_name] == expected
            after_correct = value == expected
            if not before_correct and after_correct:
                outcome = "FIX"
            elif before_correct and not after_correct:
                outcome = "HURT"
            elif before_correct and after_correct:
                outcome = "SAME_CORRECT"
            else:
                outcome = "WRONG_TO_WRONG"
            outcomes[outcome] += 1
            by_field[field_name] += 1
            by_mechanism[repair["mechanism"]] += 1
            details.append(
                {
                    "case_id": event["case_id"],
                    "field_name": field_name,
                    "mechanism": repair["mechanism"],
                    "before": before[field_name],
                    "after": value,
                    "truth": expected,
                    "outcome": outcome,
                }
            )
    return {
        "cases": len(events),
        "firing_cases": len({detail["case_id"] for detail in details}),
        "firing_repairs": len(details),
        "outcomes": dict(sorted(outcomes.items())),
        "by_field": dict(sorted(by_field.items())),
        "by_mechanism": dict(sorted(by_mechanism.items())),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    frozen = _load_jsonl(args.frozen_predictions)
    if len(frozen) != 800:
        raise ValueError("frozen prediction input must contain the exact 800 cases")
    truth = _load_truth(args.truth)
    input_cases = {
        path.stem
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == ".pdf"
    }
    if input_cases != set(truth):
        raise ValueError("input directory must contain the canonical labeled 1000")

    lock = threading.Lock()
    events: list[dict[str, Any]] = []
    original = rapid_recovery.apply_visible_candidate_complement

    def audited(row: PredictionRow, candidates: Any) -> PredictionRow:
        materialized = tuple(candidates)
        runtime_repairs = visible_candidate_complement_repairs(row, materialized)
        frozen_row = frozen.get(row.case_id, row)
        frozen_repairs = visible_candidate_complement_repairs(
            frozen_row,
            materialized,
        )
        event = {
            "case_id": row.case_id,
            "runtime_before": row.to_dict(),
            "runtime_repairs": [
                {
                    "field_name": repair.field_name,
                    "value": repair.value,
                    "mechanism": repair.mechanism,
                }
                for repair in runtime_repairs
            ],
            "frozen_before": frozen_row.to_dict(),
            "frozen_repairs": [
                {
                    "field_name": repair.field_name,
                    "value": repair.value,
                    "mechanism": repair.mechanism,
                }
                for repair in frozen_repairs
            ],
        }
        with lock:
            events.append(event)
        return original(row, materialized)

    rapid_recovery.apply_visible_candidate_complement = audited
    os.environ["MIB_MAX_WORKERS"] = str(args.workers)
    started = time.perf_counter()
    try:
        exit_code = solution.main(
            ["solution.py", str(args.input_dir), str(args.output)]
        )
    finally:
        rapid_recovery.apply_visible_candidate_complement = original
    elapsed = time.perf_counter() - started
    if exit_code != 0:
        raise RuntimeError(f"candidate solution exited {exit_code}")
    if len(events) != 1000 or len({event["case_id"] for event in events}) != 1000:
        raise RuntimeError("audit did not observe exactly 1000 unique cases")

    fit_events = [event for event in events if event["case_id"] in frozen]
    validation_events = [event for event in events if event["case_id"] not in frozen]
    report = {
        "source_commit": "6899dd2efdb6b27178c9ccb99c36c978f3d57416",
        "inputs": {
            "frozen_predictions": str(args.frozen_predictions),
            "frozen_predictions_sha256": _sha256(args.frozen_predictions),
            "truth": str(args.truth),
            "truth_sha256": _sha256(args.truth),
            "input_cases": len(input_cases),
        },
        "runtime": {
            "workers": args.workers,
            "elapsed_seconds": elapsed,
            "seconds_per_pdf_wall": elapsed / len(events),
            "output": str(args.output),
            "output_sha256": _sha256(args.output),
        },
        "exact800_frozen_census": _census(
            fit_events,
            truth,
            use_frozen=True,
        ),
        "validation200_runtime_census": _census(
            validation_events,
            truth,
            use_frozen=False,
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("runtime",)}, sort_keys=True))
    print(
        json.dumps(
            {
                "exact800": {
                    key: report["exact800_frozen_census"][key]
                    for key in (
                        "firing_cases",
                        "firing_repairs",
                        "outcomes",
                        "by_field",
                        "by_mechanism",
                    )
                },
                "validation200": {
                    key: report["validation200_runtime_census"][key]
                    for key in (
                        "firing_cases",
                        "firing_repairs",
                        "outcomes",
                        "by_field",
                        "by_mechanism",
                    )
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
