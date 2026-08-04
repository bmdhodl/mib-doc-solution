#!/usr/bin/env python3
"""Audit Midas's two-view, high-resolution B-13 risk crop.

Visible-pixel candidate generation completes before labels are loaded.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HARD_FLAGS = frozenset(
    {
        "active_warrant",
        "biohazard_red",
        "memory_tampering",
        "planetary_embargo",
    }
)


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("midas_risk_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task(
    case_id: str,
    *,
    pages: list[str],
    pdf_dir: Path,
    midas: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        flags = midas._high_resolution_risk_flags(
            pdf_dir / f"{case_id}.pdf",
            pages,
        )
        hard = sorted(set(flags) & HARD_FLAGS)
        return {
            "case_id": case_id,
            "hard_flags": hard,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "case_id": case_id,
            "hard_flags": [],
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def classify(before: str, after: str, truth: str) -> str:
    if before != truth and after == truth:
        return "FIX"
    if before == truth and after != truth:
        return "HURT"
    if before == truth and after == truth:
        return "SAME_CORRECT"
    return "WRONG_TO_WRONG"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midas-source", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--runtime-1000", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    midas = load_module(args.midas_source)
    frozen = load_jsonl(args.frozen)
    runtime = load_jsonl(args.runtime_1000)
    cache = load_jsonl(args.cache)
    baseline = dict(runtime)
    baseline.update(frozen)
    if len(frozen) != 800 or len(baseline) != 1000 or len(cache) != 1000:
        raise ValueError("audit requires frozen exact800 and full 1000 inputs")

    routed: list[tuple[str, list[str]]] = []
    for case_id, row in sorted(baseline.items()):
        if row["risk_flags"] != "none":
            continue
        pages = [
            "\n".join(str(line["text"]) for line in page["lines"])
            for page in cache[case_id]["pages"]
        ]
        _current, state = midas._extract_scoped_flags(case_id, pages)
        if state == "unknown":
            routed.append((case_id, pages))

    completed = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                task,
                case_id,
                pages=pages,
                pdf_dir=args.pdf_dir,
                midas=midas,
            ): case_id
            for case_id, pages in routed
        }
        total = len(futures)
        for index, future in enumerate(as_completed(futures), 1):
            completed.append(future.result())
            if index % 50 == 0 or index == total:
                print(
                    json.dumps({"completed": index, "total": total}),
                    flush=True,
                )

    with args.truth.open(encoding="utf-8", newline="") as handle:
        truth = {row["case_id"]: row for row in csv.DictReader(handle)}

    extraction = []
    decision = []
    for result in sorted(completed, key=lambda item: item["case_id"]):
        if len(result["hard_flags"]) != 1:
            continue
        case_id = result["case_id"]
        after_flags = result["hard_flags"][0]
        row = baseline[case_id]
        expected_flags = truth[case_id]["risk_flags"]
        extraction.append(
            {
                "case_id": case_id,
                "partition": "exact800" if case_id in frozen else "validation200",
                "before": row["risk_flags"],
                "after": after_flags,
                "truth": expected_flags,
                "outcome": classify(
                    row["risk_flags"],
                    after_flags,
                    expected_flags,
                ),
                "mechanism": "high_resolution_risk_flags",
            }
        )
        if row["adjudication"] == "NEEDS_REVIEW":
            expected_decision = truth[case_id]["adjudication"]
            decision.append(
                {
                    "case_id": case_id,
                    "partition": (
                        "exact800" if case_id in frozen else "validation200"
                    ),
                    "before": row["adjudication"],
                    "after": "DENIED",
                    "truth": expected_decision,
                    "outcome": classify(
                        row["adjudication"],
                        "DENIED",
                        expected_decision,
                    ),
                    "catastrophic_false_approval": False,
                    "mechanism": "hard_flag_review_to_denied",
                }
            )

    def census(rows: list[dict[str, Any]], partition: str) -> dict[str, Any]:
        selected = [row for row in rows if row["partition"] == partition]
        return {
            "firing_cases": len(selected),
            "outcomes": dict(sorted(Counter(
                row["outcome"] for row in selected
            ).items())),
            "catastrophic_false_approvals": sum(
                bool(row.get("catastrophic_false_approval"))
                for row in selected
            ),
            "details": selected,
        }

    report = {
        "configuration": {
            "answer_key_used_for_candidates": False,
            "applicant_identity_used_for_candidates": False,
            "native_pdf_text_used": False,
            "visible_active_case_binding": True,
            "dpi": 400,
            "psm_consensus": [11, 12],
            "workers": args.workers,
        },
        "routing": {
            "unknown_active_b13_cases": len(routed),
            "errors": sum("error" in result for result in completed),
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
        "extraction": {
            partition: census(extraction, partition)
            for partition in ("exact800", "validation200")
        },
        "decision": {
            partition: census(decision, partition)
            for partition in ("exact800", "validation200")
        },
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "routing": report["routing"],
        "runtime": report["runtime"],
        "extraction": {
            partition: {
                key: report["extraction"][partition][key]
                for key in (
                    "firing_cases",
                    "outcomes",
                    "catastrophic_false_approvals",
                )
            }
            for partition in ("exact800", "validation200")
        },
        "decision": {
            partition: {
                key: report["decision"][partition][key]
                for key in (
                    "firing_cases",
                    "outcomes",
                    "catastrophic_false_approvals",
                )
            }
            for partition in ("exact800", "validation200")
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
