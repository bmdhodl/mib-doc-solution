#!/usr/bin/env python3
"""Replay the pinned grouped calibrator over frozen predictions and PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mib_pipeline import score_heads  # noqa: E402
from mib_pipeline.grouped_confidence import (  # noqa: E402
    GroupedConfidenceRecalibrator,
    evidence_path_from_text,
)
from mib_pipeline.models import FIELD_NAMES, PredictionRow  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = load_jsonl(args.predictions)
    if len(source) != 800 or len({row["case_id"] for row in source}) != 800:
        raise ValueError("replay requires the exact unique 800-row prediction set")
    recalibrator = GroupedConfidenceRecalibrator.from_pinned_artifact()
    output: list[dict] = []
    paths: Counter[str] = Counter()
    non_confidence_drift = 0
    for index, raw in enumerate(source, start=1):
        row = PredictionRow.from_mapping(raw, fallback_case_id=raw["case_id"])
        pdf_path = args.pdf_dir / f"{row.case_id}.pdf"
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        text = score_heads._pdf_layout_text(pdf_path)
        path = evidence_path_from_text(row, text)
        paths[path] += 1
        changed = recalibrator.recalibrate(pdf_path, row)
        before = row.to_dict()
        after = changed.to_dict()
        if any(
            before[field] != after[field]
            for field in FIELD_NAMES
            if field != "confidence"
        ):
            non_confidence_drift += 1
        output.append(after)
        if index % 100 == 0:
            print(f"{index}/{len(source)}", flush=True)

    if non_confidence_drift:
        raise RuntimeError(
            f"grouped calibrator changed {non_confidence_drift} non-confidence rows"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "rows": len(output),
                "non_confidence_drift": non_confidence_drift,
                "path_counts": dict(sorted(paths.items())),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
