#!/usr/bin/env python3
"""Fit the deployable grouped-confidence artifact from an exact train-fit run.

This is a training-only utility. scikit-learn is intentionally not part of the
submission runtime closure; the emitted artifact is consumed with stdlib math.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mib_pipeline.grouped_confidence import (  # noqa: E402
    _FEATURE_ORDER,
    evidence_path_from_summary,
    feature_values,
)
from mib_pipeline.models import PredictionRow  # noqa: E402


EXPECTED_CASES = 800
REGULARIZATION_C = 0.3
SOURCE_COMMIT = "6899dd2efdb6b27178c9ccb99c36c978f3d57416"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_path(row: PredictionRow, record: dict) -> str:
    feature = record["features"]
    return evidence_path_from_summary(
        row,
        finding=bool(feature["finding_denied"] or feature["finding_review"]),
        layout_risk=bool(feature["layout_risk_count"]),
        registry_embargo=bool(feature["registry_embargo"]),
        damage=bool(
            feature["damage_illegible"]
            or feature["damage_washed"]
            or feature["damage_redacted"]
        ),
        fee_paid_proven=bool(feature["fee_paid_proven"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--visible-features", type=Path, required=True)
    parser.add_argument("--oof-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prediction_rows = load_jsonl(args.predictions)
    visible_rows = load_jsonl(args.visible_features)
    if len(prediction_rows) != EXPECTED_CASES or len(visible_rows) != EXPECTED_CASES:
        raise ValueError("grouped-confidence fit requires exactly 800 rows")
    predictions = {row["case_id"]: row for row in prediction_rows}
    visible = {row["case_id"]: row for row in visible_rows}
    if len(predictions) != EXPECTED_CASES or set(predictions) != set(visible):
        raise ValueError("prediction/feature case sets are not the same exact 800")

    matrix: list[tuple[float, ...]] = []
    target: list[int] = []
    paths: list[str] = []
    for raw_prediction in prediction_rows:
        case_id = raw_prediction["case_id"]
        raw_visible = visible[case_id]
        row = PredictionRow.from_mapping(
            raw_prediction,
            fallback_case_id=case_id,
        )
        path = visible_path(row, raw_visible)
        matrix.append(feature_values(row, evidence_path=path))
        target.append(
            int(row.adjudication == raw_visible["truth_adjudication"])
        )
        paths.append(path)

    model = LogisticRegression(
        C=REGULARIZATION_C,
        max_iter=5000,
        solver="lbfgs",
        random_state=137,
    )
    model.fit(np.asarray(matrix, dtype=float), np.asarray(target, dtype=int))
    if model.n_iter_[0] >= model.max_iter:
        raise RuntimeError("grouped-confidence logistic fit did not converge")

    oof = json.loads(args.oof_report.read_text(encoding="utf-8"))
    if (
        oof.get("n") != EXPECTED_CASES
        or oof.get("protocol", {}).get("candidate_family") != "all"
        or oof.get("cfa_delta") != 0
        or oof.get("score_delta", {}).get("total_score", 0) <= 0
    ):
        raise ValueError("OOF report does not prove a positive, zero-CFA lead")

    artifact = {
        "artifact_id": "grouped-visible-path-platt-trainfit800-v1",
        "schema_version": 1,
        "fit_metadata": {
            "source_commit": SOURCE_COMMIT,
            "split": "seed42_train_fit_exact800",
            "fit_cases": EXPECTED_CASES,
            "correct_cases": int(sum(target)),
            "incorrect_cases": int(EXPECTED_CASES - sum(target)),
            "path_counts": dict(sorted(Counter(paths).items())),
            "predictions_sha256": sha256(args.predictions),
            "visible_features_sha256": sha256(args.visible_features),
            "nested_oof_report_sha256": sha256(args.oof_report),
            "nested_oof_baseline_brier": oof["baseline_brier"],
            "nested_oof_candidate_brier": oof["nested_oof_brier"],
            "nested_oof_total_delta": oof["score_delta"]["total_score"],
            "nested_oof_cfa_delta": oof["cfa_delta"],
            "full_fit_score_claim": False,
        },
        "identity_policy": (
            "No case IDs, names, sponsor identities, page signatures, answer keys, "
            "or truth-derived runtime features. Labels are used only as the fit target."
        ),
        "model": {
            "regularization_c": REGULARIZATION_C,
            "solver": "lbfgs",
            "probability_clip": [0.01, 0.99],
            "feature_order": list(_FEATURE_ORDER),
            "intercept": float(model.intercept_[0]),
            "coefficients": [float(value) for value in model.coef_[0]],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fit_cases": EXPECTED_CASES,
                "correct_cases": int(sum(target)),
                "paths": dict(sorted(Counter(paths).items())),
                "artifact_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
