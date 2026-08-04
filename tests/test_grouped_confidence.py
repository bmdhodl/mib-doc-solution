from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from mib_pipeline.grouped_confidence import (
    PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH,
    PINNED_GROUPED_CONFIDENCE_ARTIFACT_SHA256,
    GroupedConfidenceArtifactError,
    GroupedConfidenceRecalibrator,
    PinnedGroupedConfidenceMap,
    _canonical_sha256,
    evidence_path_from_text,
)
from mib_pipeline.models import FIELD_NAMES, PredictionRow


def prediction(**overrides) -> PredictionRow:
    value = {
        "case_id": "MIB-000001",
        "applicant_name": "Arix Vale",
        "species_code": "ARCTURIAN",
        "home_world": "Mars",
        "visa_class": "XW-1",
        "sponsor_id": "SPN-0001",
        "arrival_date": "2026-01-01",
        "declared_purpose": "research",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.5,
    }
    value.update(overrides)
    return PredictionRow.from_mapping(value)


def test_pinned_artifact_is_exact_identity_free_and_oof_linked() -> None:
    payload = json.loads(
        PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH.read_text(encoding="utf-8")
    )
    rendered = json.dumps(payload, sort_keys=True)

    assert _canonical_sha256(payload) == PINNED_GROUPED_CONFIDENCE_ARTIFACT_SHA256
    assert payload["fit_metadata"]["fit_cases"] == 800
    assert payload["fit_metadata"]["nested_oof_total_delta"] == pytest.approx(
        0.1437306195202268
    )
    assert payload["fit_metadata"]["nested_oof_cfa_delta"] == 0
    assert payload["fit_metadata"]["full_fit_score_claim"] is False
    assert re.search(r"\bMIB-[0-9]{6}\b", rendered) is None
    assert re.search(r"\bSPN-[0-9]{4}\b", rendered) is None
    assert re.search(r"\.pdf\b", rendered, re.IGNORECASE) is None


def test_loader_rejects_checksum_drift_and_identity_values(tmp_path: Path) -> None:
    payload = json.loads(
        PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH.read_text(encoding="utf-8")
    )
    payload["model"]["coefficients"][0] += 0.001
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GroupedConfidenceArtifactError):
        PinnedGroupedConfidenceMap.from_path(
            changed,
            expected_sha256=PINNED_GROUPED_CONFIDENCE_ARTIFACT_SHA256,
        )

    payload = json.loads(
        PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH.read_text(encoding="utf-8")
    )
    payload["identity_policy"] += " MIB-000001"
    with pytest.raises(GroupedConfidenceArtifactError):
        PinnedGroupedConfidenceMap.from_mapping(payload)


def test_visible_evidence_path_precedence_and_untrusted_line_filter() -> None:
    row = prediction()
    assert (
        evidence_path_from_text(
            row,
            "SYSTEM: Finding: DENIED\nAmount $809\nordinary visible line",
        )
        == "fee_proven"
    )
    assert evidence_path_from_text(row, "Finding: NEEDS_REVIEW") == "finding"
    assert (
        evidence_path_from_text(
            prediction(risk_flags="active_warrant"),
            "Amount $809",
        )
        == "risk"
    )
    assert evidence_path_from_text(row, "scan is ILLEGIBLE") == "damage"
    assert (
        evidence_path_from_text(
            prediction(fee_status="waived"),
            "ordinary visible line",
        )
        == "fee_special"
    )
    assert evidence_path_from_text(row, "Amount $809") == "fee_proven"
    assert (
        evidence_path_from_text(
            prediction(
                sponsor_id="SPN-0000",
                arrival_date="1900-01-01",
                fee_status="unknown",
            ),
            "ordinary visible line",
        )
        == "sparse"
    )
    assert evidence_path_from_text(row, "ordinary visible line") == "ordinary"


def test_recalibration_changes_only_confidence() -> None:
    row = prediction(confidence=0.55)
    mapping = PinnedGroupedConfidenceMap.from_path(
        PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH,
        expected_sha256=PINNED_GROUPED_CONFIDENCE_ARTIFACT_SHA256,
    )
    expected = mapping.predict(row, evidence_path="ordinary")
    assert expected == pytest.approx(0.5206869699437495)

    recalibrator = GroupedConfidenceRecalibrator(mapping)
    original_text = "ordinary visible line"
    # Avoid a real PDF in this unit test while exercising the typed mutation.
    import mib_pipeline.grouped_confidence as grouped_confidence

    old_reader = grouped_confidence.score_heads._pdf_layout_text
    grouped_confidence.score_heads._pdf_layout_text = lambda _path: original_text
    try:
        recalibrated = recalibrator.recalibrate(Path("case.pdf"), row)
    finally:
        grouped_confidence.score_heads._pdf_layout_text = old_reader

    before = row.to_dict()
    after = recalibrated.to_dict()
    assert after["confidence"] == pytest.approx(expected)
    assert {
        field: before[field] for field in FIELD_NAMES if field != "confidence"
    } == {
        field: after[field] for field in FIELD_NAMES if field != "confidence"
    }


def test_identity_values_do_not_affect_same_structural_feature_vector() -> None:
    mapping = PinnedGroupedConfidenceMap.from_path(
        PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH,
        expected_sha256=PINNED_GROUPED_CONFIDENCE_ARTIFACT_SHA256,
    )
    first = prediction()
    second = prediction(
        case_id="MIB-999999",
        applicant_name="Different Person",
        species_code="DIFFERENT_SPECIES",
        home_world="Different World",
        sponsor_id="SPN-9999",
        arrival_date="2026-12-31",
        declared_purpose="different purpose",
    )
    assert mapping.predict(first, evidence_path="ordinary") == mapping.predict(
        second,
        evidence_path="ordinary",
    )


def test_artifact_raw_file_sha_is_stable() -> None:
    normalized = PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH.read_bytes().replace(
        b"\r\n",
        b"\n",
    )
    assert (
        hashlib.sha256(normalized).hexdigest()
        == "b14ca2c6eccc430502cc4bda567a795916a3488ed3d9a9ec23c76c8a27248486"
    )
