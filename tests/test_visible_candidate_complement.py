from __future__ import annotations

from mib_pipeline.extraction import CandidateEvidence, EvidenceType
from mib_pipeline.ingestion import Rect
from mib_pipeline.models import FIELD_NAMES, PredictionRow
from mib_pipeline.visible_candidate_complement import (
    apply_visible_candidate_complement,
    visible_candidate_complement_repairs,
)


def row(**overrides) -> PredictionRow:
    values = {
        "case_id": "MIB-000001",
        "applicant_name": "Visible Applicant",
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "XW-1",
        "sponsor_id": "SPN-1234",
        "arrival_date": "2026-04-17",
        "declared_purpose": "reactor maintenance",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.41,
    }
    values.update(overrides)
    return PredictionRow.from_mapping(values)


def candidate(
    field_name: str,
    value: str | None,
    *,
    evidence_type: EvidenceType = EvidenceType.INTAKE_FORM,
    confidence: float = 0.90,
    page: int = 0,
    cues: tuple[str, ...] = (),
    legible: bool = True,
    superseded: bool = False,
    source: str = "visible_ocr",
    case_id_hint: str | None = "MIB-999999",
    applicant_hint: str | None = "Different Applicant",
) -> CandidateEvidence:
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=page,
        box=Rect(1, 2, 3, 4),
        legible=legible,
        superseded=superseded,
        ocr_confidence=confidence,
        visual_cues=cues,
        source=source,
        case_id_hint=case_id_hint,
        applicant_hint=applicant_hint,
    )


def complete_intake_sponsor(
    value: str,
    *,
    confidence: float,
    page: int,
) -> tuple[CandidateEvidence, ...]:
    return (
        candidate("visa_class", "XW-1", confidence=0.95, page=page),
        candidate("sponsor_id", value, confidence=confidence, page=page),
        candidate("arrival_date", "2026-04-17", confidence=0.90, page=page),
        candidate(
            "declared_purpose",
            "field repair",
            confidence=0.90,
            page=page,
        ),
    )


def test_recovers_unique_nondefault_intake_purpose_only_from_default() -> None:
    original = row()
    candidates = (
        candidate("declared_purpose", "field repair", confidence=0.55),
    )

    repaired = apply_visible_candidate_complement(original, candidates)

    assert repaired.declared_purpose == "field repair"
    assert [repair.mechanism for repair in visible_candidate_complement_repairs(
        original,
        candidates,
    )] == ["unique_nondefault_intake_purpose"]
    assert {
        field: getattr(repaired, field)
        for field in FIELD_NAMES
        if field != "declared_purpose"
    } == {
        field: getattr(original, field)
        for field in FIELD_NAMES
        if field != "declared_purpose"
    }
    assert apply_visible_candidate_complement(
        row(declared_purpose="research"),
        candidates,
    ).declared_purpose == "research"


def test_purpose_abstains_on_conflict_low_confidence_and_bad_cues() -> None:
    original = row()
    assert apply_visible_candidate_complement(
        original,
        (
            candidate("declared_purpose", "field repair", confidence=0.55),
            candidate("declared_purpose", "research", confidence=0.90),
        ),
    ) == original
    assert apply_visible_candidate_complement(
        original,
        (candidate("declared_purpose", "field repair", confidence=0.49),),
    ) == original
    assert apply_visible_candidate_complement(
        original,
        (
            candidate(
                "declared_purpose",
                "field repair",
                cues=("sample_denial_watermark",),
            ),
        ),
    ) == original


def test_structured_attestation_sponsor_beats_intake_without_identity_gate() -> None:
    original = row(sponsor_id="SPN-0007")
    candidates = (
        candidate("sponsor_id", "SPN-0007", confidence=0.93),
        candidate(
            "sponsor_id",
            "SPN-5751",
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            confidence=0.82,
            page=2,
        ),
        candidate(
            "visa_class",
            "TRANSIT-7",
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            page=2,
        ),
        candidate(
            "declared_purpose",
            None,
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            confidence=0.80,
            page=2,
            legible=False,
        ),
    )

    repaired = apply_visible_candidate_complement(original, candidates)

    assert repaired.sponsor_id == "SPN-5751"
    assert visible_candidate_complement_repairs(
        original,
        candidates,
    )[0].mechanism == "structured_attestation_sponsor"


def test_structured_attestation_abstains_without_full_page_structure() -> None:
    original = row(sponsor_id="SPN-7415")
    candidates = (
        candidate("sponsor_id", "SPN-7415", confidence=0.46),
        candidate(
            "sponsor_id",
            "SPN-7196",
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            confidence=0.94,
            page=1,
        ),
        candidate(
            "visa_class",
            "TRANSIT-7",
            evidence_type=EvidenceType.SPONSOR_ATTESTATION,
            page=1,
        ),
    )
    assert apply_visible_candidate_complement(original, candidates) == original


def test_complete_intake_recovers_missing_sponsor_with_margin() -> None:
    original = row(sponsor_id="SPN-0000")
    candidates = (
        *complete_intake_sponsor("SPN-5138", confidence=0.79, page=1),
        *complete_intake_sponsor("SPN-5136", confidence=0.85, page=2),
    )

    repaired = apply_visible_candidate_complement(original, candidates)

    assert repaired.sponsor_id == "SPN-5136"
    assert any(
        repair.mechanism == "complete_intake_missing_sponsor"
        for repair in visible_candidate_complement_repairs(original, candidates)
    )
    assert apply_visible_candidate_complement(
        original,
        (
            *complete_intake_sponsor("SPN-5138", confidence=0.81, page=1),
            *complete_intake_sponsor("SPN-5136", confidence=0.85, page=2),
        ),
    ).sponsor_id == "SPN-0000"


def test_registry_repairs_only_invalid_arrival() -> None:
    candidates = (
        candidate(
            "arrival_date",
            "2026-05-03",
            evidence_type=EvidenceType.REGISTRY_EXTRACT,
            confidence=0.80,
        ),
    )
    invalid = row(arrival_date="2996-05-03")
    valid = row(arrival_date="2026-06-03")

    assert apply_visible_candidate_complement(
        invalid,
        candidates,
    ).arrival_date == "2026-05-03"
    assert apply_visible_candidate_complement(valid, candidates) == valid


def test_identity_hints_do_not_change_repair_result() -> None:
    original = row()
    first = candidate(
        "declared_purpose",
        "field repair",
        confidence=0.55,
        case_id_hint="MIB-000001",
        applicant_hint="Visible Applicant",
    )
    second = candidate(
        "declared_purpose",
        "field repair",
        confidence=0.55,
        case_id_hint="MIB-999999",
        applicant_hint="Someone Else",
    )

    assert apply_visible_candidate_complement(
        original,
        (first,),
    ) == apply_visible_candidate_complement(original, (second,))
