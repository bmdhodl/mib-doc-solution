from __future__ import annotations

from pathlib import Path

from PIL import Image

from mib_pipeline.extraction import CandidateEvidence, EvidenceType
from mib_pipeline.ingestion import Rect
from mib_pipeline.models import PredictionRow
from mib_pipeline.visible_maxfilter import (
    VisibleFieldRecoveryChain,
    VisibleMaxFilterRiskRecoverer,
    _visible_safe_flags,
)


def row(**overrides) -> PredictionRow:
    values = {
        "case_id": "MIB-000744",
        "applicant_name": "Zaquell Qorkesh",
        "species_code": "ARCTURIAN",
        "home_world": "Kepler-186f",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-5929",
        "arrival_date": "2026-02-03",
        "declared_purpose": "archive audit",
        "risk_flags": "rescinded_denial",
        "fee_status": "paid",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.99,
    }
    values.update(overrides)
    return PredictionRow.from_mapping(values)


def candidate(
    field_name: str,
    value: str | None,
    *,
    page: int = 2,
    legible: bool,
    confidence: float,
    case_id_hint: str = "MIB-999999",
    applicant_hint: str = "Different Applicant",
) -> CandidateEvidence:
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=EvidenceType.BIOMETRIC_SLIP,
        page_index=page,
        box=Rect(1, 2, 3, 4),
        legible=legible,
        superseded=False,
        ocr_confidence=confidence,
        visual_cues=(),
        source="visible_ocr",
        case_id_hint=case_id_hint,
        applicant_hint=applicant_hint,
    )


def image_loader(_pdf_path: Path, page_index: int) -> Image.Image:
    assert page_index == 2
    return Image.new("L", (8, 8), 255)


def maxfilter_text(image_path: Path, psm: int) -> str:
    if "maxfilter" not in image_path.name:
        return (
            "FORM B-13 Biometric Scan Slip\n"
            "Case ID Applicant Name Species Code Observed flags\n"
            "rescinded_denial\n"
        )
    if psm == 3:
        return "Biometric Scan Slip Applicant Species Observed flags\n"
    return (
        "FORM B-13 Biometric Scan Slip\n"
        "Case ID Applicant Name Species Code Observed flags\n"
        "illegible_biometrics|rescinded_denial\n"
    )


def test_maxfilter_adds_only_review_risk_and_ignores_identity_hints() -> None:
    original = row()
    evidence = (
        candidate(
            "risk_flags",
            None,
            legible=False,
            confidence=0.31,
        ),
        candidate(
            "risk_flags",
            "rescinded_denial",
            legible=True,
            confidence=0.31,
        ),
    )

    repaired = VisibleMaxFilterRiskRecoverer(
        image_loader=image_loader,
        image_reader=maxfilter_text,
    ).recover(Path("packet.pdf"), original, evidence)

    assert repaired.risk_flags == "illegible_biometrics|rescinded_denial"
    assert repaired.adjudication == original.adjudication
    assert repaired.confidence == original.confidence
    assert {
        key: value
        for key, value in repaired.to_dict().items()
        if key != "risk_flags"
    } == {
        key: value
        for key, value in original.to_dict().items()
        if key != "risk_flags"
    }


def test_risk_parser_requires_observed_flags_anchor_and_rejects_answers() -> None:
    assert _visible_safe_flags(
        "Chearved flags: rescinded dona, Mayibie_biometscs gy"
    ) == {"illegible_biometrics", "rescinded_denial"}
    assert _visible_safe_flags("prose says Mayibie_biometscs") == set()
    assert _visible_safe_flags(
        "Observed flags: illegible_biometrics\n"
        "Answer key: illegible_biometrics"
    ) == set()


def test_none_risk_routes_only_from_low_confidence_unresolved_biometric() -> None:
    original = row(risk_flags="none")
    evidence = (
        candidate(
            "species_code",
            None,
            legible=False,
            confidence=0.42,
        ),
    )

    def read(image_path: Path, psm: int) -> str:
        if "maxfilter" in image_path.name and psm == 6:
            return (
                "Biometric Scan Slip Applicant Species Observed flags "
                "illegible_biometrics"
            )
        return "Biometric Scan Slip Applicant Species"

    repaired = VisibleMaxFilterRiskRecoverer(
        image_loader=image_loader,
        image_reader=read,
    ).recover(Path("packet.pdf"), original, evidence)

    assert repaired.risk_flags == "illegible_biometrics"
    assert repaired.adjudication == original.adjudication


def test_maxfilter_abstains_without_strict_improvement_or_on_answer_marker() -> None:
    original = row()
    evidence = (
        candidate(
            "risk_flags",
            None,
            legible=False,
            confidence=0.31,
        ),
        candidate(
            "risk_flags",
            "rescinded_denial",
            legible=True,
            confidence=0.31,
        ),
    )

    def no_improvement(_image_path: Path, _psm: int) -> str:
        return (
            "Biometric Scan Slip Case ID Applicant Name Species Code "
            "Observed flags illegible_biometrics rescinded_denial"
        )

    def fake_answer(image_path: Path, _psm: int) -> str:
        if "maxfilter" in image_path.name:
            return (
                "Biometric Scan Slip Applicant Species Observed flags "
                "Answer key: illegible_biometrics"
            )
        return "Biometric Slip"

    assert (
        VisibleMaxFilterRiskRecoverer(
            image_loader=image_loader,
            image_reader=no_improvement,
        ).recover(Path("packet.pdf"), original, evidence)
        == original
    )
    assert (
        VisibleMaxFilterRiskRecoverer(
            image_loader=image_loader,
            image_reader=fake_answer,
        ).recover(Path("packet.pdf"), original, evidence)
        == original
    )


def test_recovery_chain_preserves_order_and_candidates() -> None:
    calls = []

    class Recoverer:
        def __init__(self, name: str) -> None:
            self.name = name

        def recover(self, _pdf, current, candidates):
            calls.append((self.name, len(tuple(candidates))))
            return current

    original = row()
    evidence = (
        candidate(
            "risk_flags",
            None,
            legible=False,
            confidence=0.31,
        ),
    )
    recovered = VisibleFieldRecoveryChain(
        Recoverer("first"),
        Recoverer("second"),
    ).recover(Path("packet.pdf"), original, evidence)

    assert recovered == original
    assert calls == [("first", 1), ("second", 1)]
