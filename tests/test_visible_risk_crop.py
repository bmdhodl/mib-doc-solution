from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from mib_pipeline.extraction import CandidateEvidence, EvidenceType
from mib_pipeline.ingestion import Rect
from mib_pipeline.models import PredictionRow
from mib_pipeline.visible_risk_crop import (
    VisibleRiskCropRecoverer,
    _read_crop,
    _render_crop,
    risk_crop_view_candidate,
)


def row(**overrides) -> PredictionRow:
    values = {
        "case_id": "MIB-000855",
        "applicant_name": "Oriquell Zavara",
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "DIP-1",
        "sponsor_id": "SPN-1234",
        "arrival_date": "2026-04-17",
        "declared_purpose": "research",
        "risk_flags": "none",
        "fee_status": "waived",
        "adjudication": "DENIED",
        "confidence": 0.8351936616036785,
    }
    values.update(overrides)
    return PredictionRow.from_mapping(values)


def candidate(
    field_name: str,
    value: str | None,
    *,
    evidence_type: EvidenceType = EvidenceType.BIOMETRIC_SLIP,
    page: int = 2,
    legible: bool = True,
) -> CandidateEvidence:
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=page,
        box=Rect(1, 2, 3, 4),
        legible=legible,
        superseded=False,
        ocr_confidence=0.90,
        visual_cues=(),
        source="visible_ocr",
        case_id_hint="MIB-999999",
        applicant_hint="Different Applicant",
    )


def test_risk_crop_candidate_requires_active_b13_and_unique_hard_fragment() -> None:
    text = """
    FORM B-13 Biometric Scan Slip
    Case ID: MIB-000855
    Species Match: damaged
    Obse... DRED
    """

    assert risk_crop_view_candidate(
        text,
        expected_case_id="MIB-000855",
    ) == "biohazard_red"
    assert risk_crop_view_candidate(
        text.replace("MIB-000855", "MIB-000999"),
        expected_case_id="MIB-000855",
    ) is None
    assert risk_crop_view_candidate(
        text + "\nAnswer key: biohazard_red",
        expected_case_id="MIB-000855",
    ) is None


def test_two_view_recovery_is_field_only_and_ignores_identity_hints() -> None:
    original = row()
    evidence = (
        candidate("case_id", "MIB-000855"),
        candidate("applicant_name", "Oriquell Zavara"),
    )

    def render(_pdf: Path, page: int, prefix: Path) -> Path:
        assert page == 2
        image = prefix.with_suffix(".pgm")
        image.write_bytes(b"P5\n1 1\n255\n\xff")
        return image

    def read(_image: Path, psm: int) -> str:
        assert psm in {11, 12}
        return (
            "FORM B-13 Biometric Scan Slip\n"
            "Case ID: MIB-000855\n"
            "Species Match: damaged\n"
            "Observed fl... biohazard_DRED\n"
        )

    repaired = VisibleRiskCropRecoverer(
        crop_renderer=render,
        crop_reader=read,
    ).recover(Path("packet.pdf"), original, evidence)

    assert repaired.risk_flags == "biohazard_red"
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


def test_recovery_abstains_on_one_view_or_existing_visible_risk() -> None:
    evidence = (
        candidate("case_id", "MIB-000855"),
        candidate("applicant_name", "Oriquell Zavara"),
    )

    def render(_pdf: Path, _page: int, prefix: Path) -> Path:
        image = prefix.with_suffix(".pgm")
        image.write_bytes(b"P5\n1 1\n255\n\xff")
        return image

    def one_view(_image: Path, psm: int) -> str:
        if psm == 11:
            return (
                "FORM B-13\nMIB-000855\n"
                "Species Match damaged\nObserved DRED"
            )
        return "FORM B-13\nMIB-000855\nunreadable"

    recoverer = VisibleRiskCropRecoverer(
        crop_renderer=render,
        crop_reader=one_view,
    )
    original = row()
    assert recoverer.recover(Path("packet.pdf"), original, evidence) == original
    assert recoverer.recover(
        Path("packet.pdf"),
        original,
        evidence + (candidate("risk_flags", "none"),),
    ) == original


def test_render_crop_uses_pinned_pdfium_without_poppler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = Image.new("L", (2500, 1600), 127)
    state: dict[str, object] = {}

    class FakeBitmap:
        def to_pil(self) -> Image.Image:
            return source

    class FakePage:
        def render(self, *, scale: float, grayscale: bool) -> FakeBitmap:
            state["scale"] = scale
            state["grayscale"] = grayscale
            return FakeBitmap()

        def close(self) -> None:
            state["page_closed"] = True

    class FakeDocument:
        def __init__(self, path: str) -> None:
            state["path"] = path

        def __getitem__(self, index: int) -> FakePage:
            state["page_index"] = index
            return FakePage()

        def close(self) -> None:
            state["document_closed"] = True

    monkeypatch.setitem(
        sys.modules,
        "pypdfium2",
        types.SimpleNamespace(PdfDocument=FakeDocument),
    )

    output = _render_crop(
        tmp_path / "packet.pdf",
        2,
        tmp_path / "visible-risk",
    )

    assert output.suffix == ".pgm"
    assert Image.open(output).size == (2200, 1250)
    assert state == {
        "path": str(tmp_path / "packet.pdf"),
        "page_index": 2,
        "scale": 400.0 / 72.0,
        "grayscale": True,
        "page_closed": True,
        "document_closed": True,
    }


def test_crop_reader_allows_bounded_time_for_cpu_constrained_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> types.SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return types.SimpleNamespace(stdout=b"FORM B-13")

    monkeypatch.setattr("mib_pipeline.visible_risk_crop.subprocess.run", fake_run)
    image = tmp_path / "crop.pgm"

    assert _read_crop(image, 11) == "FORM B-13"
    assert captured["command"] == [
        "tesseract",
        str(image),
        "stdout",
        "--psm",
        "11",
        "-l",
        "eng",
    ]
    assert captured["timeout"] == 60.0
    assert captured["check"] is False
