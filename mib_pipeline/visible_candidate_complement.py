"""Conservative output repairs from already-extracted visible OCR candidates.

The primary extractor sometimes reads a damaged field correctly, but the
general resolver keeps a stronger default or lower-precedence form value.
These repairs are deliberately narrow:

* they consume only visible OCR candidates already produced for the packet;
* they never inspect case IDs, filenames, hashes, page signatures, hidden
  text, answer keys, truth labels, or adjudication outcomes;
* they change extraction fields only and abstain on structural ambiguity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Iterable

from .extraction import CandidateEvidence, EvidenceType
from .models import PredictionRow


_BAD_CUES = frozenset({"strikethrough", "sample_denial_watermark"})
_MISSING_SPONSORS = frozenset({"", "unknown", "SPN-0000"})
_DEFAULT_PURPOSE = "reactor maintenance"
_PLAUSIBLE_ARRIVAL_MIN = date(2025, 1, 1)
_PLAUSIBLE_ARRIVAL_MAX = date(2026, 12, 31)


@dataclass(frozen=True)
class VisibleFieldRepair:
    """One inspectable, identity-free output-field proposal."""

    field_name: str
    value: str
    mechanism: str


def _safe(candidate: object) -> bool:
    return bool(
        isinstance(candidate, CandidateEvidence)
        and candidate.value is not None
        and candidate.legible
        and not candidate.superseded
        and candidate.source == "visible_ocr"
        and not (_BAD_CUES & set(candidate.visual_cues))
    )


def _structural(candidate: object) -> bool:
    """Accept an anchor-bearing candidate even when its value is unreadable."""

    return bool(
        isinstance(candidate, CandidateEvidence)
        and not candidate.superseded
        and candidate.source == "visible_ocr"
        and not (_BAD_CUES & set(candidate.visual_cues))
    )


def _unique_value(
    candidates: Iterable[CandidateEvidence],
) -> tuple[str, tuple[CandidateEvidence, ...]] | None:
    materialized = tuple(candidates)
    values = {str(candidate.value) for candidate in materialized}
    if len(values) != 1:
        return None
    return next(iter(values)), materialized


def _plausible_arrival(value: str) -> bool:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return _PLAUSIBLE_ARRIVAL_MIN <= parsed <= _PLAUSIBLE_ARRIVAL_MAX


def _purpose_repair(
    row: PredictionRow,
    candidates: tuple[CandidateEvidence, ...],
) -> VisibleFieldRepair | None:
    if row.declared_purpose != _DEFAULT_PURPOSE:
        return None
    eligible = tuple(
        candidate
        for candidate in candidates
        if _safe(candidate)
        and candidate.field_name == "declared_purpose"
        and candidate.evidence_type is EvidenceType.INTAKE_FORM
        and candidate.value != _DEFAULT_PURPOSE
        and candidate.ocr_confidence >= 0.50
    )
    resolved = _unique_value(eligible)
    if resolved is None:
        return None
    value, _ = resolved
    return VisibleFieldRepair(
        field_name="declared_purpose",
        value=value,
        mechanism="unique_nondefault_intake_purpose",
    )


def _structured_attestation_sponsor_repair(
    row: PredictionRow,
    candidates: tuple[CandidateEvidence, ...],
) -> VisibleFieldRepair | None:
    if re.fullmatch(r"SPN-\d{4}", row.sponsor_id) is None:
        return None
    intake_values = {
        candidate.value
        for candidate in candidates
        if _safe(candidate)
        and candidate.field_name == "sponsor_id"
        and candidate.evidence_type is EvidenceType.INTAKE_FORM
    }
    if row.sponsor_id not in intake_values:
        return None
    eligible = tuple(
        candidate
        for candidate in candidates
        if _safe(candidate)
        and candidate.field_name == "sponsor_id"
        and candidate.evidence_type is EvidenceType.SPONSOR_ATTESTATION
        and candidate.ocr_confidence >= 0.80
        and {
            structural.field_name
            for structural in candidates
            if _structural(structural)
            and structural.page_index == candidate.page_index
            and structural.evidence_type is EvidenceType.SPONSOR_ATTESTATION
        }
        >= {"sponsor_id", "visa_class", "declared_purpose"}
    )
    resolved = _unique_value(eligible)
    if resolved is None or resolved[0] == row.sponsor_id:
        return None
    return VisibleFieldRepair(
        field_name="sponsor_id",
        value=resolved[0],
        mechanism="structured_attestation_sponsor",
    )


def _complete_intake_sponsor_repair(
    row: PredictionRow,
    candidates: tuple[CandidateEvidence, ...],
) -> VisibleFieldRepair | None:
    if row.sponsor_id not in _MISSING_SPONSORS:
        return None
    eligible = tuple(
        candidate
        for candidate in candidates
        if _safe(candidate)
        and candidate.field_name == "sponsor_id"
        and candidate.evidence_type is EvidenceType.INTAKE_FORM
        and re.fullmatch(r"SPN-\d{4}", str(candidate.value))
        and candidate.ocr_confidence >= 0.75
        and {
            peer.field_name
            for peer in candidates
            if _safe(peer)
            and peer.page_index == candidate.page_index
            and peer.evidence_type is EvidenceType.INTAKE_FORM
        }
        >= {"visa_class", "sponsor_id", "arrival_date", "declared_purpose"}
    )
    if not eligible:
        return None
    by_value: dict[str, float] = {}
    for candidate in eligible:
        value = str(candidate.value)
        by_value[value] = max(by_value.get(value, 0.0), candidate.ocr_confidence)
    ranked = sorted(by_value.items(), key=lambda item: (-item[1], item[0]))
    best_value, best_confidence = ranked[0]
    runner_confidence = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_confidence < 0.84 or best_confidence - runner_confidence < 0.05:
        return None
    return VisibleFieldRepair(
        field_name="sponsor_id",
        value=best_value,
        mechanism="complete_intake_missing_sponsor",
    )


def _arrival_repair(
    row: PredictionRow,
    candidates: tuple[CandidateEvidence, ...],
) -> VisibleFieldRepair | None:
    if _plausible_arrival(row.arrival_date):
        return None
    eligible = tuple(
        candidate
        for candidate in candidates
        if _safe(candidate)
        and candidate.field_name == "arrival_date"
        and candidate.evidence_type is EvidenceType.REGISTRY_EXTRACT
        and candidate.ocr_confidence >= 0.75
        and _plausible_arrival(str(candidate.value))
    )
    resolved = _unique_value(eligible)
    if resolved is None:
        return None
    return VisibleFieldRepair(
        field_name="arrival_date",
        value=resolved[0],
        mechanism="valid_registry_arrival_for_invalid_output",
    )


def visible_candidate_complement_repairs(
    row: PredictionRow,
    candidates: Iterable[CandidateEvidence],
) -> tuple[VisibleFieldRepair, ...]:
    """Return deterministic repairs without reading identity or outcome fields."""

    materialized = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, CandidateEvidence)
    )
    repairs = (
        _purpose_repair(row, materialized),
        _structured_attestation_sponsor_repair(row, materialized),
        _complete_intake_sponsor_repair(row, materialized),
        _arrival_repair(row, materialized),
    )
    return tuple(repair for repair in repairs if repair is not None)


def apply_visible_candidate_complement(
    row: PredictionRow,
    candidates: Iterable[CandidateEvidence],
) -> PredictionRow:
    """Apply candidate-only extraction repairs while preserving other fields."""

    repairs = visible_candidate_complement_repairs(row, candidates)
    if not repairs:
        return row
    changes = {repair.field_name: repair.value for repair in repairs}
    return replace(row, **changes)
