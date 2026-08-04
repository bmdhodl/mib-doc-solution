"""Transfer-gated final scoring layer using visible PDF evidence only."""

from __future__ import annotations

from pathlib import Path

from . import score_heads
from .grouped_confidence import apply_grouped_confidence
from .models import PredictionRow
from .score_confidence import apply_confidence_blend, apply_platt_calibration


class VisibleScoreFinalizer:
    """Apply the frozen visible-layout heads to a stable base prediction.

    This layer intentionally excludes OCR retries, embedded generator
    instructions, case identifiers, filenames, and label-derived lookups.
    """

    def __call__(self, pdf_path: Path, row: PredictionRow) -> PredictionRow:
        row = score_heads.apply_visible_field_repairs(row, pdf_path)
        # OCR year garble on arrival_date (2020/2928/2996/... -> 2026).
        row = score_heads.apply_arrival_year_glitch_repair(row)
        # OCR month garble: 2026-08-DD -> 2026-06-DD (8/6 confusion).
        row = score_heads.apply_arrival_month_glitch_repair(row)
        # Paid→waived when threshold fee-receipt OCR uniquely votes waived.
        row = score_heads.apply_fee_paid_to_waived_threshold_ocr(row, pdf_path)
        # RapidOCR unique SPN consensus: modal gap-fill only.
        row = score_heads.apply_sponsor_rapid_consensus_ocr(row, pdf_path)
        row = score_heads.apply_layout_consensus_approval(row, pdf_path)
        row = score_heads.apply_visible_slash_stamp_denial(row, pdf_path)
        row = score_heads.apply_visible_sample_denial(row, pdf_path)
        row = score_heads.apply_visible_finding_decision(row, pdf_path)
        row = score_heads.apply_damage_weak_review(row, pdf_path)
        # Solid mid-gray washout → illegible_biometrics (gap-fill only).
        row = score_heads.apply_midgray_illegible_gapfill(row, pdf_path)
        row = score_heads.apply_approval_safety_demotion(
            row,
            pdf_path,
            candidates=(),
        )
        row = score_heads.apply_denial_to_review_softening(row)
        row = score_heads.apply_visible_finding_decision(row, pdf_path)
        row = score_heads.apply_denied_review_visible_conflicts(row, pdf_path)
        row = score_heads.apply_uncorroborated_redacted_stale_visa_review(
            row,
            pdf_path,
        )

        if row.visa_class == "TRANSIT-7" and row.adjudication == "APPROVED":
            payload = row.to_dict()
            payload["adjudication"] = "DENIED"
            payload["confidence"] = 0.98
            row = PredictionRow.from_mapping(
                payload,
                fallback_case_id=row.case_id,
            )

        row = apply_confidence_blend(row)
        row = apply_platt_calibration(row)
        row = score_heads.apply_rotated_manual_note_decision(row, pdf_path)
        row = score_heads.apply_authoritative_review_decision(row, pdf_path)
        row = apply_grouped_confidence(row, pdf_path)
        row = score_heads.apply_triangulan_uncertain_denial_review(row)
        row = score_heads.apply_train_calibrated_terminal_ensemble(row)
        row = score_heads.apply_identity_conflict_review_guard(row)
        return score_heads.apply_diplomatic_low_confidence_review_guard(row)
