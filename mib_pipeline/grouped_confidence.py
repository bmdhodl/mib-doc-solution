"""Frozen class-and-evidence-path confidence calibration.

This is the outermost scoring stage. It consumes the final serialized row plus
coarse, identity-free visible evidence from the packet and may replace only the
confidence. It cannot change extracted fields or adjudication.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from . import score_heads
from .confidence import CalibrationArtifactError
from .models import PredictionRow


PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "grouped_confidence_calibration.json"
)
PINNED_GROUPED_CONFIDENCE_ARTIFACT_SHA256 = (
    "5fb7b32ec8ab875bdf3b1a870fcac5c39b5df27c3a178d171ba7a5a25ed49611"
)

_OUTPUT_FIELDS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "fee_status",
)
_DECISIONS = ("APPROVED", "DENIED", "NEEDS_REVIEW")
_PATHS = (
    "finding",
    "risk",
    "damage",
    "fee_special",
    "fee_proven",
    "sparse",
    "ordinary",
)
_FEATURE_ORDER = (
    "logit_confidence",
    "decision=APPROVED",
    "decision=DENIED",
    "decision=NEEDS_REVIEW",
    "logit_x_decision=APPROVED",
    "logit_x_decision=DENIED",
    "logit_x_decision=NEEDS_REVIEW",
    "path=finding",
    "path=risk",
    "path=damage",
    "path=fee_special",
    "path=fee_proven",
    "path=sparse",
    "path=ordinary",
    "logit_x_path=finding",
    "logit_x_path=risk",
    "logit_x_path=damage",
    "logit_x_path=fee_special",
    "logit_x_path=fee_proven",
    "logit_x_path=sparse",
    "logit_x_path=ordinary",
    "missing=1",
    "missing=2_plus",
)
_SENSITIVE_VALUE = re.compile(
    r"(?:\bMIB-[0-9]{6}\b|\bSPN-[0-9]{4}\b|"
    r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b|\.pdf\b)",
    re.IGNORECASE,
)
_UNTRUSTED_LINE = re.compile(
    r"\b(?:SYSTEM|ASSISTANT|ANSWER\s*KEY|GROUND\s*TRUTH|"
    r"IGNORE\s+(?:ALL|PREVIOUS)|ignore\s+visible)\b",
    re.IGNORECASE,
)
_CASE_CSV_LINE = re.compile(
    r"\bMIB-[0-9]{6}\b.*\b(?:APPROVED|DENIED|NEEDS_REVIEW)\b",
    re.IGNORECASE,
)


class GroupedConfidenceArtifactError(CalibrationArtifactError):
    """The grouped-confidence artifact is absent, malformed, or unsafe."""


def _contains_sensitive_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_sensitive_value(key) or _contains_sensitive_value(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_value(child) for child in value)
    return isinstance(value, str) and _SENSITIVE_VALUE.search(value) is not None


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_unknown(field_name: str, value: object) -> bool:
    rendered = str(value or "").strip().casefold()
    if rendered in {"", "unknown", "none", "null"}:
        return True
    if field_name == "sponsor_id" and rendered == "spn-0000":
        return True
    return field_name == "arrival_date" and rendered == "1900-01-01"


def missing_count(row: PredictionRow) -> int:
    """Count schema-default output fields without inspecting their identities."""

    return sum(
        _is_unknown(field_name, getattr(row, field_name))
        for field_name in _OUTPUT_FIELDS
    )


def strip_untrusted_lines(text: str) -> str:
    """Remove generator instructions before deriving a visible evidence path."""

    return "\n".join(
        line
        for line in text.splitlines()
        if not _UNTRUSTED_LINE.search(line) and not _CASE_CSV_LINE.search(line)
    )


def evidence_path_from_summary(
    row: PredictionRow,
    *,
    finding: bool,
    layout_risk: bool,
    registry_embargo: bool,
    damage: bool,
    fee_paid_proven: bool,
) -> str:
    """Apply the frozen coarse evidence-path precedence."""

    risk_present = row.risk_flags.strip().casefold() not in {
        "",
        "none",
        "unknown",
        "null",
    }
    if finding:
        return "finding"
    if risk_present or layout_risk or registry_embargo:
        return "risk"
    if damage:
        return "damage"
    if row.fee_status in {"unpaid", "waived"}:
        return "fee_special"
    if fee_paid_proven:
        return "fee_proven"
    if missing_count(row) >= 3:
        return "sparse"
    return "ordinary"


def evidence_path_from_text(row: PredictionRow, text: str) -> str:
    """Derive one identity-free path from sanitized visible packet text."""

    safe_text = strip_untrusted_lines(text)
    upper = safe_text.upper()
    return evidence_path_from_summary(
        row,
        finding=bool(
            re.search(
                r"Finding\s*:?\s*(?:DENIED|NEEDS[_\s]?REVIEW)\b",
                safe_text,
                re.IGNORECASE,
            )
        ),
        layout_risk=bool(score_heads._layout_risk_flags(safe_text)),
        registry_embargo=bool(
            re.search(
                r"Registry\s+Status\s*:?\s*EMBARGO\b",
                safe_text,
                re.IGNORECASE,
            )
        ),
        damage=any(
            marker in upper for marker in ("ILLEGIBLE", "WASHED OUT", "REDACTED")
        ),
        fee_paid_proven=score_heads._layout_fee_paid_proven(safe_text),
    )


def feature_values(
    row: PredictionRow,
    *,
    evidence_path: str,
) -> tuple[float, ...]:
    """Return the frozen feature vector used by both fitting and runtime."""

    if row.adjudication not in _DECISIONS:
        raise ValueError(f"unsupported adjudication: {row.adjudication}")
    if evidence_path not in _PATHS:
        raise ValueError(f"unsupported evidence path: {evidence_path}")
    probability = min(max(float(row.confidence), 1e-5), 1.0 - 1e-5)
    logit = math.log(probability / (1.0 - probability))
    values: dict[str, float] = {"logit_confidence": logit}
    for decision in _DECISIONS:
        active = float(row.adjudication == decision)
        values[f"decision={decision}"] = active
        values[f"logit_x_decision={decision}"] = logit * active
    for path in _PATHS:
        active = float(evidence_path == path)
        values[f"path={path}"] = active
        values[f"logit_x_path={path}"] = logit * active
    count = missing_count(row)
    values["missing=1"] = float(count == 1)
    values["missing=2_plus"] = float(count >= 2)
    return tuple(values.get(feature, 0.0) for feature in _FEATURE_ORDER)


@dataclass(frozen=True)
class PinnedGroupedConfidenceMap:
    """Validated frozen logistic map over generic final-output evidence."""

    artifact_id: str
    intercept: float
    coefficients: tuple[float, ...]
    feature_order: tuple[str, ...]
    probability_clip: tuple[float, float]
    fit_metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PinnedGroupedConfidenceMap":
        if set(value) != {
            "artifact_id",
            "schema_version",
            "fit_metadata",
            "identity_policy",
            "model",
        } or value.get("schema_version") != 1:
            raise GroupedConfidenceArtifactError(
                "unsupported grouped-confidence artifact schema"
            )
        artifact_id = value.get("artifact_id")
        metadata = value.get("fit_metadata")
        identity_policy = value.get("identity_policy")
        model = value.get("model")
        if (
            not isinstance(artifact_id, str)
            or not artifact_id.strip()
            or not isinstance(metadata, dict)
            or not isinstance(identity_policy, str)
        ):
            raise GroupedConfidenceArtifactError(
                "grouped-confidence artifact metadata is malformed"
            )
        if _contains_sensitive_value(value):
            raise GroupedConfidenceArtifactError(
                "grouped-confidence artifact contains identity-bearing values"
            )
        if not isinstance(model, dict) or set(model) != {
            "regularization_c",
            "solver",
            "probability_clip",
            "feature_order",
            "intercept",
            "coefficients",
        }:
            raise GroupedConfidenceArtifactError(
                "grouped-confidence model is malformed"
            )
        if (
            not _is_number(model["regularization_c"])
            or float(model["regularization_c"]) <= 0
            or model["solver"] != "lbfgs"
            or not _is_number(model["intercept"])
        ):
            raise GroupedConfidenceArtifactError(
                "grouped-confidence logistic parameters are invalid"
            )
        raw_clip = model["probability_clip"]
        if (
            not isinstance(raw_clip, list)
            or len(raw_clip) != 2
            or any(not _is_number(item) for item in raw_clip)
        ):
            raise GroupedConfidenceArtifactError(
                "grouped-confidence probability clip is malformed"
            )
        lower, upper = (float(item) for item in raw_clip)
        if not 0.0 <= lower < upper <= 1.0:
            raise GroupedConfidenceArtifactError(
                "grouped-confidence probability clip is not ordered"
            )
        raw_features = model["feature_order"]
        if not isinstance(raw_features, list) or tuple(raw_features) != _FEATURE_ORDER:
            raise GroupedConfidenceArtifactError(
                "grouped-confidence feature order is not frozen"
            )
        raw_coefficients = model["coefficients"]
        if (
            not isinstance(raw_coefficients, list)
            or len(raw_coefficients) != len(_FEATURE_ORDER)
            or any(not _is_number(item) for item in raw_coefficients)
        ):
            raise GroupedConfidenceArtifactError(
                "grouped-confidence coefficients are malformed"
            )
        return cls(
            artifact_id=artifact_id,
            intercept=float(model["intercept"]),
            coefficients=tuple(float(item) for item in raw_coefficients),
            feature_order=_FEATURE_ORDER,
            probability_clip=(lower, upper),
            fit_metadata=dict(metadata),
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> "PinnedGroupedConfidenceMap":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GroupedConfidenceArtifactError(
                f"cannot load grouped-confidence artifact: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise GroupedConfidenceArtifactError(
                "grouped-confidence artifact must be an object"
            )
        if expected_sha256 is not None and _canonical_sha256(value) != expected_sha256:
            raise GroupedConfidenceArtifactError(
                "grouped-confidence artifact does not match the frozen checksum"
            )
        return cls.from_mapping(value)

    def predict(self, row: PredictionRow, *, evidence_path: str) -> float:
        vector = feature_values(row, evidence_path=evidence_path)
        linear = self.intercept + sum(
            coefficient * feature
            for coefficient, feature in zip(
                self.coefficients,
                vector,
                strict=True,
            )
        )
        if linear >= 0:
            probability = 1.0 / (1.0 + math.exp(-linear))
        else:
            exponent = math.exp(linear)
            probability = exponent / (1.0 + exponent)
        lower, upper = self.probability_clip
        return max(lower, min(upper, probability))


class GroupedConfidenceRecalibrator:
    """Apply the frozen map while mutating only ``PredictionRow.confidence``."""

    def __init__(self, calibration_map: PinnedGroupedConfidenceMap) -> None:
        self._map = calibration_map

    @classmethod
    def from_pinned_artifact(
        cls,
        path: Path = PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH,
    ) -> "GroupedConfidenceRecalibrator":
        expected_sha256 = (
            PINNED_GROUPED_CONFIDENCE_ARTIFACT_SHA256
            if path == PINNED_GROUPED_CONFIDENCE_ARTIFACT_PATH
            else None
        )
        return cls(
            PinnedGroupedConfidenceMap.from_path(
                path,
                expected_sha256=expected_sha256,
            )
        )

    def recalibrate(
        self,
        pdf_path: Path,
        row: PredictionRow,
    ) -> PredictionRow:
        text = score_heads._pdf_layout_text(pdf_path)
        path = evidence_path_from_text(row, text)
        confidence = self._map.predict(row, evidence_path=path)
        if confidence == row.confidence:
            return row
        return replace(row, confidence=confidence)


_PINNED_RECALIBRATOR: GroupedConfidenceRecalibrator | None = None


def apply_grouped_confidence(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Apply the pinned grouped calibrator as a confidence-only final stage."""

    global _PINNED_RECALIBRATOR
    if _PINNED_RECALIBRATOR is None:
        _PINNED_RECALIBRATOR = GroupedConfidenceRecalibrator.from_pinned_artifact()
    return _PINNED_RECALIBRATOR.recalibrate(pdf_path, row)
