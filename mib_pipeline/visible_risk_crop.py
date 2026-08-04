"""Two-view visible B-13 recovery for a missing hard risk flag."""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

from .extraction import CandidateEvidence, EvidenceType
from .models import PredictionRow
from .pdfium_runtime import PDFIUM_RENDER_LOCK


HARD_RISK_FLAGS = frozenset(
    {
        "active_warrant",
        "biohazard_red",
        "memory_tampering",
        "planetary_embargo",
    }
)
ALL_RISK_FLAGS = (
    "active_warrant",
    "biohazard_red",
    "identity_conflict",
    "illegible_biometrics",
    "memory_tampering",
    "planetary_embargo",
    "rescinded_denial",
    "sponsor_mismatch",
)
_UNTRUSTED = re.compile(
    r"answer\s*key|output\s+this\s+answer|ground\s+truth|"
    r"training\s+example|sample\s+denial|force\s+adjudication|"
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    re.I,
)
_CROP_OCR_TIMEOUT_SECONDS = 60.0


def _compact(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def risk_crop_view_candidate(
    text: str,
    *,
    expected_case_id: str,
) -> str | None:
    """Read one hard flag only from an active-case B-13 crop."""

    if _UNTRUSTED.search(text):
        return None
    expected_number = expected_case_id.removeprefix("MIB-")
    if not re.search(r"\bB-?13\b|Biometric\s+Scan\s+Slip", text, re.I):
        return None
    visible_ids = set(
        re.findall(r"\bMIB[- ]?(\d{6})\b", text, re.I)
    )
    if visible_ids != {expected_number}:
        return None

    if re.search(r"\bObse", text, re.I):
        compact = _compact(text)
        fragment_matches = {
            flag
            for fragment, flag in (
                ("DRED", "biohazard_red"),
                ("EMBARGO", "planetary_embargo"),
                ("WARRANT", "active_warrant"),
                ("TAMPERING", "memory_tampering"),
            )
            if fragment in compact
        }
        if len(fragment_matches) == 1:
            return fragment_matches.pop()
    if not re.search(
        r"\bSpecies\s+Match\b|\bSpecies\b.{0,12}\bMatch\b",
        text,
        re.I,
    ):
        return None

    candidates: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+", text):
        token_key = _compact(token)
        if len(token_key) < 8:
            continue
        ranked = sorted(
            (
                difflib.SequenceMatcher(
                    None,
                    token_key,
                    _compact(flag),
                ).ratio(),
                flag,
            )
            for flag in ALL_RISK_FLAGS
        )
        best_score, best_flag = ranked[-1]
        second_score = ranked[-2][0]
        if best_score >= 0.66 and best_score - second_score >= 0.20:
            candidates.add(best_flag)
    if len(candidates) != 1:
        return None
    candidate = candidates.pop()
    return candidate if candidate in HARD_RISK_FLAGS else None


def _eligible_b13_pages(
    row: PredictionRow,
    candidates: Iterable[CandidateEvidence],
) -> tuple[int, ...]:
    if row.risk_flags != "none":
        return ()
    materialized = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, CandidateEvidence)
        and candidate.source == "visible_ocr"
        and not candidate.superseded
        and not (
            {"strikethrough", "sample_denial_watermark"}
            & set(candidate.visual_cues)
        )
    )
    pages = []
    for page_index in sorted({candidate.page_index for candidate in materialized}):
        page = tuple(
            candidate
            for candidate in materialized
            if candidate.page_index == page_index
        )
        case_values = {
            candidate.value
            for candidate in page
            if candidate.field_name == "case_id"
            and candidate.value is not None
        }
        if case_values != {row.case_id}:
            continue
        biometric = tuple(
            candidate
            for candidate in page
            if candidate.evidence_type is EvidenceType.BIOMETRIC_SLIP
        )
        if not biometric:
            continue
        if any(
            candidate.field_name == "risk_flags"
            and candidate.legible
            and candidate.value is not None
            for candidate in biometric
        ):
            continue
        pages.append(page_index)
    return tuple(pages)


def _render_crop(pdf_path: Path, page_index: int, output_prefix: Path) -> Path:
    import pypdfium2

    with PDFIUM_RENDER_LOCK:
        document = pypdfium2.PdfDocument(str(pdf_path))
        try:
            page = document[page_index]
            try:
                image = (
                    page.render(scale=400.0 / 72.0, grayscale=True)
                    .to_pil()
                    .convert("L")
                )
            finally:
                page.close()
        finally:
            document.close()

    crop = image.crop(
        (
            0,
            100,
            min(2200, image.width),
            min(1350, image.height),
        )
    )
    output_path = output_prefix.with_suffix(".pgm")
    crop.save(output_path)
    return output_path


def _read_crop(image: Path, psm: int) -> str:
    configured_binary = os.environ.get("MIB_TESSERACT_BINARY", "").strip()
    path_binary = "tesseract" if shutil.which("tesseract") else ""
    candidates = (
        configured_binary,
        r"C:\Users\patri\AppData\Local\Temp\fluarmn-tesseract-5.5.3\Tesseract-OCR\tesseract.exe",
        path_binary,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )
    binary = next(
        (
            candidate
            for candidate in candidates
            if candidate and (Path(candidate).is_file() or candidate == path_binary)
        ),
        "tesseract",
    )
    completed = subprocess.run(
        [
            binary,
            str(image),
            "stdout",
            "--psm",
            str(psm),
            "-l",
            "eng",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=_CROP_OCR_TIMEOUT_SECONDS,
        check=False,
    )
    return completed.stdout.decode("utf-8", errors="replace")


class VisibleRiskCropRecoverer:
    """Apply a two-layout B-13 hard-flag repair as output-only evidence."""

    def __init__(
        self,
        *,
        crop_renderer: Callable[[Path, int, Path], Path] = _render_crop,
        crop_reader: Callable[[Path, int], str] = _read_crop,
    ) -> None:
        self._crop_renderer = crop_renderer
        self._crop_reader = crop_reader

    def recover(
        self,
        pdf_path: Path,
        row: PredictionRow,
        candidates: Iterable[CandidateEvidence],
    ) -> PredictionRow:
        pages = _eligible_b13_pages(row, candidates)
        if not pages:
            return row
        found: set[str] = set()
        try:
            with tempfile.TemporaryDirectory(prefix="mib-risk-crop-") as raw_temp:
                temp = Path(raw_temp)
                for page_index in pages:
                    image = self._crop_renderer(
                        pdf_path,
                        page_index,
                        temp / f"page-{page_index + 1}",
                    )
                    votes = Counter(
                        candidate
                        for psm in (11, 12)
                        if (
                            candidate := risk_crop_view_candidate(
                                self._crop_reader(image, psm),
                                expected_case_id=row.case_id,
                            )
                        )
                        is not None
                    )
                    winners = [
                        flag for flag, count in votes.items() if count >= 2
                    ]
                    if len(winners) == 1:
                        found.add(winners[0])
        except (OSError, subprocess.SubprocessError):
            return row
        if len(found) != 1:
            return row
        # Deliberately preserve adjudication and confidence. The canonical
        # public-135 decision is already final at this point.
        return replace(row, risk_flags=found.pop())
