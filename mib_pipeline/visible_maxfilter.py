"""Native-image MaxFilter retry for narrowly unresolved biometric risks.

This is an output-only, visible-pixel retry. It never reads PDF text, case
identity, applicant identity, signatures, or answer-key content. The retry may
only add one of two review-only risk flags and never changes adjudication or
confidence.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable, Protocol

from PIL import Image, ImageFilter

from .extraction import CandidateEvidence, EvidenceType
from .models import PredictionRow
from .pdfium_runtime import PDFIUM_RENDER_LOCK


SAFE_REVIEW_FLAGS = frozenset({"illegible_biometrics", "rescinded_denial"})
LOW_CONFIDENCE_MAXIMUM = 0.45
MINIMUM_IMAGE_EDGE = 400
DOMAIN_WORDS = frozenset(
    {
        "applicant",
        "biometric",
        "case",
        "code",
        "eyes",
        "flags",
        "id",
        "match",
        "mib",
        "name",
        "observed",
        "only",
        "scan",
        "slip",
        "species",
    }
)
BOILERPLATE_RE = re.compile(
    r"Packet MIB-\d{6}\s*/\s*page \d+|"
    r"Synthetic hiring challenge document|"
    r"MIB-\d{6}\s*\|\s*MIB Eyes Only",
    re.IGNORECASE,
)
ANSWER_MARKER_RE = re.compile(
    r"answer\s*key|output\s+this\s+answer|ground\s+truth|"
    r"ignore\s+(?:all\s+)?previous\s+instructions|"
    r"ignore\s+visible|force\s+adjudication|barcode\s+payload",
    re.IGNORECASE,
)


class FieldRecoverer(Protocol):
    def recover(
        self,
        pdf_path: Path,
        row: PredictionRow,
        candidates: Iterable[CandidateEvidence],
    ) -> PredictionRow:
        """Return an unchanged or field-only repaired row."""


def _parse_flags(value: str) -> frozenset[str]:
    return frozenset(
        token.strip().casefold()
        for token in value.split("|")
        if token.strip() and token.strip().casefold() != "none"
    )


def _word_quality(text: str) -> tuple[int, int]:
    useful = BOILERPLATE_RE.sub("", text).strip()
    known_words = sum(
        1
        for token in re.findall(r"[A-Za-z]{2,}", text)
        if token.casefold() in DOMAIN_WORDS
    )
    return known_words, len(useful)


def _visible_safe_flags(text: str) -> frozenset[str]:
    if ANSWER_MARKER_RE.search(text):
        return frozenset()
    lowered = text.casefold()
    found = {flag for flag in SAFE_REVIEW_FLAGS if flag in lowered}
    for line in text.splitlines():
        match = re.match(
            r"[^A-Za-z]*([A-Za-z]+\s?[A-Za-z]+)\s*[:;]\s*(.+)",
            line,
        )
        if (
            match is None
            or SequenceMatcher(
                None,
                match.group(1).casefold(),
                "observed flags",
            ).ratio()
            < 0.65
        ):
            continue
        for raw_token in re.split(r"[,|]", match.group(2)):
            token = re.sub(
                r"[^a-z_ ]",
                "",
                raw_token.strip().casefold(),
            ).replace(" ", "_")
            if not token or token in {"none", "nane", "nong"}:
                continue
            scored = sorted(
                (
                    (SequenceMatcher(None, token, flag).ratio(), flag)
                    for flag in SAFE_REVIEW_FLAGS
                ),
                reverse=True,
            )
            (best, flag), (second, _) = scored
            if best >= 0.60 and best - second >= 0.08:
                found.add(flag)
    return frozenset(found)


def _eligible_biometric_pages(
    row: PredictionRow,
    candidates: Iterable[CandidateEvidence],
) -> tuple[int, ...]:
    existing = _parse_flags(row.risk_flags)
    if not existing <= SAFE_REVIEW_FLAGS:
        return ()

    by_page: dict[int, list[CandidateEvidence]] = defaultdict(list)
    for candidate in candidates:
        if (
            candidate.evidence_type is EvidenceType.BIOMETRIC_SLIP
            and not candidate.superseded
        ):
            by_page[candidate.page_index].append(candidate)

    eligible = []
    for page_index, page_candidates in by_page.items():
        if existing:
            unresolved_risk = any(
                candidate.field_name == "risk_flags"
                and candidate.value is None
                and not candidate.legible
                and candidate.ocr_confidence <= LOW_CONFIDENCE_MAXIMUM
                for candidate in page_candidates
            )
            visible_partial = any(
                candidate.field_name == "risk_flags"
                and candidate.value is not None
                and _parse_flags(candidate.value) == existing
                for candidate in page_candidates
            )
            if unresolved_risk and visible_partial:
                eligible.append(page_index)
            continue

        unresolved_identity_field = any(
            candidate.field_name in {"applicant_name", "species_code"}
            and candidate.value is None
            and not candidate.legible
            and candidate.ocr_confidence <= LOW_CONFIDENCE_MAXIMUM
            for candidate in page_candidates
        )
        if unresolved_identity_field:
            eligible.append(page_index)
    return tuple(sorted(set(eligible)))


def _largest_visible_native_image(
    pdf_path: Path,
    page_index: int,
) -> Image.Image | None:
    import pypdfium2
    from pypdfium2 import raw

    with PDFIUM_RENDER_LOCK:
        document = pypdfium2.PdfDocument(pdf_path)
        try:
            if page_index < 0 or page_index >= len(document):
                return None
            page = document[page_index]
            try:
                page_width, page_height = page.get_size()
                choices = []
                for image_object in page.get_objects(
                    filter=[raw.FPDF_PAGEOBJ_IMAGE],
                ):
                    width, height = image_object.get_px_size()
                    if width < MINIMUM_IMAGE_EDGE or height < MINIMUM_IMAGE_EDGE:
                        continue
                    left, bottom, right, top = image_object.get_bounds()
                    intersects_page = (
                        min(right, page_width) > max(left, 0.0)
                        and min(top, page_height) > max(bottom, 0.0)
                    )
                    if intersects_page:
                        choices.append((width * height, image_object))
                if not choices:
                    return None
                selected = max(choices, key=lambda item: item[0])[1]
                bitmap = selected.get_bitmap(scale_to_original=True)
                try:
                    return bitmap.to_pil().convert("L").copy()
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()


def _read_image(image_path: Path, psm: int) -> str:
    completed = subprocess.run(
        [
            "tesseract",
            str(image_path),
            "stdout",
            "--psm",
            str(psm),
            "-l",
            "eng",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.decode("utf-8", errors="replace")


class VisibleMaxFilterRiskRecoverer:
    """Add review-only biometric flags when MaxFilter strictly improves OCR."""

    def __init__(
        self,
        *,
        image_loader: Callable[[Path, int], Image.Image | None] = (
            _largest_visible_native_image
        ),
        image_reader: Callable[[Path, int], str] = _read_image,
    ) -> None:
        self._image_loader = image_loader
        self._image_reader = image_reader

    def recover(
        self,
        pdf_path: Path,
        row: PredictionRow,
        candidates: Iterable[CandidateEvidence],
    ) -> PredictionRow:
        pages = _eligible_biometric_pages(row, candidates)
        if not pages:
            return row
        existing = _parse_flags(row.risk_flags)
        recovered: set[str] = set()
        try:
            with tempfile.TemporaryDirectory(prefix="mib-maxfilter-") as raw_temp:
                temp = Path(raw_temp)
                for page_index in pages:
                    image = self._image_loader(pdf_path, page_index)
                    if image is None:
                        continue
                    enlarged = image.resize((image.width * 2, image.height * 2))
                    filtered = enlarged.filter(ImageFilter.MaxFilter(3))
                    raw_path = temp / f"page-{page_index + 1}-raw.png"
                    filtered_path = temp / f"page-{page_index + 1}-maxfilter.png"
                    enlarged.save(raw_path)
                    filtered.save(filtered_path)
                    raw_attempts = [
                        self._image_reader(raw_path, psm) for psm in (3, 6)
                    ]
                    filtered_attempts = [
                        self._image_reader(filtered_path, psm) for psm in (3, 6)
                    ]
                    safe_raw = [
                        text
                        for text in raw_attempts
                        if not ANSWER_MARKER_RE.search(text)
                    ]
                    safe_filtered = [
                        text
                        for text in filtered_attempts
                        if not ANSWER_MARKER_RE.search(text)
                    ]
                    if not safe_filtered:
                        continue
                    raw_quality = max(
                        (_word_quality(text) for text in safe_raw),
                        default=(0, 0),
                    )
                    best_filtered = max(
                        safe_filtered,
                        key=_word_quality,
                    )
                    if _word_quality(best_filtered) <= raw_quality:
                        continue
                    proposed = _visible_safe_flags(best_filtered)
                    if proposed > existing:
                        recovered.update(proposed - existing)
        except (OSError, subprocess.SubprocessError):
            return row

        after = existing | recovered
        if not recovered or not after <= SAFE_REVIEW_FLAGS:
            return row
        return replace(row, risk_flags="|".join(sorted(after)))


class VisibleFieldRecoveryChain:
    """Apply independent output-only field recoverers in a fixed order."""

    def __init__(self, *recoverers: FieldRecoverer) -> None:
        self._recoverers = recoverers

    def recover(
        self,
        pdf_path: Path,
        row: PredictionRow,
        candidates: Iterable[CandidateEvidence],
    ) -> PredictionRow:
        frozen_candidates = tuple(candidates)
        recovered = row
        for recoverer in self._recoverers:
            recovered = recoverer.recover(
                pdf_path,
                recovered,
                frozen_candidates,
            )
        return recovered
