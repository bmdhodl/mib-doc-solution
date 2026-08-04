"""Raster trust boundary for approval-capable PDF text-layer evidence.

Selectable PDF text is useful for routing and field recovery, but it is not
proof that a human can see the same evidence.  This module permits selectable
text to participate in an approval only when a fresh OCR pass over rendered
pixels independently corroborates the approval-critical facts.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from .pdfium_runtime import PDFIUM_RENDER_LOCK

_UNTRUSTED_CONTENT_RE = re.compile(
    r"^\s*(?:(?:USER|HUMAN|PROMPT|INSTRUCTION)\s*:|"
    r"[\[<](?:USER|HUMAN|PROMPT|INSTRUCTION)[\]>]\s*:?)|"
    r"\b(?:SYSTEM|ASSISTANT|DEVELOPER)\b|"
    r"S[\W_]*Y[\W_]*S[\W_]*T[\W_]*E[\W_]*M|"
    r"\b(?:GROUND[\s_-]*TRUTH|ANSWER[\s_-]*KEY)\b|"
    r"\bQR(?:[\s_-]+CODE)?[\s_-]+(?:POLICY|INSTRUCTION)\b|"
    r"\b(?:IGNORE\s+(?:ALL\s+)?(?:PREVIOUS|PRIOR)(?:\s+INSTRUCTIONS?)?|"
    r"OUTPUT\s+(?:THIS\s+ANSWER|EXACTLY)|"
    r"FORCE\s+ADJUDICATION|CORRECT\s+ADJUDICATION|"
    r"TRAINING\s+EXAMPLE)\b",
    re.I | re.M,
)
_MAX_APPROVAL_WITNESS_PAGES = 6
_CASE_ID_RE = re.compile(r"\bMIB[- ]?(\d{6})\b", re.I)
_ANSWER_TABLE_ROW_RE = re.compile(
    r"MIB-\d{6},.*,(?:APPROVED|DENIED|NEEDS_REVIEW)",
    re.I,
)
_DECISION_CUE_ROUTE_PATTERNS = {
    "finding_denied": re.compile(r"Finding\s*:?\s*DENIED\b", re.I),
    "finding_review": re.compile(
        r"Finding\s*:?\s*NEEDS[_\s]?REVIEW\b",
        re.I,
    ),
    "registry_embargo": re.compile(
        r"Registry\s+Status\s*:?\s*EMBARGO\b",
        re.I,
    ),
    "damage": re.compile(r"\b(?:UNREADABLE|REDACTED)\b", re.I),
    "fee_paid": re.compile(r"\bAmount\s*:?\s*\$?\s*809\b", re.I),
    "layout_risk": re.compile(
        r"\b(?:Observed\s+flags?|Risk\s+flags?)\b",
        re.I,
    ),
}


def strip_untrusted_selectable_lines(text: str) -> str:
    """Remove explicit prompt and answer-table lines, preserving page breaks."""

    pages: list[str] = []
    for page in text.split("\f"):
        pages.append(
            "\n".join(
                line
                for line in page.splitlines()
                if not _UNTRUSTED_CONTENT_RE.search(line)
                and not _ANSWER_TABLE_ROW_RE.search(line)
            )
        )
    return "\f".join(pages)


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _case_ids(text: str) -> frozenset[str]:
    return frozenset(
        f"MIB-{match}"
        for match in _CASE_ID_RE.findall(text)
    )


def _page_indices_with_label(
    pages: list[str],
    pattern: str,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, page in enumerate(pages)
        if re.search(pattern, page, re.I)
    )


def _active_case_page(text: str, case_id: str) -> bool:
    return _case_ids(text) == {case_id.upper()}


def _labeled_person_values(
    text: str,
    label: str,
) -> frozenset[str]:
    """Collect two-token person values on or immediately after a label."""

    lines = [
        _normalized_words(line)
        for line in text.splitlines()
        if _normalized_words(line)
    ]
    label_words = _normalized_words(label)
    values: set[str] = set()
    for index, line in enumerate(lines):
        marker = line.find(label_words)
        if marker < 0:
            continue
        trailing = line[marker + len(label_words):].strip()
        candidate = (
            trailing
            if trailing
            else lines[index + 1] if index + 1 < len(lines) else ""
        )
        if re.fullmatch(r"[a-z][a-z0-9]* [a-z][a-z0-9]*", candidate):
            values.add(candidate)
    return frozenset(values)


def _amount_809_visible(text: str) -> bool:
    lines = [
        _normalized_words(line)
        for line in text.splitlines()
        if _normalized_words(line)
    ]
    for index, line in enumerate(lines):
        marker = line.find("amount")
        if marker < 0:
            continue
        trailing = line[marker + len("amount"):].strip()
        if trailing == "809 00":
            return True
        if not trailing and index + 1 < len(lines):
            if lines[index + 1] == "809 00":
                return True
    return bool(
        re.search(r"\bAmount\s*:?\s*\$?\s*809(?:[.,]00)?\b", text, re.I)
    )


def approval_layout_corroborated(
    selectable_text: str,
    raster_text: str,
    *,
    case_id: str,
    applicant_name: str,
    fee_waived: bool,
) -> bool:
    """Return true only when rendered pixels repeat all approval-critical facts.

    The selectable layer may nominate facts, but it cannot satisfy this gate.
    Prompt-like content in either representation makes the witness untrusted.
    """

    if (
        not selectable_text
        or not raster_text
        or _UNTRUSTED_CONTENT_RE.search(selectable_text)
        or _UNTRUSTED_CONTENT_RE.search(raster_text)
    ):
        return False

    selectable_pages = selectable_text.split("\f")
    raster_pages = raster_text.split("\f")
    if len(selectable_pages) != len(raster_pages):
        return False

    registry_pages = _page_indices_with_label(
        selectable_pages,
        r"Registry[ \t]+Name",
    )
    if not registry_pages or any(
        not _active_case_page(selectable_pages[index], case_id)
        for index in registry_pages
    ):
        return False
    raster_registry_values: set[str] = set()
    for index in registry_pages:
        if not _active_case_page(raster_pages[index], case_id):
            return False
        raster_registry_values.update(
            _labeled_person_values(raster_pages[index], "Registry Name")
        )
    if raster_registry_values != {_normalized_words(applicant_name)}:
        return False

    if fee_waived:
        # Payment/waiver status already has independent render-first extraction
        # provenance on the row.  The text layer contributes only name
        # consensus on this path, so those are the facts corroborated here.
        return True
    fee_pages = _page_indices_with_label(
        selectable_pages,
        r"Amount\s*\$?\s*809(?:[.,]00)?\b",
    )
    if not fee_pages or any(
        not _active_case_page(selectable_pages[index], case_id)
        for index in fee_pages
    ):
        return False
    return any(
        _active_case_page(raster_pages[index], case_id)
        and _amount_809_visible(raster_pages[index])
        for index in fee_pages
    )


def _candidate_page_indices(selectable_text: str) -> tuple[int, ...]:
    """Use the untrusted layer only as a bounded render routing hint."""

    routed = tuple(
        index
        for index, block in enumerate(selectable_text.split("\x0c"))
        if re.search(
            r"Registry[ \t]+Name|Applicant[ \t]*:?|Amount\s*\$?\s*809",
            block,
            re.I,
        )
    )
    if not routed or len(routed) > _MAX_APPROVAL_WITNESS_PAGES:
        return ()
    return routed


def _tesseract_text(image: object, path: Path, *, psm: str) -> str:
    binary = shutil.which("tesseract")
    if binary is None:
        windows_binary = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        binary = str(windows_binary) if windows_binary.is_file() else "tesseract"
    try:
        image.save(path)
        completed = subprocess.run(
            [binary, str(path), "stdout", "--psm", psm, "-l", "eng"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=40,
        )
    except (AttributeError, OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout or ""


@lru_cache(maxsize=32)
def _raster_approval_witness(
    pdf_path: Path,
    page_indices: tuple[int, ...],
    page_scales: tuple[float, ...],
) -> str:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return ""

    images: list[tuple[int, object]] = []
    try:
        with PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                if len(page_indices) != len(page_scales):
                    return ""
                if any(index < 0 or index >= len(document) for index in page_indices):
                    return ""
                for index, scale in zip(page_indices, page_scales):
                    page = document[index]
                    try:
                        images.append(
                            (
                                index,
                                page.render(scale=scale, grayscale=True)
                                .to_pil()
                                .convert("L"),
                            )
                        )
                    finally:
                        page.close()
            finally:
                document.close()
    except Exception:
        return ""

    texts: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="mib-approval-witness-") as directory:
            work = Path(directory)
            for index, image in images:
                texts.append(
                    "\n".join(
                        _tesseract_text(
                            image,
                            work / f"{index}-{psm}.png",
                            psm=psm,
                        )
                        for psm in ("6", "11")
                    )
                )
    except Exception:
        return ""
    return "\n\f\n".join(texts)


def raster_text_for_selectable_decision_cues(
    pdf_path: Path,
    selectable_text: str,
    cue_names: tuple[str, ...],
) -> str:
    """OCR only pages nominated by a selectable-text decision cue."""

    trusted_text = strip_untrusted_selectable_lines(selectable_text)
    patterns = tuple(
        _DECISION_CUE_ROUTE_PATTERNS[cue_name]
        for cue_name in cue_names
        if cue_name in _DECISION_CUE_ROUTE_PATTERNS
    )
    if not patterns:
        return ""
    page_indices = tuple(
        index
        for index, page in enumerate(trusted_text.split("\f"))
        if any(pattern.search(page) for pattern in patterns)
    )
    if not page_indices or len(page_indices) > _MAX_APPROVAL_WITNESS_PAGES:
        return ""
    return _raster_approval_witness(
        pdf_path,
        page_indices,
        tuple(2.5 for _index in page_indices),
    )


def raster_corroborates_layout_approval(
    pdf_path: Path,
    selectable_text: str,
    *,
    case_id: str,
    applicant_name: str,
    fee_waived: bool,
) -> bool:
    """Enforce the selectable-text-to-approval trust boundary."""

    # Sanitize the entire packet before using its text layer for routing.
    # Otherwise an instruction-only page would disappear from the routed view.
    if _UNTRUSTED_CONTENT_RE.search(selectable_text):
        return False
    page_indices = _candidate_page_indices(selectable_text)
    if not page_indices:
        return False
    selectable_pages = selectable_text.split("\f")
    routed_selectable_text = "\f".join(
        selectable_pages[index]
        for index in page_indices
    )
    # The pinned Linux Tesseract build reads the full recurring layout
    # consistently at 2.5x. Higher registry-only scales introduce I/l
    # substitutions (for example, Ixomora -> lxomora) without adding evidence.
    page_scales = tuple(2.5 for _index in page_indices)
    raster_text = _raster_approval_witness(
        pdf_path,
        page_indices,
        page_scales,
    )
    return approval_layout_corroborated(
        routed_selectable_text,
        raster_text,
        case_id=case_id,
        applicant_name=applicant_name,
        fee_waived=fee_waived,
    )
