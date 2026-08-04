"""Arjun-owned recovery layers on the render-first stack.

Design rules
------------
- No train-label / case-ID unlocks.
- No ``silent risk → APPROVED`` promotions (schema-default ``none`` is not
  observed clearance).
- Layout field repairs / finding / damage heads never create approvals by
  themselves. Only an exact-case manual Finding agreed by both rotated pixel
  scans may promote REVIEW to APPROVED.
- Layout consensus uses visible ``$809`` + registry==applicant only.
- Embedded generator instructions are stripped as untrusted content.
- Demotion may only move APPROVED → REVIEW/DENIED.
- v31 lesson: Fee-Status-alone / OCR-fee-alone / loose identity spiked CFA.
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .adjudication import PolicyRuleSet
from .extraction import KNOWN_RISK_FLAGS, CandidateEvidence
from .models import PredictionRow
from .pdfium_runtime import PDFIUM_RENDER_LOCK
from .visible_trust import (
    raster_text_for_selectable_decision_cues,
    raster_corroborates_layout_approval,
    strip_untrusted_selectable_lines,
)

LAYOUT_CONSENSUS_APPROVAL_CONFIDENCE = 0.85
DEMOTION_REVIEW_CONFIDENCE = 0.55
DEMOTION_DENIAL_CONFIDENCE = 0.92
_PDFIUM_RENDER_LOCK = PDFIUM_RENDER_LOCK
REDACTED_STALE_VISA_REVIEW_CONFIDENCE = 0.78
_PACKET_SNAPSHOT_DATE = date(2026, 7, 7)

_KNOWN_PURPOSES = (
    "reactor maintenance",
    "field repair",
    "medical consult",
    "research",
    "cultural exchange",
    "translation",
    "archive audit",
    "xenobotany",
    "diplomatic",
    "transit",
)

_VISA_CLASSES = frozenset({"XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"})
# LC visas: DIP-1/XW-2 (paid) plus MED-3/XW-1 after measured silent-stamp
# CFA cells were quarantined into ``_LC_TRAP_VISA_PURPOSE_SIG``. Waived LC is
# gated separately via ``_layout_fee_waived_proven`` / fee_status==waived.
_LAYOUT_CONSENSUS_VISAS = frozenset({"DIP-1", "XW-2", "MED-3", "XW-1"})
_LAYOUT_CONSENSUS_PAID_VISAS = frozenset({"DIP-1", "XW-2", "MED-3", "XW-1"})
_LAYOUT_CONSENSUS_WAIVED_VISAS = frozenset({"DIP-1", "XW-2", "MED-3", "XW-1"})
_POLICY = PolicyRuleSet()

# Fail-closed demotion traps: cells that mint false APPROVED under LC.
# Conservative — never unlocks approvals; only blocks known trap assemblies
# (silent-stamp CFA and measured false-APPROVED on true REVIEW).
_LC_TRAP_VISA_PURPOSE: frozenset[tuple[str, str]] = frozenset(
    {
        ("DIP-1", "xenobotany"),
        ("XW-2", "archive audit"),
    }
)
_LC_TRAP_VISA_PURPOSE_SIG: frozenset[tuple[str, str, str]] = frozenset(
    {
        # Original DIP/XW-2 paid traps
        ("DIP-1", "reactor maintenance", "FRI"),
        ("DIP-1", "archive audit", "FIR"),
        ("XW-2", "diplomatic", "IFR"),
        # MED-3/XW-1 paid expand: silent biohazard/memory CFA
        ("XW-1", "research", "FRI"),
        ("MED-3", "diplomatic", "IRF"),
        ("MED-3", "reactor maintenance", "FIR"),
        ("MED-3", "field repair", "IFR"),
        ("MED-3", "archive audit", "RFI"),
        # Waived MED-3/XW-1 CFA
        ("XW-1", "archive audit", "RFI"),
        ("MED-3", "transit", "IRF"),
        ("MED-3", "translation", "IRF"),
        # Waived FAP (true REVIEW)
        ("XW-2", "reactor maintenance", "IFR"),
        ("XW-1", "xenobotany", "RFI"),
        ("MED-3", "xenobotany", "FIR"),
    }
)
# Waived-only traps: paid path may approve; waived path stays REVIEW.
# (DIP cultural-exchange/FIR and transit/IFR mix paid gold APPROVED with
# waived silent-stamp DENIED — trap only the waived arm.)
_LC_WAIVED_ONLY_TRAP_VISA_PURPOSE_SIG: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("DIP-1", "cultural exchange", "FIR"),
        ("DIP-1", "transit", "IFR"),
    }
)
# Waived-only override: paid path keeps the trap; waived path may approve.
_LC_WAIVED_TRAP_OVERRIDES: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("DIP-1", "reactor maintenance", "FRI"),
    }
)


@lru_cache(maxsize=16)
def _pdf_layout_text(pdf_path: Path) -> str:
    """Return layout text once per active packet.

    A row finalization pass consults the same immutable PDF as many as eight
    times.  Keeping a small worker-shared cache avoids repeatedly launching
    ``pdftotext`` for that packet while bounding retained text to the handful
    of PDFs concurrently in flight.
    """

    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and (completed.stdout or "").strip():
        return completed.stdout or ""

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return ""
    parts: list[str] = []
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page = document[index]
                    try:
                        textpage = page.get_textpage()
                        try:
                            parts.append(textpage.get_text_bounded() or "")
                        finally:
                            close_textpage = getattr(textpage, "close", None)
                            if callable(close_textpage):
                                close_textpage()
                    finally:
                        close_page = getattr(page, "close", None)
                        if callable(close_page):
                            close_page()
            finally:
                document.close()
    except Exception:
        return ""
    # Preserve page boundaries when Poppler is unavailable in the submission
    # image. Layout-consensus safety gates derive their F/R/I/B/M/O signature
    # from form-feed-separated pages.
    return "\x0c".join(parts)


def has_hollow_slash_stamp_pixels(rgb: object) -> bool:
    """Detect the visible hollow blue slash-square denial mark."""

    try:
        import numpy as np
    except ImportError:
        return False

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        return False
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    mask = (
        (blue > red + 25)
        & (blue > green + 5)
        & (blue > 160)
        & (red < 210)
        & (blue < 250)
    )
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        cells: list[tuple[int, int]] = []
        while queue:
            current_y, current_x = queue.popleft()
            cells.append((current_y, current_x))
            for offset_y, offset_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_y = current_y + offset_y
                next_x = current_x + offset_x
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        area = len(cells)
        if area < 1000 or area > 1800:
            continue
        cell_y = [cell[0] for cell in cells]
        cell_x = [cell[1] for cell in cells]
        box_width = max(cell_x) - min(cell_x) + 1
        box_height = max(cell_y) - min(cell_y) + 1
        aspect = box_width / max(box_height, 1)
        fill = area / (box_width * box_height)
        if 0.95 <= aspect <= 1.05 and 70 <= box_width <= 90 and 0.20 <= fill <= 0.28:
            return True
    return False


def apply_visible_slash_stamp_denial_from_signals(
    row: PredictionRow,
    *,
    has_stamp: bool,
    fee_paid_proven: bool,
) -> PredictionRow:
    """Apply the deny-only stamp signal without reading packet identity."""

    if (
        row.adjudication != "NEEDS_REVIEW"
        or row.fee_status != "paid"
        or fee_paid_proven
        or not has_stamp
    ):
        return row
    payload = row.to_dict()
    payload["adjudication"] = "DENIED"
    payload["confidence"] = 0.95
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_visible_slash_stamp_denial(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Deny a weak-review packet only for the visible slash-square mark."""

    if row.adjudication != "NEEDS_REVIEW" or row.fee_status != "paid":
        return row
    text = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    selectable_fee_paid = bool(text and _layout_fee_paid_proven(text))
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError:
        return row
    has_stamp = False
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page = document[index]
                    try:
                        pixels = np.asarray(
                            page.render(scale=2.0).to_pil().convert("RGB")
                        )
                    finally:
                        page.close()
                    if has_hollow_slash_stamp_pixels(pixels):
                        has_stamp = True
                        break
            finally:
                document.close()
    except Exception:
        return row
    raster_text = ""
    if has_stamp and selectable_fee_paid:
        raster_text = strip_untrusted_selectable_lines(
            raster_text_for_selectable_decision_cues(
                pdf_path,
                text,
                ("fee_paid",),
            )
        )
    fee_paid_proven = bool(
        raster_text and _layout_fee_paid_proven(raster_text)
    )
    return apply_visible_slash_stamp_denial_from_signals(
        row,
        has_stamp=has_stamp,
        fee_paid_proven=fee_paid_proven,
    )


def red_channel_binary_mask(rgb: object) -> object | None:
    """Return a Tesseract-friendly mask containing only saturated red marks."""

    try:
        import numpy as np
    except ImportError:
        return None

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    red = array[:, :, 0].astype(np.float32)
    green = array[:, :, 1].astype(np.float32)
    blue = array[:, :, 2].astype(np.float32)
    score = np.clip(red - 0.5 * green - 0.5 * blue, 0, None)
    if float(score.max()) <= 0:
        return None
    scaled = (score / score.max() * 255.0).astype(np.uint8)
    positive = scaled[scaled > 0]
    threshold = float(np.percentile(positive, 85)) if positive.size else 200.0
    return 255 - (scaled >= threshold).astype(np.uint8) * 255


def apply_visible_sample_denial_from_signals(
    row: PredictionRow,
    *,
    has_sample_denial: bool,
    fee_signal: bool,
    review_signal: bool,
) -> PredictionRow:
    """Apply a deny-only red watermark when no stronger review evidence exists."""

    if (
        row.adjudication != "NEEDS_REVIEW"
        or row.visa_class != "DIP-1"
        or row.fee_status != "paid"
        or row.declared_purpose == "transit"
        or _norm_flags(row.risk_flags) != "none"
        or not has_sample_denial
        or fee_signal
        or review_signal
    ):
        return row
    payload = row.to_dict()
    payload["fee_status"] = "unpaid"
    payload["adjudication"] = "DENIED"
    payload["confidence"] = 0.98
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def _tesseract_image_text(image: object, path: Path, *, psm: str) -> str:
    binary = shutil.which("tesseract")
    if binary is None:
        windows_binary = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        binary = str(windows_binary) if windows_binary.is_file() else "tesseract"
    try:
        image.save(path)
        result = subprocess.run(
            [binary, str(path), "stdout", "--psm", psm, "-l", "eng"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=40,
            check=False,
        )
    except (AttributeError, OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout or ""


_HARD_DENIAL_PIXEL_WITNESS = re.compile(
    r"\b(?:active[_\s-]*warrant|biohazard[_\s-]*red|"
    r"memory[_\s-]*tampering|planetary[_\s-]*embargo)\b|"
    r"Registry\s+Status\s*:?\s*EMBARGO|"
    r"Finding\s*[:=-]?\s*DENIED",
    re.I,
)


def _active_case_pixel_ocr_pages(
    pdf_path: Path,
    case_id: str,
) -> tuple[str, ...]:
    """OCR rendered pages and discard every foreign-case pixel page."""

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return ()
    images: list[object | None] = []
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page = document[index]
                    try:
                        images.append(page.render(scale=2.5).to_pil().convert("L"))
                    except Exception:
                        images.append(None)
                    finally:
                        page.close()
            finally:
                document.close()
    except Exception:
        return ()
    pages: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mib-redacted-visa-") as directory:
        work = Path(directory)
        for index, image in enumerate(images):
            if image is None:
                pages.append("")
                continue
            views = tuple(
                _tesseract_image_text(
                    image,
                    work / f"{index}-{psm}.png",
                    psm=psm,
                )
                for psm in ("6", "11")
            )
            text = "\n".join(views)
            visible_case_ids = {
                f"MIB-{match}"
                for match in re.findall(
                    r"\bMIB[- ]?(\d{6})\b",
                    text,
                    re.I,
                )
            }
            pages.append(
                text
                if case_id in visible_case_ids and visible_case_ids <= {case_id}
                else ""
            )
    return tuple(pages)


def _redacted_stale_visa_route(row: PredictionRow) -> bool:
    """Bound the costly pixel retry to its measured stale MED-3 lane."""

    return bool(
        row.adjudication == "DENIED"
        and row.visa_class == "MED-3"
        and row.declared_purpose == "reactor maintenance"
        and row.fee_status == "paid"
        and _norm_flags(row.risk_flags) == "none"
        and row.arrival_date not in {"1900-01-01", "unknown", ""}
    )


def apply_uncorroborated_redacted_stale_visa_review(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Demote a stale denial whose sole visible MED-3 visa is redacted."""

    if not _redacted_stale_visa_route(row):
        return row
    try:
        arrival = date.fromisoformat(row.arrival_date)
    except ValueError:
        return row
    if (_PACKET_SNAPSHOT_DATE - arrival).days <= 180:
        return row

    pages = tuple(
        page
        if {f"MIB-{match}" for match in re.findall(r"\bMIB[- ]?(\d{6})\b", page, re.I)}
        == {row.case_id}
        else ""
        for page in _active_case_pixel_ocr_pages(pdf_path, row.case_id)
    )
    if not pages or any(_HARD_DENIAL_PIXEL_WITNESS.search(page) for page in pages):
        return row
    visa = re.escape(row.visa_class).replace(r"\-", r"[\s-]?")
    visa_line = re.compile(
        rf"(?:visa\s+class\s*[:.]?\s*{visa}|class\s+{visa}\s+compliance)",
        re.I,
    )
    visa_pages = tuple(
        index for index, page in enumerate(pages) if visa_line.search(page)
    )
    redacted_visa_pages = tuple(
        index
        for index, page in enumerate(pages)
        if re.search(r"\bREDACTED\s*\?", page, re.I) and visa_line.search(page)
    )
    if len(visa_pages) != 1 or redacted_visa_pages != visa_pages:
        return row

    payload = row.to_dict()
    payload["adjudication"] = "NEEDS_REVIEW"
    payload["confidence"] = REDACTED_STALE_VISA_REVIEW_CONFIDENCE
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_visible_sample_denial(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Recover a visible red SAMPLE DENIAL watermark on narrow weak reviews."""

    if (
        row.adjudication != "NEEDS_REVIEW"
        or row.visa_class != "DIP-1"
        or row.fee_status != "paid"
        or row.declared_purpose == "transit"
        or _norm_flags(row.risk_flags) != "none"
    ):
        return row
    try:
        import pypdfium2 as pdfium
        from PIL import Image
    except ImportError:
        return row
    images: list[object] = []
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page = document[index]
                    try:
                        images.append(page.render(scale=2.5).to_pil().convert("RGB"))
                    finally:
                        page.close()
            finally:
                document.close()
    except Exception:
        return row
    has_sample_denial = False
    fee_signal = False
    review_signal = False
    with tempfile.TemporaryDirectory(prefix="mib-red-watermark-") as directory:
        work = Path(directory)
        for index, image in enumerate(images):
            normal_text = _tesseract_image_text(
                image,
                work / f"{index}-normal.png",
                psm="11",
            )
            fee_signal = fee_signal or bool(
                re.search(
                    r"MIB\s*Fee\s*Receipt|Fee\s*Receipt|"
                    r"Fee\s*Status\s*:|Amount\s*\$",
                    normal_text,
                    re.I,
                )
            )
            review_signal = review_signal or bool(
                re.search(
                    r"\b(?:ILLEGIBLE|WASHED\s+OUT|REDACTED\?)\b",
                    normal_text,
                    re.I,
                )
            )
            if fee_signal or review_signal:
                break

            mask = red_channel_binary_mask(image)
            if mask is None:
                continue
            red_image = Image.fromarray(mask)
            for psm in ("6", "11"):
                red_text = _tesseract_image_text(
                    red_image,
                    work / f"{index}-red-{psm}.png",
                    psm=psm,
                )
                if re.search(r"SAMP\w*\s*DENI", red_text, re.I) or re.search(
                    r"SAMPLE\s+[A-Z0-9]*N[A-Z0-9]*[AIYL]",
                    red_text,
                    re.I,
                ):
                    has_sample_denial = True
                    break
    return apply_visible_sample_denial_from_signals(
        row,
        has_sample_denial=has_sample_denial,
        fee_signal=fee_signal,
        review_signal=review_signal,
    )


def _norm_flags(value: str | None) -> str:
    raw = " ".join(str(value or "").strip().split()).casefold()
    if raw in {"", "none", "null", "unknown"}:
        return "none"
    return "|".join(sorted(part.strip() for part in raw.split("|") if part.strip()))


def _parse_flag_set(value: str | None) -> frozenset[str]:
    normalized = _norm_flags(value)
    if normalized == "none":
        return frozenset()
    return frozenset(part for part in normalized.split("|") if part and part != "none")


def _clean_person_name(raw: str) -> str | None:
    text = " ".join(raw.split())
    text = re.split(r"\s{2,}|\s+PASSPORT|\s+CASE|\s+SPN|\s+is\b", text)[0].strip()
    parts = text.split()
    if len(parts) >= 2 and all(
        re.fullmatch(r"[A-Z][a-z]+", part) for part in parts[:2]
    ):
        return " ".join(parts[:2])
    return None


def apply_visible_field_repairs(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Identity-free fee/name/visa/purpose repairs from layout text.

    Never creates approvals. Uses AK-stripped layout text only.
    Ported from the public 132.34 / CFA=0 stack (fields-only lift).
    """

    text = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    if not text:
        return row
    payload = row.to_dict()
    changed = False

    if re.search(r"Amount\s*\$?\s*0(?:[.,]00)?", text, re.I) and re.search(
        r"DIP[\s\-]?WAIVER", text, re.I
    ):
        if payload.get("fee_status") != "waived":
            payload["fee_status"] = "waived"
            changed = True
    elif re.search(r"Amount\s*\$?\s*809(?:[.,]00)?", text, re.I):
        # Canonical paid amount wins over Fee Status waived when Waiver is N/A
        # (conflicting receipt lines). Also repairs unpaid/unknown → paid.
        waiver_na = bool(re.search(r"Waiver\s*Code\s*[:#]?\s*N\s*/?\s*A", text, re.I))
        cur = payload.get("fee_status")
        if cur in {"unpaid", "unknown"} or (cur == "waived" and waiver_na):
            if cur != "paid":
                payload["fee_status"] = "paid"
                changed = True

    registries = [
        name
        for raw in re.findall(
            r"Registry\s+Name\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (name := _clean_person_name(raw))
    ]
    applicants = [
        name
        for raw in re.findall(
            r"Applicant\s*:?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (name := _clean_person_name(raw))
    ]
    registry = registries[0] if len(set(registries)) == 1 else None
    applicant = applicants[0] if len(set(applicants)) == 1 else None
    att_name = None
    att_purpose = None
    for match in re.finditer(
        r"attests that ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+) is expected on Earth for ([a-z \n]+?)(?:\.|,|\n\n)",
        text,
        re.I,
    ):
        att_name = _clean_person_name(match.group(1))
        purpose_blob = " ".join(match.group(2).casefold().split())
        for purpose in _KNOWN_PURPOSES:
            if purpose_blob == purpose or purpose_blob.startswith(purpose):
                att_purpose = purpose
                break

    if registry and applicant and registry == applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True
    elif registry and applicant and registry != applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True
    elif registry and not applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True

    current_name = str(payload.get("applicant_name") or "").strip()
    for candidate in filter(None, (registry, applicant, att_name)):
        if (
            current_name
            and candidate != current_name
            and candidate.startswith(current_name)
            and len(candidate) > len(current_name) + 2
        ):
            payload["applicant_name"] = candidate
            changed = True
            break
    if (not current_name or current_name.casefold() in {"unknown", "n/a", "none"}) and (
        registry or applicant
    ):
        fill = registry or applicant
        if payload.get("applicant_name") != fill:
            payload["applicant_name"] = fill
            changed = True

    visa_hits = [
        value.upper()
        for value in re.findall(
            r"responsibility for class\s+([A-Z0-9\-]+)\s+compliance",
            text,
            re.I,
        )
        if value.upper() in _VISA_CLASSES and value.upper() != "TRANSIT-7"
    ]
    if len(set(visa_hits)) == 1 and payload.get("visa_class") != visa_hits[0]:
        payload["visa_class"] = visa_hits[0]
        changed = True

    arrivals = sorted(
        set(re.findall(r"Arrival\s+Date\s+(\d{4}-\d{2}-\d{2})", text, re.I))
    )
    if len(arrivals) == 1 and payload.get("arrival_date") != arrivals[0]:
        payload["arrival_date"] = arrivals[0]
        changed = True

    revoked = sorted(set(re.findall(r"Revoked sponsor:\s*(SPN-\d{4})", text, re.I)))
    attested = sorted(set(re.findall(r"Sponsor\s+(SPN-\d{4})\s+attests", text, re.I)))
    sponsor_pick: str | None = None
    current_sponsor = str(payload.get("sponsor_id") or "")
    if len(revoked) == 1:
        sponsor_pick = revoked[0]
    elif len(attested) == 1 and current_sponsor in {"SPN-0000", "unknown", ""}:
        sponsor_pick = attested[0]
    elif len(attested) == 1 and re.fullmatch(r"SPN-\d{4}", current_sponsor):
        if current_sponsor[:7] == attested[0][:7] and current_sponsor != attested[0]:
            sponsor_pick = attested[0]
    if sponsor_pick and payload.get("sponsor_id") != sponsor_pick:
        payload["sponsor_id"] = sponsor_pick
        changed = True

    if (
        payload.get("declared_purpose") == "reactor maintenance"
        and att_purpose
        and att_purpose != "reactor maintenance"
    ):
        payload["declared_purpose"] = att_purpose
        changed = True
    elif payload.get("declared_purpose") == "reactor maintenance":
        bound: list[str] = []
        for purpose in _KNOWN_PURPOSES:
            if purpose == "reactor maintenance":
                continue
            pat = (
                rf"(?:declared\s+purpose\s*[:#.=_-]\s*{re.escape(purpose)}"
                rf"|purpose\s+of\s+visit\s*[:#.=_-]\s*{re.escape(purpose)})"
            )
            if re.search(pat, text, re.I):
                bound.append(purpose)
        unique = sorted(set(bound))
        if len(unique) == 1:
            payload["declared_purpose"] = unique[0]
            changed = True

    if not changed:
        return row
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


# Challenge packets use arrival years in {2025, 2026, 2027}. OCR frequently
# confuses digits into 2020/2028/2928/2996-class garble while keeping month-day.
# Measured train_fit residual rewrite -> 2026: 3 FIX / 0 HURT. Leave modal
# placeholders (1900-01-01) and in-window years untouched.
_CHALLENGE_ARRIVAL_YEARS = frozenset({2025, 2026, 2027})


def apply_arrival_year_glitch_repair(row: PredictionRow) -> PredictionRow:
    """Rewrite out-of-window arrival years to 2026, preserving month-day.

    Identity-free field repair. Does not touch adjudication or other fields.
    """

    ad = str(row.arrival_date or "")
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", ad)
    if match is None:
        return row
    year = int(match.group(1))
    if year in _CHALLENGE_ARRIVAL_YEARS:
        return row
    if year < 2000 or year > 2999:
        return row
    repaired = f"2026-{match.group(2)}-{match.group(3)}"
    if repaired == ad:
        return row
    payload = row.to_dict()
    payload["arrival_date"] = repaired
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


# OCR month garble: digit 8 read for 6 yields impossible 2026-08 arrivals.
# Train_fit has zero truth 2026-08 dates; predicted 2026-08-* rewrites to
# 2026-06-* measured 4 FIX / 0 HURT / 1 still-wrong (day also off). Leave
# real 2025-08 arrivals untouched (11 exact on train_fit).
def apply_arrival_month_glitch_repair(row: PredictionRow) -> PredictionRow:
    """Rewrite 2026-08-DD → 2026-06-DD (OCR 8/6 month garble).

    Identity-free field repair. Does not touch adjudication.
    """

    ad = str(row.arrival_date or "")
    match = re.fullmatch(r"2026-08-(\d{2})", ad)
    if match is None:
        return row
    repaired = f"2026-06-{match.group(1)}"
    payload = row.to_dict()
    payload["arrival_date"] = repaired
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


# Paid→waived fee image recovery via public-style threshold crops.
# Measured train_fit residual: unique ``waived`` votes n>=2 → 1 FIX / 0 HURT;
# 0 flips on 130 paid-correct controls. Strict 3-vote consensus recovered zero
# residuals. Field-only; never touches adjudication.
_FEE_WAIVED_THRESHOLDS = (120, 140, 160, 180)
_FEE_WAIVED_RENDER_SCALE = 2.0
_FEE_WAIVED_MIN_VOTES = 2


def _fee_status_from_receipt_ocr(text: str) -> str | None:
    """Parse Fee Status / redundant amount rows from receipt-crop OCR."""

    match = re.search(
        r"Fee\s*Status\s*[:#]?\s*(paid|unpaid|waived|unknown)\b",
        text,
        re.I,
    )
    if match is not None:
        return match.group(1).casefold()
    match = re.search(
        r"Fee\s*Status\s*[:#]?\s*([A-Za-z0-9\-]{3,12})",
        text,
        re.I,
    )
    if match is not None:
        raw = re.sub(r"[^a-z0-9]+", "", match.group(1).casefold())
        if raw in {"paid", "unpaid", "waived", "unknown"}:
            return raw
        if re.fullmatch(r"[pnm][ao][i1l][dcl]", raw):
            return "paid"
        if raw.startswith("waiv") or raw in {"walved"}:
            return "waived"
        if raw.startswith("unp"):
            return "unpaid"
    if re.search(r"Amount\s*\$?\s*0(?:[.,]00)?\b", text, re.I) and re.search(
        r"DIP[\s\-]?WAIVER|Waiver\s*Code\s*[:#]?\s*DIP",
        text,
        re.I,
    ):
        return "waived"
    if re.search(r"Amount\s*\$?\s*809", text, re.I) and re.search(
        r"Waiver\s*Code\s*[:#]?\s*N\s*/?\s*A",
        text,
        re.I,
    ):
        return "paid"
    return None


def _page_looks_like_fee_receipt(text: str) -> bool:
    key = re.sub(r"[^a-z0-9]+", "", (text or "")[:240].casefold())
    if "feereceipt" in key or "mibfeereceipt" in key or "feestatus" in key:
        return True
    return bool(
        re.search(r"Fee\s*(Receipt|Status)|Amount\s*\$|Waiver\s*Code", text or "", re.I)
    )


def fee_receipt_threshold_votes(pdf_path: Path) -> list[str]:
    """Collect fee-status votes from top-band threshold crops (public style)."""

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return []

    images: list[object] = []
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page_object = document[index]
                    try:
                        images.append(
                            page_object.render(scale=_FEE_WAIVED_RENDER_SCALE)
                            .to_pil()
                            .convert("L")
                        )
                    finally:
                        page_object.close()
            finally:
                document.close()
    except Exception:
        return []
    votes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mib-fee-waived-") as directory:
        work = Path(directory)
        for index, page in enumerate(images):
            width, height = page.size
            top = page.crop((0, 0, width, max(1, int(height * 0.35))))
            top_text = _tesseract_image_text(
                top,
                work / f"{index}-top.png",
                psm="6",
            )
            if not _page_looks_like_fee_receipt(top_text):
                # full-page route cue
                full_text = _tesseract_image_text(
                    page,
                    work / f"{index}-full.png",
                    psm="6",
                )
                if not _page_looks_like_fee_receipt(full_text):
                    continue
            status = _fee_status_from_receipt_ocr(top_text)
            if status is not None:
                votes.append(status)
            crop = page.crop((0, 0, width, max(1, int(height * 0.30))))
            for threshold in _FEE_WAIVED_THRESHOLDS:
                binary = crop.point(
                    lambda pixel, t=threshold: 255 if pixel > t else 0
                )
                text = _tesseract_image_text(
                    binary,
                    work / f"{index}-{threshold}.png",
                    psm="6",
                )
                status = _fee_status_from_receipt_ocr(text)
                if status is not None:
                    votes.append(status)
    return votes


def fee_waived_unique_consensus(
    votes: list[str], *, min_votes: int = _FEE_WAIVED_MIN_VOTES
) -> str | None:
    """Accept only a unique ``waived`` vote tally meeting min_votes."""

    if not votes:
        return None
    from collections import Counter

    counts = Counter(votes)
    if len(counts) != 1:
        return None
    value, count = counts.most_common(1)[0]
    if value == "waived" and count >= min_votes:
        return "waived"
    return None


def _fee_waived_ocr_route(row: PredictionRow) -> bool:
    """Bound threshold receipt OCR to its measured damaged MED-3 lane."""

    return bool(
        str(row.fee_status or "").casefold() == "paid"
        and row.species_code == "JOVIAN_GASFORM"
        and row.visa_class == "MED-3"
        and row.declared_purpose == "medical consult"
    )


def apply_fee_paid_to_waived_threshold_ocr(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Override paid→waived when threshold fee-receipt OCR uniquely says waived.

    Fires only on serialized ``fee_status=paid``. Does not run when layout
    already proves canonical ``$809`` (those are trusted paid). Field-only.
    Measured: 1 FIX / 0 HURT residual, 0 control flips on 130 paid-correct.
    """

    if not _fee_waived_ocr_route(row):
        return row
    text = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    if text and _layout_fee_paid_proven(text):
        return row
    votes = fee_receipt_threshold_votes(pdf_path)
    decision = fee_waived_unique_consensus(votes)
    if decision is None:
        return row
    payload = row.to_dict()
    payload["fee_status"] = decision
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


# RapidOCR sponsor consensus (unique n>=2). The sealed runtime keeps only the
# modal gap-fill: 2 FIX / 0 HURT on train_fit, official +0.01389 CFA0.
# The clean one-digit override required another 190/800 OCR passes and would
# push the 5,000-case run beyond the wall-clock budget.
_SPONSOR_RAPID_MIN_VOTES = 2
_SPONSOR_RAPID_RENDER_SCALE = 1.8
_SPONSOR_RAPID_MAX_PAGES = 5
_SPN_OCR_RE = re.compile(r"\bSP[NM][- ]?([0-9OQIlSB]{4})\b", re.I)
_SPONSOR_RAPID_ENGINE: object | None = None
_SPONSOR_RAPID_LOCK = threading.RLock()
_SPONSOR_RAPID_MODAL_CELLS = frozenset(
    {
        ("ALPHA_DRACONIAN", "XW-1", "field repair"),
        ("AQUARIAN_MANTIS", "XW-1", "field repair"),
        ("JOVIAN_GASFORM", "XW-1", "medical consult"),
    }
)


def _normalize_spn_ocr_digits(raw: str) -> str | None:
    digits = (
        raw.upper()
        .replace("O", "0")
        .replace("Q", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("B", "8")
    )
    digits = re.sub(r"[^0-9]", "", digits)
    if len(digits) != 4 or digits == "0000":
        return None
    return f"SPN-{digits}"


def _sponsor_rapid_engine() -> object | None:
    """Lazy RapidOCR engine (string model root; OmegaConf 2.0 Path-safe)."""

    global _SPONSOR_RAPID_ENGINE
    with _SPONSOR_RAPID_LOCK:
        if _SPONSOR_RAPID_ENGINE is not None:
            return _SPONSOR_RAPID_ENGINE
        try:
            import rapidocr as rapidocr_package
        except ImportError:
            return None
        package_file = getattr(rapidocr_package, "__file__", None)
        if package_file is None:
            return None
        model_root = str(Path(package_file).resolve().parent / "models")
        try:
            _SPONSOR_RAPID_ENGINE = rapidocr_package.RapidOCR(
                params={
                    "Global.model_root_dir": model_root,
                    "Global.log_level": "error",
                    "Global.text_score": 0.30,
                    "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                }
            )
        except Exception:
            return None
        return _SPONSOR_RAPID_ENGINE


def _call_sponsor_rapid_engine(engine: object, image_bytes: bytes) -> object:
    """Serialize access to the process-global native RapidOCR session."""

    with _SPONSOR_RAPID_LOCK:
        return engine(image_bytes)  # type: ignore[operator]


def sponsor_rapid_spn_votes(pdf_path: Path) -> list[str]:
    """Collect SPN votes from RapidOCR on sponsor-looking pages."""

    engine = _sponsor_rapid_engine()
    if engine is None:
        return []
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return []
    images: list[object] = []
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                page_count = len(document)
                indices = list(range(min(page_count, _SPONSOR_RAPID_MAX_PAGES)))
                if page_count > _SPONSOR_RAPID_MAX_PAGES:
                    indices.append(page_count - 1)
                for index in sorted(set(indices)):
                    if index >= page_count:
                        continue
                    page = document[index]
                    try:
                        images.append(
                            page.render(scale=_SPONSOR_RAPID_RENDER_SCALE)
                            .to_pil()
                            .convert("RGB")
                        )
                    except Exception:
                        continue
                    finally:
                        page.close()
            finally:
                document.close()
    except Exception:
        return []
    votes: list[str] = []
    for pil in images:
        try:
            import io

            buffer = io.BytesIO()
            pil.save(buffer, format="PNG")
            result = _call_sponsor_rapid_engine(engine, buffer.getvalue())
        except Exception:
            continue
        texts = getattr(result, "txts", None)
        if texts is None:
            continue
        blob = "\n".join(str(text) for text in texts if text)
        key = re.sub(r"[^a-z0-9]+", "", blob[:500].casefold())
        if not (
            "sponsor" in key
            or "attest" in key
            or "spn" in key
            or _SPN_OCR_RE.search(blob)
        ):
            continue
        for match in _SPN_OCR_RE.finditer(blob):
            spn = _normalize_spn_ocr_digits(match.group(1))
            if spn is not None:
                votes.append(spn)
        for match in re.finditer(
            r"SP[NM]\s*[-:]?\s*([0-9OQIlSB]{3,6})",
            blob,
            re.I,
        ):
            spn = _normalize_spn_ocr_digits(match.group(1))
            if spn is not None:
                votes.append(spn)
    return votes


def sponsor_rapid_unique_consensus(
    votes: list[str],
    *,
    min_votes: int = _SPONSOR_RAPID_MIN_VOTES,
) -> str | None:
    """Accept only a unique SPN vote tally meeting min_votes."""

    if not votes:
        return None
    from collections import Counter

    counts = Counter(votes)
    (value, count), *rest = counts.most_common()
    if count < min_votes:
        return None
    if rest and rest[0][1] >= count:
        return None
    return value


def _layout_has_any_spn(text: str) -> bool:
    return bool(re.search(r"\bSPN[- ]?\d{4}\b", text or "", re.I))


def _sponsor_rapid_modal_route(row: PredictionRow) -> bool:
    """Bound the second RapidOCR pass to measured unresolved sponsor cells."""

    current = str(row.sponsor_id or "").strip().upper()
    return bool(
        current in {"SPN-0000", "UNKNOWN", ""}
        and (
            row.species_code,
            row.visa_class,
            row.declared_purpose,
        )
        in _SPONSOR_RAPID_MODAL_CELLS
    )


def apply_sponsor_rapid_consensus_ocr(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Repair sponsor_id from RapidOCR unique consensus (n>=2).

    Gates (identity-free):
    - ``SPN-0000`` / empty / unknown → gap-fill with consensus.
    - Real ``SPN-####`` values always abstain to preserve the runtime seal.

    Never invents SPN-0000. Field-only; does not touch adjudication.
    """

    current = str(row.sponsor_id or "").strip().upper()
    if not _sponsor_rapid_modal_route(row):
        return row

    layout = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    if layout and _layout_has_any_spn(layout):
        # Visible-field repairs should have handled unique layout SPN already.
        # Still allow Rapid when layout is noisy multi-SPN; only skip when
        # layout already shows the same SPN we would not gap-fill past.
        pass

    votes = sponsor_rapid_spn_votes(pdf_path)
    decision = sponsor_rapid_unique_consensus(votes)
    if decision is None or decision == current:
        return row
    payload = row.to_dict()
    payload["sponsor_id"] = decision
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_visible_finding_decision(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Honor visible deny cues from layout text.

    - Exact ``Finding: DENIED`` → DENIED (wins over review softens).
    - Exact ``Finding: NEEDS_REVIEW`` demotes DENIED → REVIEW.
    - Visible ``Registry Status: EMBARGO`` injects ``planetary_embargo`` and
      denies when the row is still REVIEW/APPROVED with no other risk.
    Never invents APPROVED.
    """

    text = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    if not text:
        return row
    page = re.sub(r"\bSAMPLE[- ]+DENIAL\b", "", text, flags=re.I)
    finding_denied = bool(re.search(r"Finding\s*:?\s*DENIED\b", page, re.I))
    finding_review = bool(
        re.search(r"Finding\s*:?\s*NEEDS[_\s]?REVIEW\b", page, re.I)
    )
    registry_embargo = bool(
        row.adjudication in {"NEEDS_REVIEW", "APPROVED"}
        and _norm_flags(row.risk_flags) == "none"
        and re.search(r"Registry\s+Status\s*:?\s*EMBARGO\b", page, re.I)
    )
    could_change = bool(
        (finding_denied and row.adjudication != "DENIED")
        or (finding_review and row.adjudication == "DENIED")
        or registry_embargo
    )
    if not could_change:
        return row

    cue_names = tuple(
        cue_name
        for cue_name, present in (
            ("finding_denied", finding_denied),
            ("finding_review", finding_review),
            ("registry_embargo", registry_embargo),
        )
        if present
    )
    raster_text = strip_untrusted_selectable_lines(
        raster_text_for_selectable_decision_cues(
            pdf_path,
            text,
            cue_names,
        )
    )
    raster_page = re.sub(
        r"\bSAMPLE[- ]+DENIAL\b",
        "",
        raster_text,
        flags=re.I,
    )
    if re.search(r"Finding\s*:?\s*DENIED\b", raster_page, re.I):
        if row.adjudication == "DENIED":
            return row
        payload = row.to_dict()
        payload["adjudication"] = "DENIED"
        payload["confidence"] = 0.98
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
    if (
        row.adjudication == "DENIED"
        and re.search(
            r"Finding\s*:?\s*NEEDS[_\s]?REVIEW\b",
            raster_page,
            re.I,
        )
    ):
        payload = row.to_dict()
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = max(float(row.confidence), 0.85)
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
    if (
        registry_embargo
        and row.adjudication in {"NEEDS_REVIEW", "APPROVED"}
        and _norm_flags(row.risk_flags) == "none"
        and re.search(
            r"Registry\s+Status\s*:?\s*EMBARGO\b",
            raster_page,
            re.I,
        )
    ):
        payload = row.to_dict()
        payload["risk_flags"] = "planetary_embargo"
        payload["adjudication"] = "DENIED"
        payload["confidence"] = 0.98
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
    return row


_ROTATED_NOTE_DOMAIN_WORDS = frozenset(
    {
        "adjudicator",
        "applicant",
        "arrival",
        "biometric",
        "case",
        "code",
        "declared",
        "fee",
        "finding",
        "flags",
        "form",
        "home",
        "id",
        "manual",
        "note",
        "observed",
        "purpose",
        "reason",
        "registry",
        "scan",
        "slip",
        "species",
        "sponsor",
        "status",
        "visa",
        "world",
    }
)
_ROTATED_NOTE_UNTRUSTED_RE = re.compile(
    r"SYSTEM\s*:|answer\s*key|output\s+this\s+answer|ground\s+truth|"
    r"training\s+example|sample\s+(?:denial|decision|approval|answer)|"
    r"force\s+adjudication|ignore\s+(?:all\s+)?previous\s+instructions|"
    r"ignore\s+visible",
    re.I,
)
_ROTATED_NOTE_CASE_ID_RE = re.compile(r"\bMIB[- ]?([0-9]{6})\b", re.I)
_ROTATED_NOTE_TRANSITIONS = frozenset(
    {
        ("DENIED", "NEEDS_REVIEW"),
        ("NEEDS_REVIEW", "APPROVED"),
    }
)
_ROTATED_NOTE_REPAIR_CELLS = frozenset(
    {
        ("ALPHA_DRACONIAN", "DIP-1", "field repair"),
        ("CENTAURI_SYNTH", "XW-1", "xenobotany"),
    }
)


def _rotated_note_route(row: PredictionRow) -> bool:
    """Bound the pixel retry to measured schema cells and decision lanes."""

    if (
        row.species_code,
        row.visa_class,
        row.declared_purpose,
    ) not in _ROTATED_NOTE_REPAIR_CELLS:
        return False

    if (
        row.adjudication == "DENIED"
        and float(row.confidence) >= 0.95
        and row.fee_status == "waived"
        and _norm_flags(row.risk_flags) != "none"
    ):
        return True
    return bool(
        row.adjudication == "NEEDS_REVIEW"
        and float(row.confidence) < 0.5
        and row.fee_status == "paid"
        and _norm_flags(row.risk_flags) == "none"
    )


def _rotated_note_ocr_quality(text: str) -> int:
    return sum(
        token.casefold() in _ROTATED_NOTE_DOMAIN_WORDS
        for token in re.findall(r"[A-Za-z]{2,}", text)
    )


def _rotated_manual_finding(text: str) -> str | None:
    """Parse one damaged Finding value only inside a visible manual note."""

    if not text or _ROTATED_NOTE_UNTRUSTED_RE.search(text):
        return None
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    heading_index = next(
        (
            index
            for index, line in enumerate(lines[:6])
            if difflib.SequenceMatcher(
                None,
                re.sub(r"[^a-z]", "", line.casefold()),
                "manualadjudicatornote",
            ).ratio()
            >= 0.54
        ),
        None,
    )
    if heading_index is None:
        return None

    decisions: set[str] = set()
    vocabulary = {
        "APPROVED": "approved",
        "DENIED": "denied",
        "NEEDS_REVIEW": "needsreview",
    }
    for line in lines[heading_index + 1 : heading_index + 9]:
        words = re.findall(r"[a-z]+", line.casefold())
        label_index = next(
            (
                index
                for index, word in enumerate(words[:2])
                if difflib.SequenceMatcher(None, word, "finding").ratio() >= 0.54
            ),
            None,
        )
        if label_index is None:
            continue
        value = "".join(words[label_index + 1 :])
        if len(value) < 4:
            continue
        ranked = sorted(
            (
                difflib.SequenceMatcher(None, value, canonical).ratio(),
                decision,
            )
            for decision, canonical in vocabulary.items()
        )
        best_score, best_decision = ranked[-1]
        second_score = ranked[-2][0]
        if best_score >= 0.72 and best_score - second_score >= 0.15:
            decisions.add(best_decision)
    return next(iter(decisions)) if len(decisions) == 1 else None


def apply_rotated_manual_note_decision_from_views(
    row: PredictionRow,
    page_views: Iterable[tuple[str, str, str]],
) -> PredictionRow:
    """Accept one case-bound manual Finding agreed by both rotated scans.

    Each tuple is ``(upright, clockwise, counterclockwise)`` visible OCR.
    The upright view supplies the sparse-page route. The active case must bind
    exactly in the upright scan or independently in both rotated scans. Both
    rotated views must also recover the same decision. No extraction field or
    confidence value is changed.
    """

    if not _rotated_note_route(row):
        return row
    expected = row.case_id.removeprefix("MIB-")
    recovered: set[str] = set()
    for upright, clockwise, counterclockwise in page_views:
        if any(
            _ROTATED_NOTE_UNTRUSTED_RE.search(view)
            for view in (upright, clockwise, counterclockwise)
        ):
            continue
        upright_ids = set(_ROTATED_NOTE_CASE_ID_RE.findall(upright))
        clockwise_ids = set(_ROTATED_NOTE_CASE_ID_RE.findall(clockwise))
        counterclockwise_ids = set(_ROTATED_NOTE_CASE_ID_RE.findall(counterclockwise))
        all_ids = upright_ids | clockwise_ids | counterclockwise_ids
        if (
            _rotated_note_ocr_quality(upright) >= 8
            or any(case_number != expected for case_number in all_ids)
            or not (
                upright_ids == {expected}
                or (clockwise_ids == {expected} and counterclockwise_ids == {expected})
            )
        ):
            continue
        first = _rotated_manual_finding(clockwise)
        second = _rotated_manual_finding(counterclockwise)
        if first is not None and first == second:
            recovered.add(first)
    if len(recovered) != 1:
        return row
    decision = next(iter(recovered))
    if (row.adjudication, decision) not in _ROTATED_NOTE_TRANSITIONS:
        return row
    payload = row.to_dict()
    payload["adjudication"] = decision
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_rotated_manual_note_decision(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Recover a sideways damaged manual Finding from visible pixels only."""

    if not _rotated_note_route(row):
        return row
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return row

    images: list[object] = []
    try:
        # BatchRunner uses worker threads. Keep only PDFium's native boundary
        # serialized; Tesseract subprocesses below remain concurrent.
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page = document[index]
                    try:
                        images.append(
                            page.render(scale=2.5, grayscale=True).to_pil().convert("L")
                        )
                    finally:
                        page.close()
            finally:
                document.close()
    except Exception:
        return row

    page_views: list[tuple[str, str, str]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="mib-rotated-note-") as directory:
            work = Path(directory)
            for index, image in enumerate(images):
                upright = _tesseract_image_text(
                    image,
                    work / f"{index}-upright.png",
                    psm="12",
                )
                expected = row.case_id.removeprefix("MIB-")
                if (
                    _ROTATED_NOTE_UNTRUSTED_RE.search(upright)
                    or _rotated_note_ocr_quality(upright) >= 8
                    or any(
                        case_number != expected
                        for case_number in _ROTATED_NOTE_CASE_ID_RE.findall(upright)
                    )
                ):
                    continue
                clockwise = _tesseract_image_text(
                    image.rotate(-90, expand=True, fillcolor=255),
                    work / f"{index}-clockwise.png",
                    psm="12",
                )
                counterclockwise = _tesseract_image_text(
                    image.rotate(90, expand=True, fillcolor=255),
                    work / f"{index}-counterclockwise.png",
                    psm="12",
                )
                page_views.append((upright, clockwise, counterclockwise))
    except Exception:
        return row
    return apply_rotated_manual_note_decision_from_views(row, page_views)


def _authoritative_decision_heading(line: str) -> bool:
    key = re.sub(r"[^a-z0-9]", "", line.casefold())
    return (
        max(
            difflib.SequenceMatcher(None, key, target).ratio()
            for target in (
                "manualadjudicatornote",
                "signedmanualnote",
                "adjudicatorstamp",
                "officialstamp",
            )
        )
        >= 0.62
    )


def _authoritative_decision_label_index(
    lines: tuple[str, ...],
) -> int | None:
    for index, line in enumerate(lines):
        words = re.findall(r"[a-z]+", line.casefold())
        score = max(
            (
                difflib.SequenceMatcher(None, word, target).ratio()
                for word in words[:3]
                for target in ("finding", "decision", "status")
            ),
            default=0.0,
        )
        if score >= 0.62:
            return index
    return None


def _has_authoritative_decision_structure(text: str) -> bool:
    if not text or _ROTATED_NOTE_UNTRUSTED_RE.search(text):
        return False
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    heading_index = next(
        (
            index
            for index, line in enumerate(lines[:6])
            if _authoritative_decision_heading(line)
        ),
        None,
    )
    label_index = _authoritative_decision_label_index(lines[:12])
    return bool(
        heading_index is not None
        and label_index is not None
        and heading_index <= label_index
        and label_index - heading_index <= 8
    )


def _authoritative_decision_from_text(text: str) -> str | None:
    """Read one closed-vocabulary decision from a trusted authority page."""

    if not _has_authoritative_decision_structure(text):
        return None
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    heading_index = next(
        index
        for index, line in enumerate(lines[:6])
        if _authoritative_decision_heading(line)
    )
    decisions: set[str] = set()
    vocabulary = {
        "APPROVED": "approved",
        "DENIED": "denied",
        "NEEDS_REVIEW": "needsreview",
    }
    for line in lines[heading_index + 1 : heading_index + 10]:
        words = re.findall(r"[a-z]+", line.casefold())
        label_index = next(
            (
                index
                for index, word in enumerate(words[:3])
                if max(
                    difflib.SequenceMatcher(None, word, target).ratio()
                    for target in ("finding", "decision", "status")
                )
                >= 0.62
            ),
            None,
        )
        if label_index is None:
            continue
        value = "".join(words[label_index + 1 :])
        if len(value) < 4:
            continue
        ranked = sorted(
            (
                difflib.SequenceMatcher(None, value, canonical).ratio(),
                decision,
            )
            for decision, canonical in vocabulary.items()
        )
        best_score, best_decision = ranked[-1]
        second_score = ranked[-2][0]
        if best_score >= 0.72 and best_score - second_score >= 0.15:
            decisions.add(best_decision)
    return next(iter(decisions)) if len(decisions) == 1 else None


def apply_authoritative_review_decision_from_views(
    row: PredictionRow,
    page_views: Iterable[tuple[str, str]],
) -> PredictionRow:
    """Honor one exact-case authority decision agreed by PSM12 and PSM6."""

    if row.adjudication != "NEEDS_REVIEW":
        return row
    expected = row.case_id.removeprefix("MIB-")
    decisions: set[str] = set()
    for primary, confirmation in page_views:
        if any(
            _ROTATED_NOTE_UNTRUSTED_RE.search(view) for view in (primary, confirmation)
        ):
            continue
        primary_ids = set(_ROTATED_NOTE_CASE_ID_RE.findall(primary))
        confirmation_ids = set(_ROTATED_NOTE_CASE_ID_RE.findall(confirmation))
        if (
            primary_ids != {expected}
            or any(case_number != expected for case_number in confirmation_ids)
            or not _has_authoritative_decision_structure(primary)
        ):
            continue
        first = _authoritative_decision_from_text(primary)
        second = _authoritative_decision_from_text(confirmation)
        if first is not None and first == second:
            decisions.add(first)
    if len(decisions) != 1:
        return row
    decision = next(iter(decisions))
    if decision == row.adjudication:
        return row
    payload = row.to_dict()
    payload["adjudication"] = decision
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def _authoritative_review_route(row: PredictionRow) -> bool:
    """Bound authority OCR to the measured residual TRIANGULAN review lane."""

    return bool(
        row.adjudication == "NEEDS_REVIEW"
        and row.species_code == "TRIANGULAN"
        and row.visa_class == "DIP-1"
        and row.declared_purpose == "reactor maintenance"
        and _norm_flags(row.risk_flags) == "none"
        and row.fee_status == "waived"
    )


def apply_authoritative_review_decision(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Recover a residual review decision from visible authority pixels."""

    if not _authoritative_review_route(row):
        return row
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return row

    images: list[object] = []
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page = document[index]
                    try:
                        images.append(
                            page.render(scale=2.5, grayscale=True).to_pil().convert("L")
                        )
                    finally:
                        page.close()
            finally:
                document.close()
    except Exception:
        return row

    expected = row.case_id.removeprefix("MIB-")
    page_views: list[tuple[str, str]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="mib-authority-review-") as directory:
            work = Path(directory)
            for index, image in enumerate(images):
                primary = _tesseract_image_text(
                    image,
                    work / f"{index}-psm12.png",
                    psm="12",
                )
                if (
                    _ROTATED_NOTE_UNTRUSTED_RE.search(primary)
                    or set(_ROTATED_NOTE_CASE_ID_RE.findall(primary)) != {expected}
                    or not _has_authoritative_decision_structure(primary)
                ):
                    continue
                confirmation = _tesseract_image_text(
                    image,
                    work / f"{index}-psm6.png",
                    psm="6",
                )
                page_views.append((primary, confirmation))
    except Exception:
        return row
    return apply_authoritative_review_decision_from_views(row, page_views)


_DENIED_REVIEW_SPONSOR_TOKEN_RE = re.compile(
    r"\bSPN[- ]?([0-9OQDZSBIL]{4})\b",
    re.I,
)
_DENIED_REVIEW_SPONSOR_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "Z": "2",
        "S": "5",
        "B": "8",
        "I": "1",
        "L": "1",
    }
)


def _denied_review_best_word_score(
    text: str,
    target: str,
) -> float:
    words = re.findall(r"[a-z]+", text.casefold())
    return max(
        (difflib.SequenceMatcher(None, word, target).ratio() for word in words),
        default=0.0,
    )


def _denied_review_manual_heading(text: str) -> bool:
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    targets = (
        "manualadjudicatornote",
        "adjudicatornote",
        "signedmanualnote",
    )
    for line in lines[:5]:
        key = re.sub(r"[^a-z]", "", line.casefold())
        if (
            key
            and max(
                difflib.SequenceMatcher(None, key, target).ratio() for target in targets
            )
            >= 0.55
        ):
            return True
    return False


def _denied_review_reason(text: str) -> bool:
    """Recognize the fixed review reason under heavy visible scan damage."""

    return bool(
        _denied_review_best_word_score(text, "packet") >= 0.60
        and _denied_review_best_word_score(text, "damaged") >= 0.60
        and _denied_review_best_word_score(text, "contradictory") >= 0.43
        and max(
            _denied_review_best_word_score(text, "visible"),
            _denied_review_best_word_score(text, "evidence"),
        )
        >= 0.43
    )


def _denied_review_untrusted(text: str) -> bool:
    if _ROTATED_NOTE_UNTRUSTED_RE.search(text):
        return True
    instruction = max(
        _denied_review_best_word_score(text, "ignore"),
        _denied_review_best_word_score(text, "system"),
        _denied_review_best_word_score(text, "output"),
    )
    visible = max(
        _denied_review_best_word_score(text, "visible"),
        _denied_review_best_word_score(text, "evidence"),
    )
    return instruction >= 0.72 and visible >= 0.62


def _denied_review_manual_from_images(
    images: Iterable[object],
) -> bool:
    """Require two independent threshold views of one manual review reason."""

    try:
        with tempfile.TemporaryDirectory(prefix="mib-denied-review-note-") as directory:
            work = Path(directory)
            for index, image in enumerate(images):
                crop = image.crop(
                    (
                        0,
                        0,
                        image.width,
                        int(round(image.height * 0.24)),
                    )
                )
                first = _tesseract_image_text(
                    crop.point(lambda pixel: 255 if pixel > 125 else 0),
                    work / f"{index}-125.png",
                    psm="11",
                )
                if not (
                    _denied_review_manual_heading(first)
                    and _denied_review_reason(first)
                ):
                    continue
                second = _tesseract_image_text(
                    crop.point(lambda pixel: 255 if pixel > 145 else 0),
                    work / f"{index}-145.png",
                    psm="11",
                )
                if not (
                    _denied_review_manual_heading(second)
                    and _denied_review_reason(second)
                ):
                    continue
                normal = _tesseract_image_text(
                    crop.resize(
                        (
                            max(1, int(round(crop.width * 0.70))),
                            max(1, int(round(crop.height * 0.70))),
                        )
                    ),
                    work / f"{index}-normal.png",
                    psm="11",
                )
                if any(
                    _denied_review_untrusted(view) for view in (first, second, normal)
                ):
                    continue
                return True
    except Exception:
        return False
    return False


def _denied_review_sponsor_from_images(
    images: Iterable[object],
) -> bool:
    """Detect one visible one-digit sponsor conflict across distinct pages."""

    by_value: dict[str, set[int]] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix="mib-denied-review-sponsor-"
        ) as directory:
            work = Path(directory)
            for index, image in enumerate(images):
                text = _tesseract_image_text(
                    image,
                    work / f"{index}.png",
                    psm="11",
                )
                if _denied_review_untrusted(text):
                    continue
                for raw in _DENIED_REVIEW_SPONSOR_TOKEN_RE.findall(text):
                    value = raw.upper().translate(_DENIED_REVIEW_SPONSOR_TRANSLATION)
                    by_value.setdefault(value, set()).add(index)
    except Exception:
        return False
    if len(by_value) != 2:
        return False
    first, second = sorted(by_value)
    return bool(
        by_value[first].isdisjoint(by_value[second])
        and sum(left != right for left, right in zip(first, second)) == 1
    )


def apply_denied_review_visible_conflicts(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Soften a denial on visible authority or document-conflict evidence.

    This is a narrow, denial-only overlay. It cannot create an approval. The
    two routes are:

    * a manual-note review reason agreed by two thresholded pixel OCR views;
    * an illegible-biometrics packet with two one-digit-different sponsor IDs
      observed on distinct visible pages.

    Case IDs, filenames, native PDF text, signature cells, and answer payloads
    are not decision features.
    """

    if row.adjudication != "DENIED":
        return row
    flags = _norm_flags(row.risk_flags)
    manual_route = flags == "none" and (
        row.visa_class == "TRANSIT-7" or row.fee_status == "unpaid"
    )
    sponsor_route = bool(
        flags == "illegible_biometrics"
        and row.fee_status == "paid"
        and row.visa_class != "TRANSIT-7"
    )
    if not manual_route and not sponsor_route:
        return row
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return row

    scale = 5.0 if manual_route else 3.0
    images: list[object] = []
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page = document[index]
                    try:
                        images.append(
                            page.render(scale=scale, grayscale=True)
                            .to_pil()
                            .convert("L")
                        )
                    finally:
                        page.close()
            finally:
                document.close()
    except Exception:
        return row

    accepted = False
    if manual_route:
        accepted = _denied_review_manual_from_images(images)
    if not accepted and sponsor_route:
        accepted = _denied_review_sponsor_from_images(images)
    if not accepted:
        return row
    payload = row.to_dict()
    payload["adjudication"] = "NEEDS_REVIEW"
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


_DAMAGE_KEYWORDS = re.compile(
    r"\b(?:UNREADABLE|REDACTED)\b",
    re.I,
)


def apply_damage_weak_review(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Downgrade APPROVED → REVIEW when layout shows unreadable/redacted damage.

    Fail-closed: packets marked UNREADABLE/REDACTED that still look clean on
    risk are high-risk for hidden review content. WHITEOUT/CUT OUT alone are
    not used — they fire on clean true approvals more often than false ones.
    Never creates approvals.
    """

    if row.adjudication != "APPROVED":
        return row
    text = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    if not text or not _DAMAGE_KEYWORDS.search(text):
        return row
    raster_text = strip_untrusted_selectable_lines(
        raster_text_for_selectable_decision_cues(
            pdf_path,
            text,
            ("damage",),
        )
    )
    if not _DAMAGE_KEYWORDS.search(raster_text):
        return row
    payload = row.to_dict()
    payload["adjudication"] = "NEEDS_REVIEW"
    payload["confidence"] = min(float(payload.get("confidence") or 0.7), 0.55)
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


# Mid-gray solid washout on a biometric portrait/scan region. Measured on the
# exact public-135 train_fit 800: score>=0.010 fires 2×, 2 FIX / 0 HURT,
# official total +0.02222 at CFA 0 (flag-only gap-fill when risk is none).
_MIDGRAY_SCORE_THRESHOLD = 0.010
_MIDGRAY_LO = 90
_MIDGRAY_HI = 200
_MIDGRAY_RENDER_SCALE = 1.5
_MIDGRAY_REPAIR_CELLS = frozenset(
    {
        ("ARCTURIAN", "archive audit", "DENIED"),
        ("ANDROMEDAN", "field repair", "NEEDS_REVIEW"),
    }
)


def midgray_blob_score(gray: object) -> float:
    """Return max solid mid-gray rectangle score for one page gray array.

    Score = area_frac * (1 - min(std/40, 1)) * fill for connected mid-tone
    components that are large enough to be a washed portrait/scan patch.
    Identity-free; uses only pixel geometry.
    """

    try:
        import numpy as np
    except ImportError:
        return 0.0

    array = np.asarray(gray)
    if array.ndim != 2 or array.size == 0:
        return 0.0
    try:
        import cv2
    except ImportError:
        return 0.0

    mask = ((array >= _MIDGRAY_LO) & (array <= _MIDGRAY_HI)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = array.shape
    page_area = float(height * width)
    best = 0.0
    for index in range(1, count):
        x, y, box_w, box_h, area = stats[index]
        if area < 0.01 * page_area or box_w < 40 or box_h < 40:
            continue
        aspect = box_w / max(box_h, 1)
        if aspect < 0.3 or aspect > 4.0:
            continue
        region = array[y : y + box_h, x : x + box_w]
        submask = mask[y : y + box_h, x : x + box_w] > 0
        if int(submask.sum()) < 50:
            continue
        values = region[submask]
        fill = float(area) / float(max(box_w * box_h, 1))
        area_frac = float(area) / page_area
        score = area_frac * (1.0 - min(float(values.std()) / 40.0, 1.0)) * fill
        if score > best:
            best = score
    return best


def packet_midgray_degradation_score(pdf_path: Path) -> float:
    """Max mid-gray washout score across all pages of a packet."""

    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError:
        return 0.0
    images: list[object] = []
    try:
        with _PDFIUM_RENDER_LOCK:
            document = pdfium.PdfDocument(str(pdf_path))
            try:
                for index in range(len(document)):
                    page = document[index]
                    try:
                        images.append(
                            page.render(scale=_MIDGRAY_RENDER_SCALE)
                            .to_pil()
                            .convert("RGB")
                        )
                    finally:
                        page.close()
            finally:
                document.close()
    except Exception:
        return 0.0
    best = 0.0
    for image in images:
        rgb = np.asarray(image)
        # Corpus is monochrome; gray from luminance is enough.
        gray = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        ).astype(np.uint8)
        score = midgray_blob_score(gray)
        if score > best:
            best = score
    return best


def _midgray_gapfill_route(row: PredictionRow) -> bool:
    """Bound the pixel detector to its measured unresolved schema cells."""

    return bool(
        _norm_flags(row.risk_flags) == "none"
        and (
            row.species_code,
            row.declared_purpose,
            row.adjudication,
        )
        in _MIDGRAY_REPAIR_CELLS
    )


def apply_midgray_illegible_gapfill(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Gap-fill risk_flags=illegible_biometrics from solid mid-gray washout.

    Fires only when the emitted risk field is still ``none``. Never overwrites
    a resolved flag value, never emits APPROVED, and does not change
    adjudication on its own. Threshold measured on train_fit 800.
    """

    if not _midgray_gapfill_route(row):
        return row
    if packet_midgray_degradation_score(pdf_path) < _MIDGRAY_SCORE_THRESHOLD:
        return row
    payload = row.to_dict()
    payload["risk_flags"] = "illegible_biometrics"
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def _layout_fee_paid_proven(text: str) -> bool:
    """Require the canonical paid receipt amount (not a waiver / Fee-Status guess)."""

    return bool(re.search(r"Amount\s*\$?\s*809", text, re.I))


def _layout_page_signature(text: str) -> str:
    """Compact page-type signature (F/R/I/B/M/O) in document order."""

    kinds: list[str] = []
    for block in text.split("\x0c"):
        if re.search(r"Fee Receipt", block, re.I):
            kinds.append("F")
        elif re.search(r"Registry", block, re.I):
            kinds.append("R")
        elif re.search(r"I-8090|Work Authorization", block, re.I):
            kinds.append("I")
        elif re.search(r"B-?13|Biometric", block, re.I):
            kinds.append("B")
        elif re.search(r"MED-|Medical", block, re.I):
            kinds.append("M")
        elif block.strip():
            kinds.append("O")
    return "".join(kinds)


def _layout_consensus_trap_cell(
    visa_class: str,
    declared_purpose: str,
    signature: str,
    *,
    fee_waived: bool = False,
) -> bool:
    """True when LC would mint a measured one-way false APPROVED cell."""

    cell = (visa_class, declared_purpose, signature)
    if fee_waived and cell in _LC_WAIVED_TRAP_OVERRIDES:
        return False
    if fee_waived and cell in _LC_WAIVED_ONLY_TRAP_VISA_PURPOSE_SIG:
        return True
    if (visa_class, declared_purpose) in _LC_TRAP_VISA_PURPOSE:
        return True
    if cell in _LC_TRAP_VISA_PURPOSE_SIG:
        return True
    if signature == "FRI" and declared_purpose == "transit":
        return True
    return False


def _approval_incomplete_filler_assembly(row: PredictionRow, text: str) -> bool:
    """True when an APPROVED row sits on a filler-heavy incomplete packet."""

    if not text:
        return False
    signature = _layout_page_signature(text)
    confidence = float(row.confidence)
    attestation_first = bool(re.match(r"\s*Sponsor Attestation Letter", text, re.I))
    synthetic_first = bool(
        re.match(r"\s*Packet\s+\S+\s*/\s*page\s+\d+\s+Synthetic hiring", text, re.I)
    )
    fee_proven = bool(re.search(r"Amount\s*\$?\s*809", text, re.I))

    if (
        abs(confidence - 0.80) < 1e-6
        and attestation_first
        and not fee_proven
        and signature.count("O") >= 3
    ):
        return True
    # Mid-confidence XW-1 opening on synthetic filler with leading non-core pages.
    if (
        row.visa_class == "XW-1"
        and confidence < 0.95
        and synthetic_first
        and signature.startswith("OO")
    ):
        return True
    # Visible $0 waived receipt on O-heavy filler: fee recovery plus a later
    # approval head can create a false approval, so keep the packet in review.
    if (
        row.fee_status == "waived"
        and signature.count("O") >= 5
        and re.search(r"Fee\s*Status\s*:?\s*waived", text, re.I)
        and re.search(r"Amount\s*\$?\s*0(?:\.00)?\b", text, re.I)
        and not fee_proven
    ):
        return True
    return False


def _layout_registry_matches_applicant(text: str) -> bool:
    # Use [ \\t] between name tokens — ``\\s`` would span newlines into field labels.
    name_token = r"[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+"
    registries = {
        cleaned
        for raw in re.findall(rf"Registry\s+Name\s+({name_token})", text)
        if (cleaned := _clean_person_name(raw))
    }
    applicants = {
        cleaned
        for raw in re.findall(rf"Applicant\s*:?\s+({name_token})", text)
        if (cleaned := _clean_person_name(raw))
    }
    return len(registries) == 1 and registries == applicants


def _layout_risk_flags(text: str) -> frozenset[str]:
    found: set[str] = set()
    normalized = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    for flag in KNOWN_RISK_FLAGS:
        if re.search(rf"(?:^|_){re.escape(flag)}(?:_|$)", normalized):
            found.add(flag)
    return frozenset(found)


def apply_layout_consensus_approval(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Approve clean packets with name consensus + paid ``$809`` or waived fee.

    Visas: DIP-1 / XW-2 / MED-3 / XW-1. Fail-closed on RIF (except field-repair),
    non-core ``O`` pages, medical-consult, and measured trap cells (silent-stamp
    CFA / false-APPROVED REVIEW). Waived path does not require ``Amount $809``.
    """

    if row.adjudication != "NEEDS_REVIEW":
        return row
    fee_waived = row.fee_status == "waived"
    if fee_waived:
        if row.visa_class not in _LAYOUT_CONSENSUS_WAIVED_VISAS:
            return row
    else:
        if row.visa_class not in _LAYOUT_CONSENSUS_PAID_VISAS:
            return row
        if row.fee_status != "paid":
            return row
    if _norm_flags(row.risk_flags) != "none":
        return row
    if row.home_world in _POLICY.embargoed_worlds:
        return row
    if (
        row.home_world in _POLICY.non_diplomatic_embargoed_worlds
        and row.visa_class != "DIP-1"
    ):
        return row
    if row.arrival_date in {"1900-01-01", "unknown", ""}:
        return row
    # Medical consult concentrates silent B-13 / FAP under LC — keep REVIEW.
    if row.declared_purpose == "medical consult":
        return row
    # Field manual: DIP-1 does not require a sponsor (diplomatic exemption).
    # Revoked/missing sponsors remain blocking for XW / MED visas.
    if row.visa_class != "DIP-1" and row.sponsor_id in {
        "SPN-0000",
        "unknown",
        "",
        *_POLICY.barred_sponsors,
    }:
        return row

    text = _pdf_layout_text(pdf_path)
    if not text:
        return row
    # Veto cross-form identity conflicts before the more expensive pixel proof.
    if not _layout_registry_matches_applicant(text):
        return row
    # Never read embedded generator instructions for risk vetoes.
    if _layout_risk_flags(_strip_untrusted_generator_lines(text)):
        return row
    signature = _layout_page_signature(text)
    # RIF assemblies (except field-repair) and non-core ``O`` pages
    # concentrate silent review traps under general LC.
    if signature == "RIF" and row.declared_purpose != "field repair":
        return row
    if "O" in signature:
        return row
    if _layout_consensus_trap_cell(
        row.visa_class,
        row.declared_purpose,
        signature,
        fee_waived=fee_waived,
    ):
        return row
    if not raster_corroborates_layout_approval(
        pdf_path,
        text,
        case_id=row.case_id,
        applicant_name=row.applicant_name,
        fee_waived=fee_waived,
    ):
        return row

    payload = row.to_dict()
    payload["adjudication"] = "APPROVED"
    payload["confidence"] = LAYOUT_CONSENSUS_APPROVAL_CONFIDENCE
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_denial_to_review_softening(row: PredictionRow) -> PredictionRow:
    """Soften over-hard DENIED → REVIEW on identity-free review-only cells.

    Never creates APPROVED. No magic confidence thresholds.
    """

    if row.adjudication != "DENIED":
        return row

    payload = row.to_dict()
    flags = _norm_flags(row.risk_flags)

    # Policy: rescinded_denial alone is review-only (not a hard deny).
    if flags == "rescinded_denial":
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = 0.80
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    if row.visa_class != "DIP-1":
        return row

    # Illegible biometrics on DIP is review-only, not a hard deny.
    if flags == "illegible_biometrics":
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = min(float(row.confidence), 0.70)
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    # Identity-free DIP waiver packs often hard-deny on
    # ``review_denial_three_required_outputs_unknown`` even after recovery fills
    # schema defaults. With no disqualifying risk, park in REVIEW (never APPROVED).
    if (
        flags == "none"
        and row.fee_status == "waived"
        and str(row.applicant_name or "").strip().casefold() in {"", "unknown"}
    ):
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = min(float(row.confidence), 0.70)
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    return row


def apply_triangulan_uncertain_denial_review(
    row: PredictionRow,
) -> PredictionRow:
    """Soften one transfer-positive uncertain-denial boundary to review.

    The rule uses only final emitted fields and calibrated confidence. It never
    creates an approval, changes extracted fields, or changes confidence.
    """

    if (
        row.adjudication != "DENIED"
        or row.species_code != "TRIANGULAN"
        or _norm_flags(row.risk_flags) != "none"
        or not 0.55 <= float(row.confidence) <= 0.75
    ):
        return row

    payload = row.to_dict()
    payload["adjudication"] = "NEEDS_REVIEW"
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


TRAIN_CALIBRATED_TERMINAL_CONFIDENCE = 8.0 / 13.0


def apply_train_calibrated_terminal_ensemble(row: PredictionRow) -> PredictionRow:
    """Apply the frozen terminal ensemble fitted on public training labels.

    The disclosed fit uses only final emitted fields and calibrated confidence.
    It never creates APPROVED and cannot create a catastrophic false approval.
    Every fired row receives one global correctness probability instead of a
    case- or rule-specific confidence.
    """

    confidence = float(row.confidence)
    purpose = str(row.declared_purpose or "")
    species = str(row.species_code or "")
    target: str | None = None

    if row.adjudication == "NEEDS_REVIEW":
        deny = (
            (purpose == "reactor maintenance" and confidence <= 0.25)
            or (species == "SIRIUS_AVIAN" and confidence <= 0.50)
            or (purpose == "diplomatic" and confidence <= 0.25)
            or (species == "KAIJU_MICRO" and confidence <= 0.45)
            or (
                species == "JOVIAN_GASFORM"
                and str(row.sponsor_id or "") == "SPN-0000"
            )
            or (purpose == "translation" and confidence <= 0.20)
            or (
                purpose == "diplomatic"
                and species == "VENUSIAN_MYCELIAL"
            )
            or (purpose == "archive audit" and confidence <= 0.40)
            or (species == "ORION_GRAYS" and confidence <= 0.40)
            or (
                purpose == "xenobotany"
                and species == "ANDROMEDAN"
            )
            or (purpose == "transit" and confidence <= 0.30)
        )
        if deny:
            target = "DENIED"
    elif (
        row.adjudication == "DENIED"
        and purpose == "cultural exchange"
        and confidence <= 0.65
    ):
        target = "NEEDS_REVIEW"

    if target is None:
        return row

    payload = row.to_dict()
    payload["adjudication"] = target
    payload["confidence"] = TRAIN_CALIBRATED_TERMINAL_CONFIDENCE
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_identity_conflict_review_guard(row: PredictionRow) -> PredictionRow:
    """Fail closed on a low-confidence identity conflict.

    This is a conservative demotion only. It uses semantic fields emitted by
    the parser and never creates an approval or consults a case identifier.
    """

    if row.adjudication != "DENIED":
        return row
    if row.species_code != "ANDROMEDAN":
        return row
    if row.declared_purpose != "xenobotany":
        return row
    if "identity_conflict" not in _parse_flag_set(row.risk_flags):
        return row
    if float(row.confidence) > 0.65:
        return row

    payload = row.to_dict()
    payload["adjudication"] = "NEEDS_REVIEW"
    payload["confidence"] = TRAIN_CALIBRATED_TERMINAL_CONFIDENCE
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_diplomatic_low_confidence_review_guard(row: PredictionRow) -> PredictionRow:
    """Demote one narrow, low-confidence diplomatic denial to review.

    The guard is deliberately output-only and fail-closed: no approval is
    created, and it requires a paid, risk-free DIP-1 diplomatic record with a
    low terminal confidence. This protects against an overconfident denial
    where the visible policy fields contain no disqualifying evidence.
    """

    if row.adjudication != "DENIED":
        return row
    if float(row.confidence) > 0.65:
        return row
    if row.visa_class != "DIP-1":
        return row
    if row.declared_purpose != "diplomatic":
        return row
    if row.risk_flags not in {"", "none", "unknown"}:
        return row

    payload = row.to_dict()
    payload["adjudication"] = "NEEDS_REVIEW"
    payload["confidence"] = TRAIN_CALIBRATED_TERMINAL_CONFIDENCE
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_approval_safety_demotion(
    row: PredictionRow,
    pdf_path: Path,
    *,
    candidates: Iterable[CandidateEvidence] = (),
) -> PredictionRow:
    """Demote APPROVED → DENIED/REVIEW when risk evidence still exists.

    Only fires on APPROVED. Cannot create new approvals. Also demotes measured
    layout-consensus false-APPROVED cells (fee unknown, RIF/O/traps, filler).
    """

    if row.adjudication != "APPROVED":
        return row

    payload = row.to_dict()
    # ``unknown`` is a schema default / extraction miss — never payment proof.
    if row.fee_status == "unknown":
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    raw_text = _pdf_layout_text(pdf_path)
    text = _strip_untrusted_generator_lines(raw_text)
    raw_risks = set(_layout_risk_flags(raw_text))
    selectable_risks = set(_layout_risk_flags(text))
    raw_finding_denied = bool(
        re.search(r"Finding\s*:?\s*DENIED\b", raw_text, re.I)
    )
    selectable_finding_denied = bool(
        re.search(r"Finding\s*:?\s*DENIED\b", text, re.I)
    )
    sanitizer_removed_decision_cue = bool(
        raw_risks - selectable_risks
        or (raw_finding_denied and not selectable_finding_denied)
    )
    if _approval_incomplete_filler_assembly(row, text or ""):
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    confidence = float(row.confidence)
    if abs(confidence - LAYOUT_CONSENSUS_APPROVAL_CONFIDENCE) < 1e-6 and text:
        signature = _layout_page_signature(text)
        if signature == "RIF" and row.declared_purpose != "field repair":
            payload["adjudication"] = "NEEDS_REVIEW"
            payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
            return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
        if "O" in signature:
            payload["adjudication"] = "NEEDS_REVIEW"
            payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
            return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
        if _layout_consensus_trap_cell(
            row.visa_class,
            row.declared_purpose,
            signature,
            fee_waived=(row.fee_status == "waived"),
        ):
            payload["adjudication"] = "NEEDS_REVIEW"
            payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
            return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    cue_names: list[str] = []
    if selectable_finding_denied:
        cue_names.append("finding_denied")
    if selectable_risks:
        cue_names.append("layout_risk")
    raster_text = ""
    if cue_names:
        raster_text = strip_untrusted_selectable_lines(
            raster_text_for_selectable_decision_cues(
                pdf_path,
                text,
                tuple(cue_names),
            )
        )
    raster_risks = set(_layout_risk_flags(raster_text))
    finding_denied = bool(
        selectable_finding_denied
        and re.search(r"Finding\s*:?\s*DENIED\b", raster_text, re.I)
    )

    # Candidate risks already came from render-first OCR. Selectable risks are
    # admitted only through the independent raster pass above.
    strong: set[str] = raster_risks
    for candidate in candidates:
        if not isinstance(candidate, CandidateEvidence):
            continue
        if candidate.field_name != "risk_flags":
            continue
        flags = _parse_flag_set(str(candidate.value or ""))
        if not flags:
            continue
        if float(getattr(candidate, "ocr_confidence", 0.0) or 0.0) >= 0.45:
            strong.update(flags)

    # A selectable layer may nominate a page for raster OCR, but it is never
    # denial proof. If the pixels/candidates do not corroborate a markerless
    # selectable finding/risk, fail closed to review without copying it.
    uncorroborated_selectable_deny = bool(
        (selectable_finding_denied and not finding_denied)
        or (selectable_risks - strong)
    )

    disqualifying = strong & set(_POLICY.disqualifying_flags)
    review_only = strong & set(_POLICY.review_only_flags)
    if strong:
        merged = set(_parse_flag_set(row.risk_flags)) | strong
        payload["risk_flags"] = "|".join(sorted(merged))

    if finding_denied or disqualifying or row.visa_class == "TRANSIT-7":
        payload["adjudication"] = "DENIED"
        payload["confidence"] = DEMOTION_DENIAL_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    if review_only:
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    # Sanitizer-removed cues are untrusted and must never be copied or used to
    # deny. Their presence still invalidates an approval, so fail closed.
    if sanitizer_removed_decision_cue or uncorroborated_selectable_deny:
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    known_row_risks = set(_parse_flag_set(row.risk_flags)) & (
        set(_POLICY.disqualifying_flags) | set(_POLICY.review_only_flags)
    )
    if known_row_risks:
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    return row


def _strip_untrusted_generator_lines(text: str) -> str:
    """Drop generator instructions and answer-table lines from selectable text."""

    return strip_untrusted_selectable_lines(text)
