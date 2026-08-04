# Technical memo

## Result

Safe-plus checkpoint source `e5f419333e755c3487b0ef4bcfafb7e72066b056` scored
**137.22837960254876 / 150** on all 1,000 public training packets. The
container produced 1,000 valid records with zero missing, extra, duplicate,
blank, or invalid rows and zero catastrophic false approvals.

| Component | Score |
| --- | ---: |
| Field extraction | 45.1177777778 / 50 |
| Adjudication | 74.2900000000 / 80 |
| Calibration | 17.8206018248 / 20 |
| Mean Brier error | 0.0544849544 |
| Total | **137.2283796025 / 150** |

Checkpoint prediction SHA-256:
`9a675507c686fca0544802d10f260e70ab77f0ee3b2219a2e56d93a2e7a0620a`.
The safe-plus image was built locally for smoke testing under the official
four-CPU, 8 GiB, network-disabled, read-only-root constraints. A complete
1,000-case runtime receipt for this source lineage has not been re-earned yet.

This is public-training evidence, not a private-test score or ranking
guarantee.

## System

The base is a render-first document pipeline. `pypdfium2` rasterizes pages;
RapidOCR performs primary OCR; bounded Tesseract passes recover difficult
regions. Candidates retain document type, geometry, OCR confidence, and visual
cues. Cross-page linking and precedence rules resolve fields before a
conservative adjudication engine and calibrated output layer.

The finalizer is separate from the OCR core. It applies, in order:

1. identity-free fee, name, visa, sponsor, date, and purpose repairs;
2. visible-layout clean-packet consensus;
3. narrowly gated blue-slash and red-watermark denial signals;
4. explicit finding and document-damage decisions;
5. approval safety demotions and denial softening;
6. a second explicit-finding pass and a `TRANSIT-7` hard gate;
7. identity-free confidence blending and calibration.

Raster OCR and visible geometry are the primary evidence path. Sanitized
native selectable text may nominate pages, regions, or field candidates. It is
not authoritative for approval or denial: decision-changing selectable cues
require independent raster or pixel corroboration. Some field recovery still
uses sanitized selectable layout, so this is not a claim that arbitrary hidden
text can never influence extraction.

Keeping the finalizer separate made the full 1,000-case run stable. A
monolithic variant with additional OCR heads repeatedly exited with signal 139
during long runs, while the stable core plus layout finalizer completed the
same workload. The `pypdfium2` fallback also preserves form-feed page
boundaries; without that fix, page-signature safety gates were silently
disabled in the submission image.

## Public-train selection disclosure

The terminal ensemble was selected against public training labels. It contains
eleven `NEEDS_REVIEW` to `DENIED` boundaries and one `DENIED` to
`NEEDS_REVIEW` boundary over emitted schema fields and confidence. It contains
no case IDs, filenames, hashes, names, sponsor identities, or per-case
predictions; it never creates `APPROVED`; changed rows receive confidence
`8/13`.

A grouped nested transfer ablation did not show a reliable positive mean lift.
That negative result is the principal transfer warning: the public score is
measured, but the terminal ensemble's public lift is not an unbiased estimate
of private performance.

Other public-training-selected tables use identity-free layout, visa, purpose,
fee, risk, and page-order features. Runtime source contains no validation case
IDs, filenames, hashes, applicant identities, sponsor identities, answer-key
module, or per-case prediction table. Tests enforce these invariants.

The safe-plus checkpoint adds two fail-closed review guards: a low-confidence
ANDROMEDAN/xenobotany identity-conflict denial and a low-confidence paid DIP-1
diplomatic denial with no emitted risk flags. Neither guard creates APPROVED;
each only changes a narrow DENIED result to `NEEDS_REVIEW`.

## Final validation

The safe-plus source has a format-valid 5,000-row preflight artifact, but its
measured wall time exceeded the 30,000-second limit. Until a compliant sealed
receipt exists, no completion or private-performance claim is made.

Final 5,000-packet receipt pending: `<FINAL5K_ROWS>`,
`<FINAL5K_RUNTIME_SECONDS>`, `<FINAL5K_PREDICTION_BYTES>`,
`<FINAL5K_SHA256>`, `<FINAL5K_VALIDATOR_RESULT>`.

## Failure modes

- Layout promotion depends on recurring visible form structure. A private
  distribution with different templates may reduce the measured public gain.
- OCR can still miss faint, rotated, occluded, or handwritten evidence.
- Some packets do not visibly contain decision-critical facts. Those remain
  review rather than being guessed.
- Public-training-selected identity-free tables can overfit even though they
  never use case identity.
- Confidence calibration may drift if the private class mix differs.
- Watermark OCR may miss new colors, fonts, or severe scan degradation.

The fail-closed behavior is deliberate: uncertain packets go to
`NEEDS_REVIEW`, and visible disqualifiers dominate.

## With another week

I would replace the remaining train-selected terminal boundaries with a
grouped nested model that demonstrates positive held-out transfer, expand
pixel-provenance coverage for extraction fields, and profile the remaining
OCR-heavy cases. Any new approval path would still require zero catastrophic
false approvals in every development partition.
