# Attribution and provenance

This solution combines and modifies public MIT-licensed challenge entries.
Validation predictions are generated locally by this repository; no other
participant's validation output is copied.

## Organizer materials

- Repository: <https://github.com/8090-inc/mib-doc-challenge>
- Pinned commit: `38ce8883dea9f87c27a8a95f134e54fe8b673064`
- License: MIT

The schemas, evaluator, Docker runner, field manual, public labels, and
synthetic dataset are organizer materials.

## Render-first OCR core

- Chris Strobl: <https://github.com/strobl/mib-doc-solution>
- Pinned commit: `d6752ecd88220e8fcd07f6d6825d2b8d642c9edc`
- Abhishek Enaguthi fork:
  <https://github.com/Abhishek21g/mib-doc-challenge-solution>
- Pinned clean commit: `08947bd72b0058d5803d9bfe08428cd7311485b7`
- License: MIT; notice retained at
  `third_party_licenses/PublicSolutions/LICENSE-Strobl` and
  `third_party_licenses/PublicSolutions/LICENSE-Abhishek`

The base render-first pipeline, RapidOCR recovery, resolution, adjudication,
and confidence layers derive from these repositories. The Abhishek commit was
selected before later answer-key-enabled history and independently reproduced
at 129.8470 after fixing its `python -I` import bug.

## Visible-layout scoring layer

- Arjun Shah: <https://github.com/arjunkshah12345-hash/mib-doc-solution>
- Pinned commit: `798ad2277a89a25cd3dfc596be572df4aade55c6`
- License: MIT; notice retained at
  `third_party_licenses/PublicSolutions/LICENSE-Arjun`

The visible field repairs, layout consensus, finding/damage heads, approval
safety demotions, denial softening, confidence table, and red SAMPLE DENIAL
concept are adapted from this commit. The answer-key transcription module and
broad visible-OCR retry module are not included or called. This repository
adds the stable postprocessing boundary, form-feed-preserving PDFium fallback,
blue-slash geometry detector, guarded PDFium red-watermark OCR, explicit
transit gate, OOF-supported Platt transform, anti-answer-key tests, and audited
orchestration.

## Native-image MaxFilter retry

- Kirtan Desai: <https://github.com/kirtandesai/mib-doc-solution>
- Pinned commit: `288da617686a115ef1b27fcf751dea9e61948226`
- License: MIT; notice retained at
  `third_party_licenses/PublicSolutions/LICENSE-Kirtan`

The visibly placed native-image selection and grayscale, 2x resize,
`MaxFilter(3)` OCR retry are adapted from this commit. This repository adds a
review-risk-only route, a raw-image improvement control, explicit answer-marker
rejection, and field-only output semantics. Native PDF text, signatures,
identity hints, and case IDs never route or validate the retry.

## Public-training-selected guardrails

The runtime contains identity-free tables selected against the public training
set. They are disclosed because they may overfit even though they contain no
case IDs, filenames, document hashes, applicant names, sponsor identities, or
per-case predictions:

- `mib_pipeline/score_heads.py` contains two visa/purpose trap cells, fourteen
  visa/purpose/page-order trap cells, two waived-only trap cells, and one
  waived override. These tables only block or constrain layout-consensus
  approvals.
- The four layout-consensus visa allowlists contain schema values, not packet
  identities.
- The public-train-calibrated terminal ensemble contains eleven
  `NEEDS_REVIEW` to `DENIED` boundaries and one `DENIED` to `NEEDS_REVIEW`
  boundary over schema fields and emitted confidence. It never creates
  `APPROVED`; rows it changes receive one disclosed confidence value.
- One TRIANGULAN confidence boundary can soften an uncertain denial to review.
- Four expensive visible-OCR retries are bounded by public-training-selected
  schema-field lanes: one fee-receipt lane, three unresolved-sponsor cells,
  one authority-note lane, and one stale-redacted-visa lane. These gates decide
  whether local OCR runs; the visible OCR result still decides whether a field
  or adjudication changes.
- The JSON files under `mib_pipeline/artifacts/` contain public-train-fitted
  confidence calibration. Their runtime features are semantic output or
  evidence-path features; `policy_exceptions.json` contains no exceptions.

The source repository may retain case-named public-train audit reports under
`artifacts/`. That directory is excluded from the Docker context and is not a
runtime lookup surface. The image contains no validation predictions or
per-case answer table.

## OCR models and dependencies

RapidOCR/PaddleOCR model provenance, Apache-2.0 notices, and a supplemental
FlatBuffers notice are retained under `third_party_licenses/`. Installed
Python wheels retain the license and notice files they actually distribute;
supplemental notices cover audited omissions. See
`third_party_licenses/MODEL_PROVENANCE.md` and
`third_party_licenses/README.md`.

The machine-readable cap, model, lineage, and hardcoding inventory is
`PACKAGING_AUDIT.json`.
