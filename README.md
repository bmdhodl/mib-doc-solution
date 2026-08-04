# MIB Doc Challenge - offline visible-evidence submission

Offline CPU submission for the
[8090 MIB Document Challenge](https://github.com/8090-inc/mib-doc-challenge).

This solution scores **137.2283796025 / 150** on all 1,000 public training
packets under the organizer's own evaluator:

| Section | Score |
| --- | ---: |
| Field extraction | 45.1177777778 / 50 |
| Adjudication | 74.2900000000 / 80 |
| Confidence calibration | 17.8206018248 / 20 |
| Total | **137.2283796025 / 150** |

It emits 1,000 schema-valid rows with no omissions and no catastrophic false
approvals. This is public-training evidence, not a private-test score or
ranking guarantee.

Every push runs [`.github/workflows/public-contract.yml`](.github/workflows/public-contract.yml),
which builds this repository from a clean clone, installs the pinned runtime
with `--require-hashes`, runs the full test suite, builds the submission image,
and asserts the fail-closed exit-64 contract. Reproducibility is checked, not
claimed.

## Approach

The stable core renders every page, recovers visible text with RapidOCR and
targeted Tesseract fallbacks, links evidence across pages, resolves conflicting
fields by document precedence, and applies conservative policy rules.

A separate final scoring layer then:

- repairs fee, name, visa, sponsor, arrival, and purpose fields;
- honors visible findings and damage cues;
- detects narrowly gated visible blue and red denial marks;
- promotes only tightly gated, cross-form clean packets;
- demotes unsafe approvals and hard-gates `TRANSIT-7`;
- blends confidence with frozen identity-free reliability features;
- applies a disclosed public-train-selected terminal ensemble that never
  creates approvals and whose transfer ablation did not show reliable positive
  mean lift.

Raster OCR and visible geometry are the primary evidence path. Sanitized
native selectable text may nominate pages, regions, or field candidates. It is
not authoritative for approval or denial: decision-changing selectable cues
require independent raster or pixel corroboration. Some field recovery still
uses sanitized selectable layout, so this is not a claim that arbitrary hidden
text can never influence extraction.

There is no answer-key parser, opt-in switch, case-ID lookup, filename feature,
network call, LLM, or VLM. The image runs with a read-only root filesystem and
retains UID 0 solely so it can write the official runner's host-created output
bind mount. It uses no network, GPU, API key, external service, or runtime model
download.

## Run

```bash
docker build -t mib-visible-submission .
mkdir -p /tmp/mib-output
docker run --rm --network none \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --mount type=bind,src="/absolute/path/to/pdfs",dst=/input,readonly \
  --mount type=bind,src="/tmp/mib-output",dst=/output \
  mib-visible-submission /input /output/predictions.jsonl
```

The image accepts exactly two arguments: the input PDF directory and exact
output JSONL path.

Use the organizer's exact runner and validator:

```bash
python3 scripts/run_docker_submission.py \
  --repo . \
  --input-dir /path/to/mib-doc-challenge/data/validation \
  --output /tmp/mib-output/predictions.jsonl \
  --manifest /path/to/mib-doc-challenge/data/validation_manifest.csv \
  --timeout-seconds 30000 \
  --require-complete

python3 scripts/validate_submission.py \
  --submission /tmp/mib-output/predictions.jsonl \
  --manifest /path/to/mib-doc-challenge/data/validation_manifest.csv \
  --require-complete
```

See [MEMO.md](MEMO.md) for the score audit, transfer warning, and failure
modes, and [ATTRIBUTION.md](ATTRIBUTION.md) for exact upstream commits.
