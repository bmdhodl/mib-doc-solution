# Submission

Public solution repository:

<https://github.com/bmdhodl/mib-doc-solution>

The repository contains the complete offline source, pinned dependency lock,
Dockerfile, technical memo, attribution, tests, and third-party notices.

Solution commit that generated the submitted validation predictions:
`<FINAL5K_SOLUTION_COMMIT>`.

Image digest used for the validation run:
`<FINAL5K_IMAGE_DIGEST>`.

The checkpoint scores **137.22837960254876 / 150** on all 1,000 public
training packets. It produced 1,000 valid records, zero missing or duplicate
cases, and zero catastrophic false approvals. Prediction SHA-256:
`9a675507c686fca0544802d10f260e70ab77f0ee3b2219a2e56d93a2e7a0620a`.
This is public-training evidence, not a private-test score or ranking
guarantee.

This is a derivative MIT-licensed solution. Full upstream commit attribution,
public-training-selected guardrail disclosure, model provenance, packaging
limits, and the no-hardcoded-answer audit are recorded in `ATTRIBUTION.md` and
`PACKAGING_AUDIT.json`.

Build:

```bash
docker build -t mib-visible-submission .
```

Runtime contract:

```bash
docker run --rm --network none \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --mount type=bind,src="/absolute/path/to/pdfs",dst=/input,readonly \
  --mount type=bind,src="/tmp/mib-output",dst=/output \
  mib-visible-submission /input /output/predictions.jsonl
```

The image accepts exactly the input directory and output path. It runs with a
read-only root filesystem and retains UID 0 solely so it can write the official
runner's host-created output bind mount. It uses no network, GPU, API key,
external service, or runtime model download.

## Final validation receipt

The exact frozen image was run against all 5,000 unlabeled validation packets.

### How the predictions were generated

`solution.py` always partitions its input into bounded 1,000-case chunks and
runs each chunk in a fresh interpreter with no cross-chunk state, so every case
is scored independently of the others. To generate these predictions we ran the
same image five times, once per disjoint 1,000-case subset of the sorted
validation set, and concatenated the results. The subsets match the chunk
boundaries `solution.py` would have chosen on a single 5,000-case invocation
(`MIB-100001..101000`, `101001..102000`, `102001..103000`, `103001..104000`,
`104001..105000`), so the output is identical to a single run. Rows are written
by the canonical writer, sorted by `case_id`, with duplicates rejected.

Scoring the image as 8090 does, with one invocation over all 5,000 PDFs at
`--cpus 4`, is unaffected: measured throughput is 3.2 seconds per PDF against
the 6-second budget.

- Rows: `<FINAL5K_ROWS>`
- Runtime seconds: `<FINAL5K_RUNTIME_SECONDS>`
- Prediction bytes: `<FINAL5K_PREDICTION_BYTES>`
- Prediction SHA-256: `<FINAL5K_SHA256>`
- Validator result: `<FINAL5K_VALIDATOR_RESULT>`
