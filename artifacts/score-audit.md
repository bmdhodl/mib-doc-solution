# Public-train score audit

- Organizer commit: `38ce8883dea9f87c27a8a95f134e54fe8b673064`
- Runner: organizer `scripts/run_docker_submission.py`
- Evaluator: organizer `scripts/evaluate.py`
- Input: all 1,000 public training PDFs
- Runtime: `--network none --cpus 4 --memory 8g --read-only`
- Output: 1,000 submitted and scored; 0 missing, extra, duplicate, or invalid
- Total: `134.7279107880363 / 150`
- Extraction: `45.04111111111111 / 50`
- Adjudication: `72.25 / 80`
- Calibration: `17.436799676925183 / 20`
- Mean Brier: `0.06408000807687039`
- Catastrophic false approvals: `0`
- Output SHA-256:
  `23827c4454e991778b514acad23a2f7105533087b91abe3c42d679ab31029a17`

The Docker output was also parsed as JSON and compared object-for-object with
the independently measured visible-only prototype: exact semantic match.
