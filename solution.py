#!/usr/bin/env python3
"""Offline two-argument runtime for the MIB case processor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from mib_pipeline import (
    AdjudicationEngine,
    BatchRunReport,
    BatchRunner,
    CalibrationArtifactError,
    CaseLinker,
    ConfidenceCalibrator,
    DocumentRenderer,
    EvidencePrecedenceResolver,
    GeneralizablePolicyExceptionStore,
    OutputConfidenceRecalibrator,
    PolicyArtifactError,
    RapidOutputRecoveryProcessor,
    ReviewDenialRecoveryAdjudicator,
    SafeFallbackProcessor,
    VisibleEvidenceExtractor,
    CanonicalJsonlWriter,
    PredictionRow,
    discover_case_pdfs,
)
from mib_pipeline.batch import CaseFailure
from mib_pipeline.score_finalizer import VisibleScoreFinalizer
from mib_pipeline.visible_maxfilter import (
    VisibleFieldRecoveryChain,
    VisibleMaxFilterRiskRecoverer,
)
from mib_pipeline.visible_risk_crop import VisibleRiskCropRecoverer


USAGE = "usage: solution.py <input_pdf_dir> <output_predictions_path>"
MAX_WORKERS = 4
COMPLETENESS_FAILURE_EXIT = 65
MAX_OMISSION_RATE = 0.001
MAX_OMISSIONS = 5
PROCESS_RECYCLE_BATCH_SIZE = 1000
OMISSION_RECOVERY_PASSES = 2
CHUNK_WORKER_FLAG = "--internal-chunk-worker"
CRASH_RECOVERY_ENV = "MIB_CRASH_RECOVERY"
CHECKPOINT_DIR_ENV = "MIB_CHECKPOINT_DIR"


class ContractError(ValueError):
    """Raised when the two-argument runtime contract is not satisfied."""


def configured_worker_limit() -> int:
    """Return a valid worker limit that never exceeds the four-vCPU budget."""

    raw_value = os.environ.get("MIB_MAX_WORKERS", str(MAX_WORKERS))
    try:
        requested = int(raw_value)
    except ValueError as exc:
        raise ContractError("MIB_MAX_WORKERS must be an integer") from exc
    if requested < 1:
        raise ContractError("MIB_MAX_WORKERS must be at least 1")
    return min(requested, MAX_WORKERS)


def configured_recycle_batch_size() -> int:
    """Return a positive process lifetime capped at the sealed safe bound."""

    raw_value = os.environ.get(
        "MIB_PROCESS_RECYCLE_BATCH_SIZE",
        str(PROCESS_RECYCLE_BATCH_SIZE),
    )
    try:
        requested = int(raw_value)
    except ValueError as exc:
        raise ContractError(
            "MIB_PROCESS_RECYCLE_BATCH_SIZE must be an integer"
        ) from exc
    if requested < 1:
        raise ContractError("MIB_PROCESS_RECYCLE_BATCH_SIZE must be at least 1")
    return min(requested, PROCESS_RECYCLE_BATCH_SIZE)


def crash_recovery_enabled() -> bool:
    """Enable native-worker retry only in the sealed container runtime."""

    return os.environ.get(CRASH_RECOVERY_ENV, "0").strip().casefold() in {
        "1",
        "true",
        "yes",
    }


def checkpoint_root(output_path: Path) -> Path:
    """Return a durable checkpoint directory beside the bound output file."""

    configured = os.environ.get(CHECKPOINT_DIR_ENV, "").strip()
    root = Path(configured) if configured else output_path.parent / f".{output_path.name}.checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    return root


def parse_paths(argv: Sequence[str]) -> tuple[Path, Path]:
    """Validate and return the input directory and exact output file path."""

    if len(argv) != 3:
        raise ContractError(USAGE)

    input_dir = Path(argv[1])
    output_path = Path(argv[2])

    if not input_dir.is_dir():
        raise ContractError(f"input PDF directory does not exist: {input_dir}")
    if not output_path.name:
        raise ContractError("output predictions path must name a file")
    if output_path.exists() and output_path.is_dir():
        raise ContractError(f"output predictions path is a directory: {output_path}")
    if not output_path.parent.is_dir():
        raise ContractError(
            f"output directory does not exist: {output_path.parent}"
        )

    resolved_input = input_dir.resolve()
    resolved_output = output_path.resolve()
    try:
        resolved_output.relative_to(resolved_input)
    except ValueError:
        output_is_inside_input = False
    else:
        output_is_inside_input = True
    if output_is_inside_input:
        raise ContractError("output predictions path must not be inside the input directory")

    return input_dir, output_path


def allowed_omissions(attempted: int) -> int:
    """Return the small omission budget allowed before the run fails loudly."""

    if attempted < 1:
        return 0
    proportional_budget = max(1, int(attempted * MAX_OMISSION_RATE))
    return min(MAX_OMISSIONS, proportional_budget)


def _one_line(value: object) -> str:
    """Keep one case failure on one stderr line."""

    return " ".join(str(value).split())


def emit_run_report(report: BatchRunReport) -> int:
    """Print the complete run ledger and return the completeness exit code."""

    for failure in report.failures:
        print(
            "omission "
            f"source={_one_line(failure.source_name)} "
            f"reason={_one_line(failure.reason)}",
            file=sys.stderr,
        )
    print(
        f"attempted={report.attempted} answered={report.answered} omitted={report.omitted}",
        file=sys.stderr,
    )

    if report.answered == 0:
        print(
            "error: completeness gate failed: zero cases were answered",
            file=sys.stderr,
        )
        return COMPLETENESS_FAILURE_EXIT

    omission_budget = allowed_omissions(report.attempted)
    if report.omitted > omission_budget:
        print(
            "error: completeness gate failed: "
            f"omitted={report.omitted} exceeds allowed={omission_budget} "
            f"for attempted={report.attempted}",
            file=sys.stderr,
        )
        return COMPLETENESS_FAILURE_EXIT
    return 0


def build_runner() -> BatchRunner:
    """Construct the frozen processing pipeline."""

    return BatchRunner(
        RapidOutputRecoveryProcessor(
            renderer=DocumentRenderer(),
            primary_extractor=VisibleEvidenceExtractor(
                packet_page_type_markers=True,
            ),
            linker=CaseLinker(),
            resolver=EvidencePrecedenceResolver(),
            adjudicator=ReviewDenialRecoveryAdjudicator(
                AdjudicationEngine(
                    calibrator=ConfidenceCalibrator.from_pinned_artifact(),
                    exceptions=GeneralizablePolicyExceptionStore.from_pinned_artifact(),
                )
            ),
            output_recalibrator=OutputConfidenceRecalibrator.from_pinned_artifact(),
            row_finalizer=VisibleScoreFinalizer(),
            post_field_recoverer=VisibleFieldRecoveryChain(
                VisibleRiskCropRecoverer(),
                VisibleMaxFilterRiskRecoverer(),
            ),
        ),
        max_workers=configured_worker_limit(),
    )


def partition_paths(
    pdf_paths: Sequence[Path],
    batch_size: int = PROCESS_RECYCLE_BATCH_SIZE,
) -> tuple[tuple[Path, ...], ...]:
    """Split deterministic input order into bounded process-lifetime batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return tuple(
        tuple(pdf_paths[index : index + batch_size])
        for index in range(0, len(pdf_paths), batch_size)
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write one internal control artifact without exposing partial state."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _worker_main(argv: Sequence[str]) -> int:
    """Process one bounded manifest in a fresh interpreter."""

    if len(argv) != 5 or argv[1] != CHUNK_WORKER_FLAG:
        raise ContractError("invalid internal chunk worker invocation")
    manifest_path = Path(argv[2])
    output_path = Path(argv[3])
    report_path = Path(argv[4])
    raw_paths = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_paths, list) or not all(
        isinstance(value, str) for value in raw_paths
    ):
        raise ContractError("internal chunk manifest must be a JSON string array")
    pdf_paths = tuple(Path(value) for value in raw_paths)
    if not pdf_paths or any(
        not path.is_file() or path.suffix.casefold() != ".pdf"
        for path in pdf_paths
    ):
        raise ContractError("internal chunk manifest contains an invalid PDF path")

    report = build_runner().run_paths(pdf_paths, output_path)
    _write_json_atomic(
        report_path,
        {
            "attempted": report.attempted,
            "answered": report.answered,
            "omitted": report.omitted,
            "failures": [
                {
                    "source_name": failure.source_name,
                    "reason": failure.reason,
                }
                for failure in report.failures
            ],
        },
    )
    return 0


def _read_rows(path: Path) -> tuple[PredictionRow, ...]:
    """Read canonical child rows for one final atomic merge."""

    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(PredictionRow.from_mapping(json.loads(line)))
    return tuple(rows)


def _read_worker_result(
    *,
    worker_label: str,
    expected_paths: Sequence[Path],
    output_path: Path,
    report_path: Path,
) -> tuple[tuple[PredictionRow, ...], tuple[CaseFailure, ...]]:
    """Validate and return one successful worker's complete artifacts."""

    if not output_path.is_file() or not report_path.is_file():
        raise OSError(f"{worker_label} did not publish complete artifacts")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    attempted = int(payload["attempted"])
    answered = int(payload["answered"])
    omitted = int(payload["omitted"])
    raw_failures = payload["failures"]
    rows = _read_rows(output_path)
    if (
        not isinstance(raw_failures, list)
        or attempted != len(expected_paths)
        or answered != len(rows)
        or omitted != attempted - answered
        or len(raw_failures) != omitted
    ):
        raise OSError(f"{worker_label} report is inconsistent")

    expected_names = {path.name for path in expected_paths}
    if len(expected_names) != len(expected_paths):
        raise OSError(f"{worker_label} manifest contains duplicate source names")
    row_ids = {row.case_id for row in rows}
    if len(row_ids) != len(rows) or not row_ids.issubset(
        {path.stem for path in expected_paths}
    ):
        raise OSError(f"{worker_label} output contains unexpected case IDs")
    failures = tuple(
        CaseFailure(
            source_name=str(failure["source_name"]),
            reason=str(failure["reason"]),
        )
        for failure in raw_failures
    )
    failure_names = {failure.source_name for failure in failures}
    if (
        len(failure_names) != len(failures)
        or not failure_names.issubset(expected_names)
        or {f"{case_id}.pdf" for case_id in row_ids} | failure_names
        != expected_names
    ):
        raise OSError(f"{worker_label} artifacts do not cover its manifest")
    return rows, failures


def _checkpoint_path(chunk_root: Path, index: int) -> Path:
    """Name the durable completion marker for one deterministic chunk."""

    return chunk_root / f"completed-{index:04d}.json"


def _write_checkpoint(
    path: Path,
    *,
    expected_paths: Sequence[Path],
    rows: Sequence[PredictionRow],
    failures: Sequence[CaseFailure],
) -> None:
    """Persist one fully recovered chunk before publishing the merged output."""

    _write_json_atomic(
        path,
        {
            "source_names": [item.name for item in expected_paths],
            "rows": [row.to_dict() for row in rows],
            "failures": [
                {"source_name": failure.source_name, "reason": failure.reason}
                for failure in failures
            ],
        },
    )


def _read_checkpoint(
    path: Path,
    *,
    expected_paths: Sequence[Path],
) -> tuple[tuple[PredictionRow, ...], tuple[CaseFailure, ...]]:
    """Read and validate a durable chunk marker, rejecting stale partial state."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_names = [item.name for item in expected_paths]
    if payload.get("source_names") != expected_names:
        raise OSError("checkpoint source manifest does not match current input")
    rows = tuple(
        PredictionRow.from_mapping(value)
        for value in payload.get("rows", [])
    )
    failures = tuple(
        CaseFailure(str(value["source_name"]), str(value["reason"]))
        for value in payload.get("failures", [])
    )
    row_ids = {row.case_id for row in rows}
    failure_names = {failure.source_name for failure in failures}
    if len(row_ids) != len(rows) or len(failure_names) != len(failures):
        raise OSError("checkpoint contains duplicate results")
    if {f"{case_id}.pdf" for case_id in row_ids} | failure_names != set(expected_names):
        raise OSError("checkpoint does not cover its source manifest")
    return rows, failures


def _worker_command(
    solution_path: Path,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(solution_path),
        CHUNK_WORKER_FLAG,
        str(manifest_path),
        str(output_path),
        str(report_path),
    ]


def _publish_completed_rows(
    output_path: Path,
    rows: Sequence[PredictionRow],
) -> None:
    """Atomically expose every validated row from completed chunks."""

    CanonicalJsonlWriter().write(output_path, rows)


def run_chunked(
    pdf_paths: Sequence[Path],
    output_path: Path,
    *,
    batch_size: int = PROCESS_RECYCLE_BATCH_SIZE,
) -> BatchRunReport:
    """Recycle OCR state and atomically publish each completed chunk."""

    all_rows: list[PredictionRow] = []
    all_failures: list[CaseFailure] = []
    attempted = 0
    solution_path = Path(__file__).resolve()

    chunk_root = checkpoint_root(output_path)
    for index, chunk in enumerate(partition_paths(pdf_paths, batch_size), start=1):
            completed_checkpoint = _checkpoint_path(chunk_root, index)
            try:
                checkpoint_rows, checkpoint_failures = _read_checkpoint(
                    completed_checkpoint,
                    expected_paths=chunk,
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                checkpoint_rows = ()
                checkpoint_failures = ()
            else:
                attempted += len(chunk)
                all_rows.extend(checkpoint_rows)
                all_failures.extend(checkpoint_failures)
                _publish_completed_rows(output_path, all_rows)
                continue

            manifest_path = chunk_root / f"manifest-{index:04d}.json"
            chunk_output = chunk_root / f"predictions-{index:04d}.jsonl"
            report_path = chunk_root / f"report-{index:04d}.json"
            _write_json_atomic(
                manifest_path,
                [str(path.resolve()) for path in chunk],
            )
            completed = subprocess.run(
                _worker_command(
                    solution_path,
                    manifest_path,
                    chunk_output,
                    report_path,
                ),
                check=False,
            )
            if completed.returncode != 0 and crash_recovery_enabled():
                print(
                    f"chunk worker {index} failed with exit {completed.returncode}; "
                    "retrying with one worker",
                    file=sys.stderr,
                )
                previous_worker_limit = os.environ.get("MIB_MAX_WORKERS")
                os.environ["MIB_MAX_WORKERS"] = "1"
                try:
                    completed = subprocess.run(
                        _worker_command(
                            solution_path,
                            manifest_path,
                            chunk_output,
                            report_path,
                        ),
                        check=False,
                    )
                finally:
                    if previous_worker_limit is None:
                        os.environ.pop("MIB_MAX_WORKERS", None)
                    else:
                        os.environ["MIB_MAX_WORKERS"] = previous_worker_limit
            if completed.returncode != 0:
                raise OSError(
                    f"chunk worker {index} failed with exit {completed.returncode}"
                )
            chunk_rows, chunk_failures = _read_worker_result(
                worker_label=f"chunk worker {index}",
                expected_paths=chunk,
                output_path=chunk_output,
                report_path=report_path,
            )
            persistent_failures = chunk_failures

            for recovery_pass in range(1, OMISSION_RECOVERY_PASSES + 1):
                if not persistent_failures:
                    break
                paths_by_name = {path.name: path for path in chunk}
                retry_paths = tuple(
                    paths_by_name[failure.source_name]
                    for failure in persistent_failures
                )
                retry_manifest = (
                    chunk_root
                    / f"retry-{recovery_pass:02d}-manifest-{index:04d}.json"
                )
                retry_output = (
                    chunk_root
                    / f"retry-{recovery_pass:02d}-predictions-{index:04d}.jsonl"
                )
                retry_report = (
                    chunk_root
                    / f"retry-{recovery_pass:02d}-report-{index:04d}.json"
                )
                _write_json_atomic(
                    retry_manifest,
                    [str(path.resolve()) for path in retry_paths],
                )
                retry_completed = subprocess.run(
                    _worker_command(
                        solution_path,
                        retry_manifest,
                        retry_output,
                        retry_report,
                    ),
                    check=False,
                )
                if retry_completed.returncode == 0:
                    recovered_rows, persistent_failures = _read_worker_result(
                        worker_label=(
                            f"retry worker {index} pass {recovery_pass}"
                        ),
                        expected_paths=retry_paths,
                        output_path=retry_output,
                        report_path=retry_report,
                    )
                    chunk_rows += recovered_rows
                else:
                    break

            if persistent_failures:
                paths_by_name = {path.name: path for path in chunk}
                fallback_processor = SafeFallbackProcessor()
                fallback_rows: list[PredictionRow] = []
                for failure in persistent_failures:
                    pdf_path = paths_by_name[failure.source_name]
                    print(
                        "technical-fallback "
                        f"source={_one_line(failure.source_name)} "
                        f"reason={_one_line(failure.reason)}",
                        file=sys.stderr,
                    )
                    fallback_rows.append(
                        PredictionRow.from_mapping(
                            fallback_processor.process_case(pdf_path),
                            fallback_case_id=pdf_path.stem,
                        )
                    )
                chunk_rows += tuple(fallback_rows)
                persistent_failures = ()

            attempted += len(chunk)
            all_rows.extend(chunk_rows)
            all_failures.extend(persistent_failures)
            _write_checkpoint(
                completed_checkpoint,
                expected_paths=chunk,
                rows=chunk_rows,
                failures=persistent_failures,
            )
            _publish_completed_rows(output_path, all_rows)

    return BatchRunReport(
        attempted=attempted,
        answered=len(all_rows),
        omitted=len(all_failures),
        failures=tuple(all_failures),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline processor and return a process exit code."""

    arguments = sys.argv if argv is None else argv
    try:
        if len(arguments) > 1 and arguments[1] == CHUNK_WORKER_FLAG:
            return _worker_main(arguments)
        input_dir, output_path = parse_paths(arguments)
        pdf_paths = discover_case_pdfs(input_dir)
        recycle_batch_size = configured_recycle_batch_size()
        if pdf_paths:
            report = run_chunked(
                pdf_paths,
                output_path,
                batch_size=recycle_batch_size,
            )
        else:
            report = build_runner().run(input_dir, output_path)
    except (
        CalibrationArtifactError,
        ContractError,
        json.JSONDecodeError,
        OSError,
        PolicyArtifactError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 64
    return emit_run_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
