"""Benchmark AudBre's real separation job on local clips."""
from __future__ import annotations

import argparse
import atexit
import ast
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
DEFAULT_CLIPS = BENCHMARKS / "clips"
RESULTS_JSON = BENCHMARKS / "results.json"
RESULTS_MD = BENCHMARKS / "results.md"
PROMPT = "remove background noise"
TIMED_RUNS = 3
NEAR_SILENCE_RMS = 1e-4
NEIGHBOR_RMS_FOR_GAP = 1e-3
ABNORMAL_PEAK_RATIO = 1.5
CLIPPING_PEAK = 0.999


def _windows_system_info() -> dict[str, Any]:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return {"query_error": "PowerShell was not found"}

    command = (
        "$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1 "
        "-ExpandProperty Name); "
        "$ram = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
        "$procs = Get-Process | Sort-Object WorkingSet64 -Descending | "
        "Select-Object -First 10 @{N='name';E={$_.ProcessName}},"
        "@{N='working_set_bytes';E={$_.WorkingSet64}}; "
        "[ordered]@{cpu_model=$cpu;total_ram_bytes=$ram;processes=$procs} | "
        "ConvertTo-Json -Depth 4 -Compress"
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _configured_modal_gpu() -> str | None:
    source = ROOT / "modal_app.py"
    if not source.is_file():
        return None
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "GPU"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return value if isinstance(value, str) else None
    return None


def _package_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "soundfile",
        "httpx",
        "fastapi",
        "uvicorn",
        "torch",
        "torchaudio",
        "sam-audio",
        "modal",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _environment() -> dict[str, Any]:
    from audbre import config

    system: dict[str, Any] = {}
    query_error = None
    if os.name == "nt":
        try:
            system = _windows_system_info()
            query_error = system.pop("query_error", None)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            query_error = f"{type(exc).__name__}: {exc}"

    processes = system.get("processes", [])
    heavy_apps = [
        item
        for item in processes
        if item.get("working_set_bytes", 0) >= 1024**3
        and item.get("name", "").lower() not in {"memory compression", "system"}
    ]
    return {
        "os": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_model": system.get("cpu_model") or platform.processor() or None,
        "ram_bytes": system.get("total_ram_bytes"),
        "top_processes": processes,
        "heavy_app_threshold_bytes": 1024**3,
        "heavy_apps_at_least_threshold": heavy_apps,
        "process_query_error": query_error,
        "libraries": _package_versions(),
        "audbre_model": config.MODEL,
        "configured_engine": config.ENGINE,
        "modal_url_configured": bool(config.MODAL_URL),
        "modal_worker_token_configured": bool(config.WORKER_TOKEN),
        "modal_gpu_type_configured": _configured_modal_gpu(),
        "modal_gpu_type_observed": None,
    }


def _format_seconds(value: float | None) -> str:
    return "—" if value is None else str(value)


def _render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# AudBre benchmark results",
        "",
        f"Prompt: `{data['prompt']}`. Each measured value is from "
        f"{TIMED_RUNS} timed runs after one excluded warm-up.",
        "",
        "| Clip duration (s) | Backend | Median time (s) | Min (s) | Max (s) "
        "| Real-time factor | Chunks processed | Status |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in data["results"]:
        lines.append(
            "| {duration} | {backend} | {median} | {minimum} | {maximum} "
            "| {rtf} | {chunks} | {status} |".format(
                duration=row["clip_duration_s"],
                backend=row["backend"],
                median=_format_seconds(row["median_time_s"]),
                minimum=_format_seconds(row["min_time_s"]),
                maximum=_format_seconds(row["max_time_s"]),
                rtf=_format_seconds(row["real_time_factor"]),
                chunks=_format_seconds(row["chunks"]),
                status=row["status"],
            )
        )

    cold = data["cold_start"]
    lines.extend(["", "## Cold start", ""])
    if cold is None:
        lines.append("No Modal request was made; cold-start time was not measured.")
    else:
        lines.append(
            f"The first Modal request (warm-up, excluded from warm-run statistics) "
            f"took {cold['elapsed_s']} s and had status `{cold['status']}`."
        )

    lines.extend(["", "## Output and seam checks", ""])
    checked = 0
    for row in data["results"]:
        for run in row.get("timed_run_details", []):
            quality = run.get("quality")
            if quality is None:
                continue
            checked += 1
            issues = quality["seam_issue_count"]
            lines.append(
                f"- {row['clip_name']} / {row['backend']} run "
                f"{run['run_number']}: target and residual files exist; output "
                f"duration {quality['output_duration_s']} s "
                f"(difference {quality['duration_difference_s']} s, within "
                f"0.1 s: {quality['duration_within_0_1_s']}); "
                f"{quality['seams_checked']} overlap seams checked, "
                f"{issues} signal-level warning(s); output peak "
                f"{quality['output_peak']} "
                f"(flagged: {quality['output_peak_flagged']})."
            )
    if not checked:
        lines.append(
            "No successful timed output was produced, so file-duration and seam "
            "checks could not be run."
        )
    lines.extend(
        [
            "",
            "These are signal-level checks, not a listening test or a judgment "
            "of sound-removal quality. A seam warning means either a peak at "
            "least 1.5x both adjacent flank peaks, or seam RMS below 1e-4 while "
            "both flanks exceed 1e-3. An output peak at or above 0.999 is also "
            "flagged.",
            "",
            "## Environment",
            "",
            f"- OS: {data['environment']['os']}",
            f"- CPU: {data['environment']['cpu_model']}",
            f"- RAM: {data['environment']['ram_bytes']} bytes",
            f"- Python: {data['environment']['python'].splitlines()[0]}",
            f"- Libraries: `{json.dumps(data['environment']['libraries'], sort_keys=True)}`",
            f"- AudBre model / configured engine: "
            f"{data['environment']['audbre_model']} / "
            f"{data['environment']['configured_engine']}",
            f"- Modal URL configured: "
            f"{data['environment']['modal_url_configured']}; worker token "
            f"configured: {data['environment']['modal_worker_token_configured']}",
            f"- Modal GPU configured in `modal_app.py`: "
            f"{data['environment']['modal_gpu_type_configured']}",
            f"- Modal GPU observed during a run: "
            f"{data['environment']['modal_gpu_type_observed']}",
            f"- Other processes at least 1 GiB each: "
            f"{data['environment']['heavy_apps_at_least_threshold']}",
            f"- Top processes at capture: "
            f"`{json.dumps(data['environment']['top_processes'], sort_keys=True)}`",
            "",
            "## Run notes",
            "",
            "The timer encloses the application's `jobs._run` separation job, "
            "including project read/render, engine construction, chunk processing, "
            "and writing target/residual WAVs. Project upload/initial decode and "
            "post-run quality checks are outside the timer. LocalEngine is "
            "constructed for each job by AudBre, so its per-job model loading is "
            "included in every timed local run. Chunks are computed from the "
            "project's own `chunk_bounds` function.",
        ]
    )
    for note in data["notes"]:
        lines.append(f"- {note}")
    planned = {
        row["clip_duration_s"]: row["planned_chunks"]
        for row in data["results"]
    }
    planned_text = ", ".join(
        f"{duration} s: {chunks}"
        for duration, chunks in sorted(planned.items())
    )
    lines.append(
        f"- Expected chunk counts from AudBre `chunk_bounds` (not processed in "
        f"this run): {planned_text}."
    )
    errors = sorted(
        {
            (row["backend"], row["error"])
            for row in data["results"]
            if row["error"]
        }
    )
    for backend, error in errors:
        lines.append(f"- {backend} error: `{error}`")
    return "\n".join(lines) + "\n"


def _rms(samples: Any) -> float:
    import numpy as np

    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _quality_check(
    target_path: Path,
    residual_path: Path,
    expected_duration_s: float,
    bounds: list[tuple[int, int]],
    sample_rate: int,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    if not target_path.is_file() or not residual_path.is_file():
        return {
            "target_exists": target_path.is_file(),
            "residual_exists": residual_path.is_file(),
            "duration_within_0_1_s": False,
            "seams_checked": 0,
            "seam_issue_count": None,
            "error": "target.wav or residual.wav was not written",
        }

    info = sf.info(str(residual_path))
    output, output_sr = sf.read(
        str(residual_path), dtype="float32", always_2d=False
    )
    if output.ndim > 1:
        output = output.mean(axis=1)
    duration_s = len(output) / output_sr
    duration_difference_s = abs(duration_s - expected_duration_s)
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    seams: list[dict[str, Any]] = []
    flank_width = int(2 * sample_rate)
    for left, right in zip(bounds, bounds[1:]):
        seam_start, seam_end = right[0], left[1]
        if seam_end <= seam_start:
            continue
        seam = output[seam_start:seam_end]
        before = output[max(0, seam_start - flank_width):seam_start]
        after = output[seam_end:min(output.size, seam_end + flank_width)]
        before_peak = float(np.max(np.abs(before))) if before.size else 0.0
        after_peak = float(np.max(np.abs(after))) if after.size else 0.0
        seam_peak = float(np.max(np.abs(seam))) if seam.size else 0.0
        seam_rms = _rms(seam)
        flank_rms = max(_rms(before), _rms(after))
        flank_peak = max(before_peak, after_peak)
        abnormal_peak = flank_peak > 0.01 and seam_peak >= ABNORMAL_PEAK_RATIO * flank_peak
        silence_gap = (
            seam_rms < NEAR_SILENCE_RMS
            and _rms(before) > NEIGHBOR_RMS_FOR_GAP
            and _rms(after) > NEIGHBOR_RMS_FOR_GAP
        )
        seams.append(
            {
                "start_s": seam_start / sample_rate,
                "end_s": seam_end / sample_rate,
                "rms": seam_rms,
                "peak": seam_peak,
                "flank_rms_max": flank_rms,
                "flank_peak_max": flank_peak,
                "abnormal_peak": abnormal_peak,
                "near_silence_gap": silence_gap,
            }
        )
    issue_count = sum(
        int(seam["abnormal_peak"] or seam["near_silence_gap"]) for seam in seams
    )
    return {
        "target_exists": True,
        "residual_exists": True,
        "output_format": info.format,
        "output_subtype": info.subtype,
        "output_sample_rate": output_sr,
        "output_duration_s": duration_s,
        "duration_difference_s": duration_difference_s,
        "duration_within_0_1_s": duration_difference_s <= 0.1,
        "output_peak": peak,
        "samples_at_or_above_0_999": int(np.count_nonzero(np.abs(output) >= CLIPPING_PEAK)),
        "output_peak_flagged": peak >= CLIPPING_PEAK,
        "seams_checked": len(seams),
        "seam_issue_count": issue_count,
        "seams": seams,
    }


def _attempt(
    project_id: str,
    backend: str,
    description: str,
    expected_duration_s: float,
    bounds: list[tuple[int, int]],
    sample_rate: int,
    run_number: int,
    is_warmup: bool,
) -> dict[str, Any]:
    from audbre import config, jobs

    config.ENGINE = backend
    job = jobs.Job(
        id=uuid.uuid4().hex[:10],
        project_id=project_id,
        description=description,
        anchors=[],
    )
    started = time.perf_counter()
    jobs._run(job)
    elapsed_s = time.perf_counter() - started
    result: dict[str, Any] = {
        "run_number": run_number,
        "kind": "warmup" if is_warmup else "timed",
        "elapsed_s": elapsed_s,
        "status": job.status,
    }
    if job.status != "done":
        result["error"] = job.error
        return result
    if not is_warmup:
        result["quality"] = _quality_check(
            job.dir() / "target.wav",
            job.dir() / "residual.wav",
            expected_duration_s,
            bounds,
            sample_rate,
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("all", "modal", "local"),
        default="all",
        help="backend(s) to run; all runs Modal first so its first call is cold",
    )
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    clips = list(args.clips_dir.glob("*.wav"))
    if not clips:
        raise FileNotFoundError(f"No WAV clips found in {args.clips_dir}")

    runtime_storage = tempfile.TemporaryDirectory(
        prefix="audbre-runtime-", dir=BENCHMARKS
    )
    atexit.register(runtime_storage.cleanup)
    os.environ["AUDBRE_STORAGE"] = runtime_storage.name
    sys.path.insert(0, str(ROOT))

    import soundfile as sf

    clips.sort(key=lambda path: sf.info(str(path)).duration)

    from audbre import audio as audbre_audio
    from audbre import config
    from audbre.engine.base import chunk_bounds
    from audbre.store import Project

    environment = _environment()
    clip_data: list[dict[str, Any]] = []
    for path in clips:
        info = sf.info(str(path))
        clip_data.append(
            {
                "path": str(path),
                "name": path.name,
                "duration_s": info.frames / info.samplerate,
                "frames": info.frames,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "subtype": info.subtype,
            }
        )

    projects: dict[str, Project] = {}
    clip_processing: dict[str, dict[str, Any]] = {}
    for clip in clip_data:
        project = Project.create(
            Path(clip["path"]).name,
            Path(clip["path"]).read_bytes(),
        )
        projects[clip["name"]] = project
        audio, sample_rate = audbre_audio.read(project.base_wav)
        clip_processing[clip["name"]] = {
            "duration_s": audio.size / sample_rate,
            "sample_rate": sample_rate,
            "bounds": chunk_bounds(audio.size, sample_rate),
            "chunks": len(chunk_bounds(audio.size, sample_rate)),
        }

    selected = ("modal", "local") if args.backend == "all" else (args.backend,)
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    cold_start: dict[str, Any] | None = None
    modal_call_count = 0
    blocked: dict[str, str] = {}
    modal_preflight_error = None
    if "modal" in selected:
        if not config.MODAL_URL:
            modal_preflight_error = (
                "AUDBRE_MODAL_URL is not configured; no request sent"
            )
        elif not config.WORKER_TOKEN:
            modal_preflight_error = (
                "AUDBRE_WORKER_TOKEN is not configured; no audio was sent"
            )
        if modal_preflight_error:
            blocked["modal"] = modal_preflight_error

    for backend in ("modal", "local"):
        for clip in clip_data:
            processing = clip_processing[clip["name"]]
            row: dict[str, Any] = {
                "clip_name": clip["name"],
                "clip_duration_s": clip["duration_s"],
                "backend": backend,
                "chunks": None,
                "planned_chunks": processing["chunks"],
                "status": "not_requested",
                "warmup_time_s": None,
                "cold_start_s": None,
                "timed_run_s": [],
                "timed_run_details": [],
                "median_time_s": None,
                "min_time_s": None,
                "max_time_s": None,
                "real_time_factor": None,
                "error": None,
            }
            rows.append(row)
            if backend not in selected:
                continue
            if backend in blocked:
                row["status"] = "not_run"
                row["error"] = blocked[backend]
                continue

            try:
                warmup = _attempt(
                    projects[clip["name"]].id,
                    backend,
                    PROMPT,
                    processing["duration_s"],
                    processing["bounds"],
                    processing["sample_rate"],
                    0,
                    True,
                )
            except Exception as exc:
                warmup = {
                    "run_number": 0,
                    "kind": "warmup",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if backend == "modal" and modal_call_count == 0:
                modal_call_count += 1
                cold_start = {
                    "clip_name": clip["name"],
                    "elapsed_s": warmup.get("elapsed_s"),
                    "status": warmup["status"],
                    "error": warmup.get("error"),
                }
                row["cold_start_s"] = warmup.get("elapsed_s")
            row["warmup_time_s"] = warmup.get("elapsed_s")
            if warmup["status"] != "done":
                row["status"] = "unavailable"
                row["error"] = warmup.get("error", "warm-up did not complete")
                blocked[backend] = row["error"]
                continue
            row["chunks"] = processing["chunks"]

            timed_details: list[dict[str, Any]] = []
            for run_number in range(1, TIMED_RUNS + 1):
                try:
                    timed = _attempt(
                        projects[clip["name"]].id,
                        backend,
                        PROMPT,
                        processing["duration_s"],
                        processing["bounds"],
                        processing["sample_rate"],
                        run_number,
                        False,
                    )
                except Exception as exc:
                    timed = {
                        "run_number": run_number,
                        "kind": "timed",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                timed_details.append(timed)
                if timed["status"] != "done":
                    row["error"] = timed.get("error", "timed run did not complete")
                    break

            row["timed_run_details"] = timed_details
            row["timed_run_s"] = [
                run["elapsed_s"]
                for run in timed_details
                if run["status"] == "done"
            ]
            if len(row["timed_run_s"]) == TIMED_RUNS:
                row["status"] = "complete"
                row["median_time_s"] = statistics.median(row["timed_run_s"])
                row["min_time_s"] = min(row["timed_run_s"])
                row["max_time_s"] = max(row["timed_run_s"])
                row["real_time_factor"] = (
                    row["median_time_s"] / processing["duration_s"]
                )
            else:
                row["status"] = "incomplete"
                if row["error"]:
                    blocked[backend] = row["error"]

    backend_order = {"modal": 0, "local": 1}
    rows.sort(
        key=lambda row: (
            row["clip_duration_s"],
            backend_order[row["backend"]],
        )
    )

    if "modal" not in selected:
        notes.append("Modal was explicitly skipped for this run; no audio was sent.")
    elif modal_preflight_error:
        notes.append(f"Modal was not attempted: {modal_preflight_error}.")
    if "local" in selected and any(row["backend"] == "local" and row["status"] == "unavailable" for row in rows):
        notes.append("The local backend could not complete its warm-up; subsequent local clips were not attempted.")
    notes.append(
        "The JSON also records planned_chunks from AudBre's chunk_bounds function; "
        "the table reports chunks processed only after a complete warm-up."
    )
    notes.append(
        "Cold-start time is the first Modal warm-up request; it is excluded from "
        "the three-run warm statistics."
    )

    data = {
        "prompt": PROMPT,
        "timed_runs_per_clip_backend": TIMED_RUNS,
        "warmup_runs_per_clip_backend": 1,
        "cold_start": cold_start,
        "environment": environment,
        "clips": clip_data,
        "selected_backends": list(selected),
        "results": rows,
        "notes": notes,
    }
    RESULTS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(_render_markdown(data), encoding="utf-8")
    print(f"Wrote {RESULTS_JSON}")
    print(f"Wrote {RESULTS_MD}")
    print(f"Prompt: {PROMPT}")
    print(f"Backends selected: {', '.join(selected)}")
    successful = sum(row["status"] == "complete" for row in rows)
    print(f"Complete clip/backend measurements: {successful}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
