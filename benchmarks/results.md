# AudBre benchmark results

Prompt: `remove background noise`. Each measured value is from 3 timed runs after one excluded warm-up.

| Clip duration (s) | Backend | Median time (s) | Min (s) | Max (s) | Real-time factor | Chunks processed | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 10.0 | modal | — | — | — | — | — | not_requested |
| 10.0 | local | — | — | — | — | — | unavailable |
| 30.0 | modal | — | — | — | — | — | not_requested |
| 30.0 | local | — | — | — | — | — | not_run |
| 60.0 | modal | — | — | — | — | — | not_requested |
| 60.0 | local | — | — | — | — | — | not_run |
| 120.0 | modal | — | — | — | — | — | not_requested |
| 120.0 | local | — | — | — | — | — | not_run |
| 300.0 | modal | — | — | — | — | — | not_requested |
| 300.0 | local | — | — | — | — | — | not_run |

## Cold start

No Modal request was made; cold-start time was not measured.

## Output and seam checks

No successful timed output was produced, so file-duration and seam checks could not be run.

These are signal-level checks, not a listening test or a judgment of sound-removal quality. A seam warning means either a peak at least 1.5x both adjacent flank peaks, or seam RMS below 1e-4 while both flanks exceed 1e-3. An output peak at or above 0.999 is also flagged.

## Environment

- OS: Windows-11-10.0.26200-SP0
- CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
- RAM: 8358518784 bytes
- Python: 3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)]
- Libraries: `{"fastapi": "0.141.1", "httpx": "0.28.1", "modal": "1.5.5", "numpy": "2.5.2", "sam-audio": null, "soundfile": "0.14.0", "torch": null, "torchaudio": null, "uvicorn": "0.52.4"}`
- AudBre model / configured engine: facebook/sam-audio-small / modal
- Modal URL configured: True; worker token configured: False
- Modal GPU configured in `modal_app.py`: A10G
- Modal GPU observed during a run: None
- Other processes at least 1 GiB each: []
- Top processes at capture: `[{"name": "Memory Compression", "working_set_bytes": 836259840}, {"name": "Code", "working_set_bytes": 466956288}, {"name": "Code", "working_set_bytes": 444395520}, {"name": "copilot-runtime", "working_set_bytes": 296013824}, {"name": "Code", "working_set_bytes": 238153728}, {"name": "MsMpEng", "working_set_bytes": 196984832}, {"name": "Code", "working_set_bytes": 133193728}, {"name": "Code", "working_set_bytes": 118022144}, {"name": "WhatsApp.Root", "working_set_bytes": 113704960}, {"name": "Wispr Flow", "working_set_bytes": 111681536}]`

## Run notes

The timer encloses the application's `jobs._run` separation job, including project read/render, engine construction, chunk processing, and writing target/residual WAVs. Project upload/initial decode and post-run quality checks are outside the timer. LocalEngine is constructed for each job by AudBre, so its per-job model loading is included in every timed local run. Chunks are computed from the project's own `chunk_bounds` function.
- Modal was explicitly skipped for this run; no audio was sent.
- The local backend could not complete its warm-up; subsequent local clips were not attempted.
- The JSON also records planned_chunks from AudBre's chunk_bounds function; the table reports chunks processed only after a complete warm-up.
- Cold-start time is the first Modal warm-up request; it is excluded from the three-run warm statistics.
- Expected chunk counts from AudBre `chunk_bounds` (not processed in this run): 10.0 s: 1, 30.0 s: 1, 60.0 s: 3, 120.0 s: 5, 300.0 s: 11.
- local error: `ModuleNotFoundError: No module named 'torch'`
