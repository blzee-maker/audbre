# Benchmark clips

All clips are stored in `benchmarks/clips/`. Metadata below was read from the
generated/selected WAV files with SoundFile; each duration is frames divided
by sample rate.

| File | Duration (s) | Frames | Sample rate (Hz) | Channels | Format / subtype |
|---|---:|---:|---:|---:|---|
| `10s_test_input.wav` | 10.0 | 480000 | 48000 | 1 | WAV / PCM_16 |
| `30s_synthetic.wav` | 30.0 | 1440000 | 48000 | 1 | WAV / PCM_16 |
| `60s_synthetic.wav` | 60.0 | 2880000 | 48000 | 1 | WAV / PCM_16 |
| `120s_synthetic.wav` | 120.0 | 5760000 | 48000 | 1 | WAV / PCM_16 |
| `300s_synthetic.wav` | 300.0 | 14400000 | 48000 | 1 | WAV / PCM_16 |

The 10-second input is a copy of the repository's existing `test_input.wav`.
The 30-, 60-, 120-, and 300-second supplementary clips are generated offline
from Windows text-to-speech and locally generated pink noise, then mixed and
trimmed with FFmpeg. They contain synthetic speech and noise, not recordings
of real speech. No audio or credentials are sent outside the project's
existing Modal setup.

The benchmark uses the fixed prompt `remove background noise` and no timeline
anchors. This makes all chunks process instead of passing unanchored chunks
through untouched.
