# Archived baselines

Superseded baselines, kept for provenance. Nothing here is comparable to a
current run and `compare.py` will say so.

## `M3-macos-preisolation.json`

The original M3 baseline (`fd6544e`), recorded on macOS arm64 before the bench
harness was reworked. Not comparable to anything produced since, for two
independent reasons:

1. **Different metric definitions.** `rss_peak_mb` was `resident_size_max` read
   once at the end — a *process-lifetime* high-water mark, while every other
   metric in the same block was a delta scoped to the measurement loop. The
   replacement splits that into `rss_peak_lifetime_mb` and
   `rss_peak_window_mb`, and the key `rss_peak_mb` no longer exists.

2. **Different machine, no fingerprint.** The file records `"os": "macos"` and
   nothing else, so there is no way to check what it should be compared
   against. Runs now carry an `environment` block for exactly this reason.

Its `page_faults_major: 0` is also misleading in hindsight: the model's disk
reads all happened during load and warmup, outside the delta window, so the
zero means "no I/O in the window", not "no I/O".
