# Flux2Kit Release Benchmark

**Generated:** 2026-07-27 02:12:54 UTC
**Host:** arm64, macOS 26.5.1
**Workload:** 512×512, 4 steps, guidance 1.0, seed 42
**Build:** Swift release

| Path | Pipeline load | Text encode | Denoise | VAE decode | To image | Generation total | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eager-first` | 511.3 ms | 200.1 ms | 2422.9 ms | 301.6 ms | 2.4 ms | 2934.5 ms | 4110.0 ms |
| `eager-warm` | 501.4 ms | 200.9 ms | 2430.8 ms | 304.8 ms | 1.7 ms | 2945.8 ms | 3610.0 ms |
| `eager-eval4` | 498.5 ms | 200.5 ms | 2419.7 ms | 304.0 ms | 1.9 ms | 2933.6 ms | 3590.0 ms |
| `compile` | 499.5 ms | 195.6 ms | 2926.3 ms | 303.9 ms | 1.8 ms | 3435.8 ms | 4100.0 ms |
| `heun` | 494.4 ms | 196.1 ms | 4444.8 ms | 329.7 ms | 2.0 ms | 4981.3 ms | 5640.0 ms |
| `img2img` | 558.3 ms | — ms | — ms | — ms | — ms | — ms | 3860.0 ms |
| `inpaint` | 497.0 ms | — ms | — ms | — ms | — ms | — ms | 3780.0 ms |
| `outpaint` | 524.3 ms | — ms | — ms | — ms | — ms | — ms | 4620.0 ms |
| `reference` | 505.3 ms | 195.5 ms | 6519.5 ms | 297.5 ms | 2.3 ms | 7201.8 ms | 7910.0 ms |

## Policy review

- Compile versus warm eager: **+16.6%** generation time (slower); compile remains opt-in.
- Eval-frequency 4 versus warm eager: **−0.4%** generation time; too small to justify a default change.
- Defaults change only when a quality-gated path improves its target workload by at least 5%.
- Reference-conditioning optimizations require at least 10%.

The CLI is a measurement harness. App integrations should benchmark their own resolution,
residency, and repetition pattern. `keepResident` favors repeated generation;
`unloadAfterUse` favors minimum between-call memory.
