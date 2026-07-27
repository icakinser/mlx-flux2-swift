# Flux2Kit Release Benchmark

**Generated:** 2026-07-27 01:15:04 UTC  
**Host:** arm64, macOS 26.5.1  
**Workload:** 512×512, 4 steps, guidance 1.0, seed 42  
**Build:** Swift release

| Path | Pipeline load | Text encode | Denoise | VAE decode | To image | Generation total |
|---|---:|---:|---:|---:|---:|---:|
| `eager-first` | 563.9 ms | 213.7 ms | 2957.2 ms | 333.4 ms | 2.0 ms | 3518.0 ms |
| `eager-warm` | 540.3 ms | 204.8 ms | 2872.9 ms | 358.5 ms | 2.0 ms | 3451.7 ms |
| `compile` | 586.4 ms | 216.8 ms | 2932.9 ms | 342.7 ms | 1.8 ms | 3505.9 ms |
| `heun` | 600.3 ms | 222.0 ms | 4646.5 ms | 330.6 ms | 1.9 ms | 5212.2 ms |

The CLI is a measurement harness. App integrations should benchmark their own resolution,
residency, and repetition pattern. `keepResident` favors repeated generation;
`unloadAfterUse` favors minimum between-call memory.
