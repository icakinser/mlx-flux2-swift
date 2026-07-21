// Flux2Kit — memory system: quantization filter, residency policy, MLX cache/memory limits, and
// per-stage reporting. Opt-in; the default keeps all models resident in bf16 (the parity path).
//
// Inference is bandwidth-bound and the three sub-models (text encoder, transformer, VAE) run
// sequentially, so the big levers are (1) quantization (fewer weight bytes to store AND read) and
// (2) staged residency (free each model when its stage is done). See Flux2Pipeline for wiring.

import Foundation
import MLX
import MLXNN

/// Whether sub-models stay resident for the pipeline's lifetime (fast for repeated generations) or
/// are freed after each stage (lowest peak memory, single-shot friendly).
public enum ResidencyPolicy: Sendable {
    case keepResident
    case unloadAfterUse
}

/// Quantization layer filter. Quantizes only 2-D `Linear` weights whose dims align to `groupSize`,
/// and skips the `adaLN_modulation` `[SiLU, Linear]` container — that mixed-type module array breaks
/// `MLXNN.quantize`'s tree-walk (`unexpectedStructure(key: "adaLN_modulation")`). Skipping the small
/// modulation/norm layers and quantizing the big attention/MLP matmuls is the standard FLUX recipe.
public func flux2QuantFilter(groupSize: Int) -> (String, Module) -> Bool {
    { path, m in
        guard let lin = m as? Linear else { return false }
        if path.contains("adaLN") { return false }
        let w = lin.weight
        return w.ndim == 2 && w.dim(0) % groupSize == 0 && w.dim(1) % groupSize == 0
    }
}

/// Quantize a module in place using the FLUX filter. No-op unless `mode` is "int8"/"int4".
public func quantizeModule(_ module: Module, mode: String?, groupSize: Int = 64) {
    guard let mode, mode == "int8" || mode == "int4" else { return }
    let bits = mode == "int8" ? 8 : 4
    MLXNN.quantize(
        model: module, groupSize: groupSize, bits: bits,
        filter: flux2QuantFilter(groupSize: groupSize))
}

/// Apply MLX buffer-cache / soft memory limits, in megabytes. `nil` leaves MLX defaults.
public func applyMemoryLimits(cacheLimitMB: Int?, memoryLimitMB: Int?) {
    if let c = cacheLimitMB { MLX.Memory.cacheLimit = c * 1_048_576 }
    if let m = memoryLimitMB { MLX.Memory.memoryLimit = m * 1_048_576 }
}

/// Process resident set size (bytes) — the true footprint, including mmap'd weight pages that MLX's
/// own counters do not track. Returns 0 if the mach call fails.
public func processResidentBytes() -> Int {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Int(info.resident_size) : 0
}

/// One-line memory report for `--mem-report`: process RSS (the real footprint) plus MLX active/peak
/// buffer counters. MLX's counters exclude mmap'd weights, so RSS is the number that reflects the
/// staged-residency win.
public func memoryReportLine(_ stage: String) -> String {
    func gb(_ bytes: Int) -> String { String(format: "%.2fGB", Double(bytes) / 1_073_741_824) }
    let label = stage.padding(toLength: 18, withPad: " ", startingAt: 0)
    return "[mem] \(label) rss=\(gb(processResidentBytes())) "
        + "mlx_active=\(gb(MLX.Memory.activeMemory)) mlx_peak=\(gb(MLX.Memory.peakMemory))"
}
