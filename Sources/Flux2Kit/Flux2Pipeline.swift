// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// 2026-07-19 EDT | PERMANENT (Flux2Kit t2i port) — numerical parity with the reference implementation is the
// contract; do not refactor without re-running the parity harness.

import CoreGraphics
import Foundation
import MLX
import MLXNN
import MLXRandom

// Casts floating-point parameters only (set_dtype semantics).
private func setDtype(_ module: Module, _ dtype: DType) {
    let cast = Dictionary(
        uniqueKeysWithValues: module.parameters().flattened().map { key, value in
            (key, isFloatingPoint(value.dtype) ? value.asType(dtype) : value)
        })
    module.update(parameters: ModuleParameters.unflattened(cast))
}

private func isFloatingPoint(_ dtype: DType) -> Bool {
    switch dtype {
    case .float16, .float32, .bfloat16, .float64:
        return true
    default:
        return false
    }
}

public final class Flux2Pipeline {

    public let repoId: String
    public let repoPath: URL
    public let weightsPath: URL
    public let safeAttn: Bool
    public let vaeFp16: Bool
    public let dtype: DType
    public let quantizeMode: String?
    public private(set) var isDistilled: Bool = false

    // Implicitly-unwrapped so existing access sites are unchanged, but the backing storage can be
    // freed (set to nil) between stages under `.unloadAfterUse` and lazily reloaded.
    public private(set) var model: Flux2Transformer!
    public private(set) var vae: AutoEncoder!
    public private(set) var textEncoder: Qwen3Embedder!

    public var residency: ResidencyPolicy
    public let memReport: Bool
    /// If set, VAE decode is tiled with this latent-space tile size to cap decode-stage memory.
    public var vaeTileLatent: Int?

    // Cached configs + tokenizer so any sub-model can be (re)built on demand.
    private let fluxCfg: Flux2Config
    private let vaeCfg: VAEConfig
    private let qwenCfg: Qwen3Config
    private let tokenizer: Qwen3Tokenizer

    private var cachedEmptyCtx: MLXArray?

    /// `compile` is accepted but ignored — the mx.compile fast path is
    /// deferred in the Swift port (numerics are identical either way; compile is perf-only).
    public init(
        repoId: String = defaultRepoId,
        repoPath: URL? = nil,
        dtype: String = "bfloat16",
        quantize: String? = nil,
        safeAttn: Bool = false,
        vaeFp16: Bool = false,
        compile: Bool = false,
        residency: ResidencyPolicy = .keepResident,
        cacheLimitMB: Int? = nil,
        memoryLimitMB: Int? = nil,
        memReport: Bool = false
    ) async throws {
        self.repoId = repoId
        self.repoPath = try resolveRepoPath(repoId, repoPath)
        self.weightsPath = self.repoPath
        self.safeAttn = safeAttn
        self.vaeFp16 = vaeFp16

        switch dtype {
        case "bfloat16": self.dtype = .bfloat16
        case "float16": self.dtype = .float16
        default: throw Flux2Error.configMissing("Unsupported dtype: \(dtype)")
        }
        self.quantizeMode = quantize
        self.residency = residency
        self.memReport = memReport
        applyMemoryLimits(cacheLimitMB: cacheLimitMB, memoryLimitMB: memoryLimitMB)

        // --- Configs + tokenizer (cheap). Heavy weights load per residency policy below. ---

        let indexPath = self.weightsPath.appendingPathComponent("model_index.json")
        let distilled: Bool
        if let data = try? Data(contentsOf: indexPath),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            distilled = (json["is_distilled"] as? Bool) ?? false
        } else {
            distilled = !repoId.lowercased().contains("base")
        }
        self.isDistilled = distilled

        self.fluxCfg = try loadFlux2Config(
            self.weightsPath.appendingPathComponent("transformer/config.json"))
        self.vaeCfg = try loadVaeConfig(self.weightsPath.appendingPathComponent("vae/config.json"))
        self.qwenCfg = try loadQwen3Config(
            self.weightsPath.appendingPathComponent("text_encoder/config.json"))
        self.tokenizer = try await Qwen3Tokenizer.fromRepo(self.repoPath)

        // keepResident (default): load all three now — identical to prior behavior.
        // unloadAfterUse: nothing heavy is resident yet; models load lazily at stage boundaries.
        if residency == .keepResident {
            self.textEncoder = try makeTextEncoder()
            self.model = try makeTransformer()
            self.vae = try makeVAE()
        }
        if memReport { print(memoryReportLine("after init")) }
    }

    // MARK: - Model loaders (build → dtype → weights → quantize → clearCache)

    private func makeTransformer() throws -> Flux2Transformer {
        let m = try Flux2Transformer(params: fluxCfg)
        m.safeAttn = safeAttn
        setDtype(m, dtype)
        // PE dtype matches model dtype unless safe_attn keeps fp32
        m.peEmbedder.outputDtype = safeAttn ? nil : dtype
        // Transformer weights: native single-file fast path, else diffusers conversion
        var modelWeight: URL?
        for weightFile in weightFiles {
            let candidate = weightsPath.appendingPathComponent(weightFile)
            if FileManager.default.fileExists(atPath: candidate.path) {
                modelWeight = candidate
                break
            }
        }
        if let modelWeight {
            try alignAndLoad(m, try loadSafetensors([modelWeight]), strict: true)
        } else {
            // Sharded diffusers format (e.g. 9B model with index json), then single file fallback
            let diffusersDir = weightsPath.appendingPathComponent("transformer")
            let indexPath = diffusersDir.appendingPathComponent(
                "diffusion_pytorch_model.safetensors.index.json")
            if FileManager.default.fileExists(atPath: indexPath.path) {
                let shardPaths = try resolveShardPaths(diffusersDir, indexFileName: "diffusion_pytorch_model.safetensors.index.json")
                let raw = try loadSafetensors(shardPaths)
                let mapped = try convertFlux2DiffusersWeights(raw, fluxCfg)
                try alignAndLoadFromTorch(m, mapped, strict: true)
            } else {
                let diffusersPath = diffusersDir.appendingPathComponent(
                    "diffusion_pytorch_model.safetensors")
                guard FileManager.default.fileExists(atPath: diffusersPath.path) else {
                    throw Flux2Error.loadFailed("Could not locate transformer weights in repo")
                }
                let raw = try loadSafetensors([diffusersPath])
                let mapped = try convertFlux2DiffusersWeights(raw, fluxCfg)
                try alignAndLoadFromTorch(m, mapped, strict: true)
            }
        }
        quantizeModule(m, mode: quantizeMode)
        MLX.Memory.clearCache()
        return m
    }

    private func makeVAE() throws -> AutoEncoder {
        let v = AutoEncoder(params: vaeCfg)
        if vaeFp16 {
            v.forceUpcast = false
            setDtype(v, .float16)
        } else if vaeCfg.forceUpcast {
            setDtype(v, .float32)
        } else {
            setDtype(v, dtype)
        }
        let vaeWeight = weightsPath.appendingPathComponent(
            "vae/diffusion_pytorch_model.safetensors")
        guard FileManager.default.fileExists(atPath: vaeWeight.path) else {
            throw Flux2Error.loadFailed("Could not locate VAE weights")
        }
        let vaeRaw = try loadSafetensors([vaeWeight])
        let vaeMapped = convertVaeDiffusersWeights(vaeRaw)
        try alignAndLoadFromTorch(v, vaeMapped, strict: true)
        MLX.Memory.clearCache()
        return v
    }

    private func makeTextEncoder() throws -> Qwen3Embedder {
        let emb = Qwen3Embedder(qwenCfg, tokenizer: tokenizer, safeAttn: safeAttn)
        setDtype(emb.model, dtype)
        // Text encoder shards — silent-overwrite merge across shards
        var teDir = weightsPath.appendingPathComponent("text_encoder")
        var shardPaths = listSafetensors(teDir)
        if shardPaths.isEmpty {
            // Base-repo fallback
            if let baseSnapshot = try? resolveRepoPath("black-forest-labs/FLUX.2-klein-4B", nil) {
                teDir = baseSnapshot.appendingPathComponent("text_encoder")
                shardPaths = listSafetensors(teDir)
            }
        }
        guard !shardPaths.isEmpty else {
            throw Flux2Error.loadFailed(
                "Could not locate text encoder weights. "
                    + "Please ensure black-forest-labs/FLUX.2-klein-4B is downloaded.")
        }
        var teWeights: [String: MLXArray] = [:]
        for sp in shardPaths {
            let shard = try loadSafetensors([sp])
            teWeights.merge(shard) { _, new in new }
        }
        teWeights = fuseQkvWeights(teWeights)
        try alignAndLoadFromTorch(emb.model, teWeights, strict: true)
        quantizeModule(emb.model, mode: quantizeMode)
        MLX.Memory.clearCache()
        return emb
    }

    // MARK: - Ensure / unload (staged residency)

    public func ensureTextEncoder() throws {
        if textEncoder == nil { textEncoder = try makeTextEncoder() }
    }
    public func ensureTransformer() throws {
        if model == nil { model = try makeTransformer() }
    }
    public func ensureVAE() throws {
        if vae == nil { vae = try makeVAE() }
    }

    /// Free a stage's model — no-op under `.keepResident`.
    public func unloadTextEncoder() {
        if residency == .unloadAfterUse { textEncoder = nil; MLX.Memory.clearCache() }
    }
    public func unloadTransformer() {
        if residency == .unloadAfterUse { model = nil; MLX.Memory.clearCache() }
    }
    public func unloadVAE() {
        if residency == .unloadAfterUse { vae = nil; MLX.Memory.clearCache() }
    }

    /// Per-stage memory line when `--mem-report` is on.
    public func reportMemory(_ stage: String) {
        if memReport { print(memoryReportLine(stage)) }
    }

    // Tokenize and run the text encoder model over a batch of prompts.
    private func encodeText(_ prompts: [String], verbose: Bool = false) throws
        -> (MLXArray, [String: Double])
    {
        var timings: [String: Double] = [:]

        // Load the text encoder if a low-memory session freed it (safe no-op when resident).
        try ensureTextEncoder()

        var t0 = ProcessInfo.processInfo.systemUptime
        let (inputIds, attentionMask) = try textEncoder.tokenize(prompts)
        if verbose { eval(inputIds, attentionMask) }
        timings["tokenize"] = ProcessInfo.processInfo.systemUptime - t0

        t0 = ProcessInfo.processInfo.systemUptime
        let ctx = textEncoder.model(inputIds, attentionMask)
        if verbose { eval(ctx) }
        timings["model"] = ProcessInfo.processInfo.systemUptime - t0

        return (ctx, timings)
    }

    // Encode a prompt into its final context tensor, handling CFG empty-context caching.
    public func encodePrompt(
        _ prompt: String, guidanceDistilled: Bool, verbose: Bool = false
    ) throws -> (MLXArray, MLXArray, [String: Double]?) {
        var allTimings: [String: Double] = [:]

        var ctx: MLXArray
        if guidanceDistilled {
            let (encoded, timings) = try encodeText([prompt], verbose: verbose)
            ctx = encoded
            allTimings.merge(timings) { _, new in new }
        } else {
            var ctxPrompt: MLXArray
            if cachedEmptyCtx == nil {
                let (both, timings) = try encodeText(["", prompt], verbose: verbose)
                allTimings.merge(timings) { _, new in new }
                eval(both)
                cachedEmptyCtx = both[..<1]
                ctxPrompt = both[1...]
            } else {
                let (encoded, timings) = try encodeText([prompt], verbose: verbose)
                ctxPrompt = encoded
                allTimings.merge(timings) { _, new in new }
            }
            guard let cachedEmptyCtx else {
                throw Flux2Error.generationFailed("CFG empty-context cache unexpectedly nil")
            }
            ctx = concatenated([cachedEmptyCtx, ctxPrompt], axis: 0)
        }

        let t0 = ProcessInfo.processInfo.systemUptime
        let (ctxOut, ctxIds) = batchedPrcTxt(ctx)
        if verbose { eval(ctxOut, ctxIds) }
        allTimings["prc_txt"] = ProcessInfo.processInfo.systemUptime - t0

        return (ctxOut, ctxIds, verbose ? allTimings : nil)
    }

    // Full text-to-image generation pipeline: encode, denoise, decode.
    public func generate(
        prompt: String,
        width: Int = defaultWidth,
        height: Int = defaultHeight,
        numSteps: Int = defaultSteps,
        guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil,
        inputImages: [CGImage]? = nil,
        guidanceDistilled: Bool? = nil,
        verbose: Bool = false,
        evalFreq: Int = 1
    ) throws -> CGImage {
        let guidanceDistilled = guidanceDistilled ?? isDistilled

        guard width % 16 == 0, height % 16 == 0 else {
            throw Flux2Error.generationFailed("width and height must be divisible by 16")
        }

        if let seed {
            MLXRandom.seed(seed)
        }

        var timings: [String: Double] = [:]
        let totalStart = ProcessInfo.processInfo.systemUptime

        try ensureTextEncoder()
        reportMemory("pre-encode")
        var t0 = ProcessInfo.processInfo.systemUptime
        let (ctx, ctxIds, teBreakdown) = try encodePrompt(
            prompt, guidanceDistilled: guidanceDistilled, verbose: verbose)
        if verbose { eval(ctx, ctxIds) }
        timings["text_encode"] = ProcessInfo.processInfo.systemUptime - t0

        if verbose {
            let ms = (timings["text_encode"] ?? 0) * 1000
            print(String(format: "[%7.1fms] Text encode: %d tokens, shape %@",
                         ms, ctx.dim(1), String(describing: ctx.shape)))
            if let teBreakdown {
                print(String(format: "           ├─ tokenize: %.1fms", (teBreakdown["tokenize"] ?? 0) * 1000))
                print(String(format: "           ├─ model:    %.1fms", (teBreakdown["model"] ?? 0) * 1000))
                print(String(format: "           └─ prc_txt:  %.1fms", (teBreakdown["prc_txt"] ?? 0) * 1000))
            }
        }

        var imgCondSeq: MLXArray?
        var imgCondSeqIds: MLXArray?
        if let inputImages, !inputImages.isEmpty {
            try ensureVAE()
            t0 = ProcessInfo.processInfo.systemUptime
            (imgCondSeq, imgCondSeqIds) = try encodeImageRefs(vae, inputImages)
            if verbose, let s = imgCondSeq, let i = imgCondSeqIds {
                eval(s, i)
                timings["ref_encode"] = ProcessInfo.processInfo.systemUptime - t0
                print(String(format: "[%7.1fms] Ref encode: %d tokens",
                             (timings["ref_encode"] ?? 0) * 1000, s.dim(1)))
            }
        }

        unloadTextEncoder()
        try ensureTransformer()
        reportMemory("pre-denoise")
        t0 = ProcessInfo.processInfo.systemUptime
        let batchSize = 1
        let latentChannels = model.inChannels
        let noise = MLXRandom.normal(
            [batchSize, latentChannels, height / 16, width / 16], dtype: dtype)
        var (x, xIds) = batchedPrcImg(noise)
        if verbose { eval(x, xIds) }
        timings["noise_init"] = ProcessInfo.processInfo.systemUptime - t0

        let timesteps = getSchedule(numSteps, x.dim(1))
        if verbose {
            print(String(format: "[%7.1fms] Noise init: %d latent tokens",
                         (timings["noise_init"] ?? 0) * 1000, x.dim(1)))
        }

        t0 = ProcessInfo.processInfo.systemUptime
        var imgInputIds = xIds
        if let imgCondSeqIds {
            imgInputIds = concatenated([xIds, imgCondSeqIds], axis: 1)
        }
        if !guidanceDistilled {
            imgInputIds = concatenated([imgInputIds, imgInputIds], axis: 0)
        }
        let peX = model.peEmbedder(imgInputIds)
        let peCtx = model.peEmbedder(ctxIds)
        if verbose { eval(peX, peCtx) }
        timings["pe_embed"] = ProcessInfo.processInfo.systemUptime - t0

        if verbose {
            print(String(format: "[%7.1fms] Position embeddings", (timings["pe_embed"] ?? 0) * 1000))
        }

        let stepTimes = Flux2StepTimes()

        let logStep: (Int, Double, Double, MLXArray, MLXArray) -> Void = { step, tCurr, tPrev, _, _ in
            let stepTime = stepTimes.values.last ?? 0
            print(String(format: "[%7.1fms] Step %d/%d  t=%.4f→%.4f",
                         stepTime * 1000, step + 1, numSteps, tCurr, tPrev))
        }

        let modelFn: Flux2ModelFn = { [model] x, xIds, t, ctx, ctxIds, g, peX, peCtx, txtEmb, gEmb in
            model(x, xIds, t, ctx, ctxIds, g, peX, peCtx, txtEmb, gEmb)
        }
        let modelFnCfg: Flux2ModelCfgFn = { [model] x, xIds, t, ctx, ctxIds, peX, peCtx, txtEmb in
            model(x, xIds, t, ctx, ctxIds, nil, peX, peCtx, txtEmb, nil)
        }

        t0 = ProcessInfo.processInfo.systemUptime
        if guidanceDistilled {
            x = denoise(
                model, x, xIds, ctx, ctxIds,
                timesteps: timesteps,
                guidance: guidance,
                imgCondSeq: imgCondSeq,
                imgCondSeqIds: imgCondSeqIds,
                logFn: verbose ? logStep : nil,
                peX: peX,
                peCtx: peCtx,
                modelFn: modelFn,
                stepTimes: verbose ? stepTimes : nil,
                evalFreq: evalFreq)
        } else {
            x = denoiseCfg(
                model, x, xIds, ctx, ctxIds,
                timesteps: timesteps,
                guidance: guidance,
                imgCondSeq: imgCondSeq,
                imgCondSeqIds: imgCondSeqIds,
                logFn: verbose ? logStep : nil,
                peX: peX,
                peCtx: peCtx,
                modelFn: modelFn,
                modelFnCfg: modelFnCfg,
                stepTimes: verbose ? stepTimes : nil,
                evalFreq: evalFreq)
        }
        timings["denoise"] = ProcessInfo.processInfo.systemUptime - t0

        if verbose {
            let total = timings["denoise"] ?? 0
            print(String(format: "[%7.1fms] Denoise total (%d steps, %.1fms/step avg)",
                         total * 1000, numSteps, total / Double(numSteps) * 1000))
        }

        t0 = ProcessInfo.processInfo.systemUptime
        x = concatenated(scatterIds(x, xIds), axis: 0)
        if x.dim(2) == 1 {
            x = x.squeezed(axis: 2)
        } else {
            if verbose {
                print("         Warning: time dimension \(x.dim(2)) > 1, using t=0 slice")
            }
            x = x[0..., 0..., 0, 0..., 0...]
        }
        x = x.transposed(0, 2, 3, 1)
        eval(x)
        timings["scatter"] = ProcessInfo.processInfo.systemUptime - t0

        if verbose {
            print(String(format: "[%7.1fms] Scatter/reshape", (timings["scatter"] ?? 0) * 1000))
        }

        unloadTransformer()
        try ensureVAE()
        reportMemory("pre-decode")
        t0 = ProcessInfo.processInfo.systemUptime
        let decoded = decodeMaybeTiled(x)
        eval(decoded)
        timings["vae_decode"] = ProcessInfo.processInfo.systemUptime - t0

        if verbose {
            print(String(format: "[%7.1fms] VAE decode", (timings["vae_decode"] ?? 0) * 1000))
        }

        t0 = ProcessInfo.processInfo.systemUptime
        let result = try arrayToCGImage(decoded[0])
        timings["to_pil"] = ProcessInfo.processInfo.systemUptime - t0

        let totalTime = ProcessInfo.processInfo.systemUptime - totalStart
        if verbose {
            print(String(format: "[%7.1fms] To PIL", (timings["to_pil"] ?? 0) * 1000))
            print(String(format: "[%7.1fms] TOTAL", totalTime * 1000))
        }

        return result
    }
}
