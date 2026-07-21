// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// 2026-07-19 EDT | PERMANENT (Flux2Kit t2i port) — numerical parity with the reference implementation is the
// contract; do not refactor without re-running the parity harness.

import Foundation
import MLX
import MLXFast
import MLXNN
import Tokenizers

// Layer indices at which hidden states are tapped for the encoder output.
public let outputLayersQwen3 = textEncoderOutputLayers

// FastRMSNorm (HF weight key "weight")
public final class Qwen3RMSNorm: Module {

    @ParameterInfo(key: "weight") public var weight: MLXArray
    public let eps: Float

    public init(_ dim: Int, eps: Float = 1e-6) {
        self._weight.wrappedValue = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// Fused qkv projection (see fuseQkvWeights)
public final class Qwen3Attention: Module {

    public let safeAttn: Bool
    public let nHeads: Int
    public let nKvHeads: Int
    public let headDim: Int
    public let scale: Float
    public let nRep: Int
    public let qDim: Int
    public let kvDim: Int
    public let ropeTheta: Float

    @ModuleInfo(key: "qkv_proj") public var qkvProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: Qwen3RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: Qwen3RMSNorm

    public init(_ cfg: Qwen3Config, safeAttn: Bool = false) {
        self.safeAttn = safeAttn
        self.nHeads = cfg.numAttentionHeads
        self.nKvHeads = cfg.numKeyValueHeads
        self.headDim = cfg.headDim
        self.scale = Float(pow(Double(cfg.headDim), -0.5))
        self.nRep = cfg.numAttentionHeads / cfg.numKeyValueHeads
        self.qDim = cfg.numAttentionHeads * cfg.headDim
        self.kvDim = cfg.numKeyValueHeads * cfg.headDim
        self.ropeTheta = cfg.ropeTheta

        self._qkvProj.wrappedValue = Linear(cfg.hiddenSize, qDim + 2 * kvDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, cfg.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Qwen3RMSNorm(cfg.headDim, eps: cfg.rmsNormEps)
        self._kNorm.wrappedValue = Qwen3RMSNorm(cfg.headDim, eps: cfg.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ mask: MLXArray?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let origDtype = x.dtype

        let qkv = qkvProj(x)
        var q = qkv[.ellipsis, ..<qDim].reshaped(b, l, nHeads, headDim)
        var k = qkv[.ellipsis, qDim ..< (qDim + kvDim)].reshaped(b, l, nKvHeads, headDim)
        var v = qkv[.ellipsis, (qDim + kvDim)...].reshaped(b, l, nKvHeads, headDim)

        q = qNorm(q).transposed(0, 2, 1, 3)
        k = kNorm(k).transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: ropeTheta, scale: 1.0, offset: 0)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: ropeTheta, scale: 1.0, offset: 0)

        if nRep > 1 {
            // Broadcast (not repeat) to expand KV heads
            k = broadcast(k[0..., 0..., .newAxis, 0..., 0...], to: [b, nKvHeads, nRep, l, headDim])
                .reshaped(b, nHeads, l, headDim)
            v = broadcast(v[0..., 0..., .newAxis, 0..., 0...], to: [b, nKvHeads, nRep, l, headDim])
                .reshaped(b, nHeads, l, headDim)
        }

        var out: MLXArray
        if safeAttn {
            let qf = q.asType(.float32)
            let kf = k.asType(.float32)
            let vf = v.asType(.float32)
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
                mask.map { .array($0.asType(.float32)) } ?? .none
            out = MLXFast.scaledDotProductAttention(
                queries: qf, keys: kf, values: vf, scale: scale, mask: maskMode)
            out = out.asType(origDtype)
        } else {
            var m = mask
            if let mm = m, mm.dtype != q.dtype {
                m = mm.asType(q.dtype)
            }
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = m.map { .array($0) } ?? .none
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        }

        out = out.transposed(0, 2, 1, 3).reshaped(b, l, -1)
        return oProj(out)
    }
}

// Qwen3 MLP block (SwiGLU: silu(gate) * up, then down projection).
public final class Qwen3MLP: Module {

    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(_ dim: Int, _ hiddenDim: Int) {
        self._gateProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._upProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDim, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// Transformer block: self-attention + MLP with pre-norm.
public final class Qwen3Block: Module {

    @ModuleInfo(key: "self_attn") public var selfAttn: Qwen3Attention
    @ModuleInfo(key: "mlp") public var mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") public var inputLayernorm: Qwen3RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayernorm: Qwen3RMSNorm

    public init(_ cfg: Qwen3Config, safeAttn: Bool = false) {
        self._selfAttn.wrappedValue = Qwen3Attention(cfg, safeAttn: safeAttn)
        self._mlp.wrappedValue = Qwen3MLP(cfg.hiddenSize, cfg.intermediateSize)
        self._inputLayernorm.wrappedValue = Qwen3RMSNorm(cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = Qwen3RMSNorm(cfg.hiddenSize, eps: cfg.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ mask: MLXArray?) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask)
        return h + mlp(postAttentionLayernorm(h))
    }
}

// Taps hidden states at layers 9/18/27
public final class Qwen3Backbone: Module {

    public let cfg: Qwen3Config

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Qwen3Block]
    @ModuleInfo(key: "norm") public var norm: Qwen3RMSNorm

    public init(_ cfg: Qwen3Config, safeAttn: Bool = false) {
        self.cfg = cfg
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        self._layers.wrappedValue = (0 ..< cfg.numHiddenLayers).map { _ in
            Qwen3Block(cfg, safeAttn: safeAttn)
        }
        self._norm.wrappedValue = Qwen3RMSNorm(cfg.hiddenSize, eps: cfg.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ inputIds: MLXArray, _ attentionMask: MLXArray? = nil) -> MLXArray {
        var h = embedTokens(inputIds)
        var mask: MLXArray?
        if let attentionMask {
            mask = buildCausalPaddingMask(attentionMask, dtype: h.dtype)
        }
        let outputLayers = Set(outputLayersQwen3)
        let maxOutputLayer = outputLayersQwen3.max() ?? 0
        var selectedStates: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask)
            if outputLayers.contains(i + 1) {
                selectedStates.append(h)
            }
            if (i + 1) >= maxOutputLayer {
                break
            }
        }
        if !selectedStates.isEmpty {
            return concatenated(selectedStates, axis: -1)
        }
        return norm(h)
    }
}

// Top-level Qwen3 model wrapping the backbone.
public final class Qwen3Model: Module {

    @ModuleInfo(key: "model") public var model: Qwen3Backbone

    public init(_ cfg: Qwen3Config, safeAttn: Bool = false) {
        self._model.wrappedValue = Qwen3Backbone(cfg, safeAttn: safeAttn)
        super.init()
    }

    public func callAsFunction(_ inputIds: MLXArray, _ attentionMask: MLXArray? = nil) -> MLXArray {
        model(inputIds, attentionMask)
    }
}

// Plain class, not a Module
public final class Qwen3Embedder {

    public let cfg: Qwen3Config
    public let model: Qwen3Model
    public let tokenizer: Qwen3Tokenizer
    public let maxLength: Int

    public init(_ cfg: Qwen3Config, tokenizer: Qwen3Tokenizer, safeAttn: Bool = false) {
        self.cfg = cfg
        self.model = Qwen3Model(cfg, safeAttn: safeAttn)
        self.tokenizer = tokenizer
        self.maxLength = textEncoderMaxLength
    }

    public func tokenize(_ prompts: [String]) throws -> (MLXArray, MLXArray) {
        try tokenizer.encodeBatch(prompts, maxLength: maxLength)
    }

    public func callAsFunction(_ prompts: [String]) throws -> MLXArray {
        let (inputIds, attentionMask) = try tokenize(prompts)
        return model(inputIds, attentionMask)
    }
}

// These tiny tensors are recomputed on each call rather than cached in module-global dicts;
// recomputation is numerically identical (perf-only deviation).
private func negInf(_ dtype: DType) -> MLXArray {
    // mx.finfo(dtype).min — minimum FINITE value (not -inf), avoiding NaN on fully-masked rows.
    let minValue = dtype.finfo?.min ?? -Double(Float.greatestFiniteMagnitude)
    return MLXArray(minValue).asType(dtype)
}

// True = invalid/future position
private func causalTriangle(_ l: Int) -> MLXArray {
    let idx = MLXArray(0 ..< l)
    return idx[0..., .newAxis] .< idx[.newAxis, 0...]
}

// Builds a causal-only attention mask.
public func buildCausalOnlyMask(_ l: Int, dtype: DType) -> MLXArray {
    let causal = causalTriangle(l)
    let neg = negInf(dtype)
    let zero = MLXArray(0).asType(dtype)
    let mask = MLX.where(causal, neg, zero)
    return mask[.newAxis, .newAxis, 0..., 0...]
}

// Builds a causal mask combined with padding.
public func buildCausalPaddingMask(_ attentionMask: MLXArray, dtype: DType = .float32) -> MLXArray {
    let l = attentionMask.dim(1)

    // Fast path: no padding anywhere (host sync intentional)
    let allValid = MLX.all(attentionMask .== 1)
    eval(allValid)
    if allValid.item(Bool.self) {
        return buildCausalOnlyMask(l, dtype: dtype)
    }

    let causal = causalTriangle(l)
    let keyMask = attentionMask.asType(.bool)
    let invalid = causal[.newAxis, 0..., 0...] .|| .!keyMask[0..., .newAxis, 0...]

    let neg = negInf(dtype)
    let zero = MLXArray(0).asType(dtype)
    let mask = MLX.where(invalid, neg, zero)
    return mask[0..., .newAxis, 0..., 0...]
}

// Backed by swift-transformers instead of the HF `tokenizers` Rust library. The jinja chat
// template is NOT rendered at runtime: for the only message shape this pipeline ever produces
// (single user message, add_generation_prompt=true, enable_thinking=false),
// Models/FLUX-2/tokenizer/chat_template.jinja renders to the exact string below — verified
// against the reference implementation (repr-compared, token ids golden-tested in Flux2KitTests).
public final class Qwen3Tokenizer {

    public let tokenizer: any Tokenizer
    public let padId: Int
    public let eosId: Int

    public init(tokenizer: any Tokenizer, padId: Int, eosId: Int) {
        self.tokenizer = tokenizer
        self.padId = padId
        self.eosId = eosId
    }

    public static func fromRepo(_ repoPath: URL) async throws -> Qwen3Tokenizer {
        var tokDir = repoPath.appendingPathComponent("tokenizer")
        if !FileManager.default.fileExists(
            atPath: tokDir.appendingPathComponent("tokenizer.json").path)
        {
            // cwd fallback directory
            let fallback = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(tokenizerFallbackDir)
                .appendingPathComponent("tokenizer")
            if FileManager.default.fileExists(atPath: fallback.path) {
                tokDir = fallback
            }
        }
        guard FileManager.default.fileExists(
            atPath: tokDir.appendingPathComponent("tokenizer.json").path)
        else {
            throw Flux2Error.tokenizerFailed(
                "Tokenizer not found at \(tokDir.appendingPathComponent("tokenizer.json").path)")
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)

        // Special tokens from special_tokens_map.json with defaults
        var padToken = "<|endoftext|>"
        var eosToken = "<|im_end|>"
        let specialMap = tokDir.appendingPathComponent("special_tokens_map.json")
        if let data = try? Data(contentsOf: specialMap),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let pad = json["pad_token"] as? [String: Any], let c = pad["content"] as? String {
                padToken = c
            }
            if let eos = json["eos_token"] as? [String: Any], let c = eos["content"] as? String {
                eosToken = c
            }
        }
        guard let padId = tokenizer.convertTokenToId(padToken),
            let eosId = tokenizer.convertTokenToId(eosToken)
        else {
            throw Flux2Error.tokenizerFailed("Tokenizer missing required special tokens")
        }
        return Qwen3Tokenizer(tokenizer: tokenizer, padId: padId, eosId: eosId)
    }

    // Pre-rendered template (see class comment); verified render:
    // '<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    public func applyChatTemplate(_ prompt: String, addGenerationPrompt: Bool = true) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    // Encodes a batch of prompts into padded input ids and attention masks.
    public func encodeBatch(
        _ prompts: [String], maxLength: Int, padToMax: Bool = false
    ) throws -> (MLXArray, MLXArray) {
        guard !prompts.isEmpty else {
            throw Flux2Error.tokenizerFailed("encodeBatch() requires at least one prompt")
        }
        let texts = prompts.map { applyChatTemplate($0, addGenerationPrompt: true) }

        var inputIds: [[Int]] = []
        var attentionMasks: [[Int]] = []
        for text in texts {
            // parity: HF tokenizers encode_batch applies no extra specials for Qwen (the template
            // string already contains them); golden-token test pins this behavior.
            let ids = Array(tokenizer.encode(text: text, addSpecialTokens: false).prefix(maxLength))
            inputIds.append(ids)
            attentionMasks.append(Array(repeating: 1, count: ids.count))
        }

        let actualMax = inputIds.map(\.count).max() ?? 0
        let targetLen = padToMax ? maxLength : min(maxLength, ((actualMax + 63) / 64) * 64)

        for i in inputIds.indices {
            let padLen = targetLen - inputIds[i].count
            if padLen > 0 {
                inputIds[i] += Array(repeating: padId, count: padLen)
                attentionMasks[i] += Array(repeating: 0, count: padLen)
            }
        }

        let idsFlat = inputIds.flatMap { $0.map { Int32($0) } }
        let maskFlat = attentionMasks.flatMap { $0.map { Int32($0) } }
        let shape = [inputIds.count, targetLen]
        return (MLXArray(idsFlat, shape), MLXArray(maskFlat, shape))
    }
}
