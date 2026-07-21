// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// 2026-07-19 EDT | PERMANENT (Flux2Kit t2i port) — numerical parity with the reference implementation is the
// contract; do not refactor without re-running the parity harness.

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// RMS normalization with a learnable scale parameter.
public final class Flux2RMSNorm: Module {

    @ParameterInfo(key: "scale") public var scale: MLXArray
    public let eps: Float

    public init(_ dim: Int, eps: Float = 1e-6) {
        self._scale.wrappedValue = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: scale, eps: eps)
    }
}

// LayerNorm without learnable parameters
public final class FastLayerNorm: Module {

    public let eps: Float

    public init(_ dim: Int, eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
    }
}

// Applies RMSNorm to query and key projections.
public final class QKNorm: Module {

    @ModuleInfo(key: "query_norm") public var queryNorm: Flux2RMSNorm
    @ModuleInfo(key: "key_norm") public var keyNorm: Flux2RMSNorm

    public init(_ dim: Int, eps: Float = 1e-6) {
        self._queryNorm.wrappedValue = Flux2RMSNorm(dim, eps: eps)
        self._keyNorm.wrappedValue = Flux2RMSNorm(dim, eps: eps)
        super.init()
    }

    public func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let qn = queryNorm(q)
        let kn = keyNorm(k)
        return (qn.asType(v.dtype), kn.asType(v.dtype))
    }
}

// Gated SiLU over a channel split
public final class SiLUActivation: Module, UnaryLayer {

    override public init() {
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(x, parts: 2, axis: -1)
        return silu(parts[0]) * parts[1]
    }
}

// The freqs tensor is cached in a module-global
// dict; recomputation here is numerically identical (perf-only deviation, ~64 values per call).
public func timestepEmbedding(
    _ t: MLXArray, _ dim: Int, maxPeriod: Int = 10000, timeFactor: Float = 1000.0
) -> MLXArray {
    let t1 = t * timeFactor
    let half = dim / 2
    let freqs = exp(
        -log(Float(maxPeriod)) * MLXArray(0 ..< half).asType(.float32) / Float(half)
    )
    let args = t1[0..., .newAxis].asType(.float32) * freqs[.newAxis]
    var emb = concatenated([cos(args), sin(args)], axis: -1)
    if dim % 2 != 0 {
        emb = concatenated([emb, MLXArray.zeros([emb.dim(0), 1], dtype: emb.dtype)], axis: -1)
    }
    return emb.asType(t1.dtype)
}

// Computes RoPE frequencies for a given dimension and theta.
public func computeRopeFrequencies(_ dim: Int, _ theta: Float) -> MLXArray {
    precondition(dim % 2 == 0, "RoPE dim must be even")  // RoPE dim must be even
    let scale = MLXArray(stride(from: Int32(0), to: Int32(dim), by: 2)).asType(.float32) / Float(dim)
    return 1.0 / pow(MLXArray(theta), scale)
}

// float32 compute, optional cast to model dtype
public func rope(
    _ pos: MLXArray, _ dim: Int, _ theta: Float, _ omega: MLXArray? = nil, dtype: DType? = nil
) -> MLXArray {
    let omega = omega ?? computeRopeFrequencies(dim, theta)
    let out = pos.asType(.float32)[.ellipsis, .newAxis] * omega
    let cosOut = cos(out)
    let sinOut = sin(out)
    let stackedOut = stacked([cosOut, -sinOut, sinOut, cosOut], axis: -1)
    let newShape = Array(stackedOut.shape.dropLast()) + [2, 2]
    var result = stackedOut.reshaped(newShape)
    if let dtype, dtype != .float32 {
        result = result.asType(dtype)
    }
    return result
}

// Applies rotary position embeddings to query and key tensors.
public func applyRope(_ xq: MLXArray, _ xk: MLXArray, _ freqsCis: MLXArray) -> (MLXArray, MLXArray) {
    let (b, h, l, d) = (xq.dim(0), xq.dim(1), xq.dim(2), xq.dim(3))
    let xq_ = xq.reshaped(b, h, l, d / 2, 1, 2)
    let xk_ = xk.reshaped(b, h, l, d / 2, 1, 2)
    let xqOut = freqsCis[.ellipsis, 0] * xq_[.ellipsis, 0] + freqsCis[.ellipsis, 1] * xq_[.ellipsis, 1]
    let xkOut = freqsCis[.ellipsis, 0] * xk_[.ellipsis, 0] + freqsCis[.ellipsis, 1] * xk_[.ellipsis, 1]
    return (xqOut.reshaped(b, h, l, d), xkOut.reshaped(b, h, l, d))
}

// Scaled dot-product attention with RoPE applied to queries and keys.
public func attention(
    _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ pe: MLXArray, _ scale: Float,
    safeAttn: Bool = false
) -> MLXArray {
    var out: MLXArray
    if safeAttn {
        let qf = q.asType(.float32)
        let kf = k.asType(.float32)
        let vf = v.asType(.float32)
        let (qr, kr) = applyRope(qf, kf, pe)
        out = MLXFast.scaledDotProductAttention(
            queries: qr, keys: kr, values: vf, scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
        out = out.asType(q.dtype)
    } else {
        let (qr, kr) = applyRope(q, k, pe)
        out = MLXFast.scaledDotProductAttention(
            queries: qr, keys: kr, values: v, scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
    }
    out = out.transposed(0, 2, 1, 3)
    return out.reshaped(out.dim(0), out.dim(1), -1)
}

// 4-axis RoPE embedder
public final class EmbedND: Module {

    public let dim: Int
    public let theta: Float
    public let axesDim: [Int]
    // Leading underscore keeps these out of the parameter tree (same rule as the
    // _omega_cache / _output_dtype exclusion).
    private let _omegaCache: [MLXArray]
    /// Set by the pipeline for dtype matching (pe_embedder._output_dtype).
    public var outputDtype: DType?

    public init(dim: Int, theta: Float, axesDim: [Int]) {
        self.dim = dim
        self.theta = theta
        self.axesDim = axesDim
        self._omegaCache = axesDim.map { computeRopeFrequencies($0, theta) }
        self.outputDtype = nil
        super.init()
    }

    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let parts = axesDim.indices.map { i in
            rope(ids[.ellipsis, i], axesDim[i], theta, _omegaCache[i], dtype: outputDtype)
        }
        let emb = concatenated(parts, axis: -3)
        return expandedDimensions(emb, axis: 1)
    }
}

// Two-layer MLP embedder with SiLU activation.
public final class MLPEmbedder: Module {

    @ModuleInfo(key: "in_layer") public var inLayer: Linear
    @ModuleInfo(key: "out_layer") public var outLayer: Linear

    public init(inDim: Int, hiddenDim: Int, disableBias: Bool = false) {
        self._inLayer.wrappedValue = Linear(inDim, hiddenDim, bias: !disableBias)
        self._outLayer.wrappedValue = Linear(hiddenDim, hiddenDim, bias: !disableBias)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        outLayer(silu(inLayer(x)))
    }
}

public typealias ModulationTriple = (MLXArray, MLXArray, MLXArray)

// Produces modulation (shift, scale, gate) triples from a conditioning vector.
public final class Modulation: Module {

    public let isDouble: Bool
    public let multiplier: Int

    @ModuleInfo(key: "lin") public var lin: Linear

    public init(_ dim: Int, double: Bool, disableBias: Bool = false) {
        self.isDouble = double
        self.multiplier = double ? 6 : 3
        self._lin.wrappedValue = Linear(dim, self.multiplier * dim, bias: !disableBias)
        super.init()
    }

    public func callAsFunction(_ vec: MLXArray) -> (ModulationTriple, ModulationTriple?) {
        var out = lin(silu(vec))
        if out.ndim == 2 {
            out = out[0..., .newAxis, 0...]
        }
        let chunks = split(out, parts: multiplier, axis: -1)
        let first = (chunks[0], chunks[1], chunks[2])
        if isDouble {
            return (first, (chunks[3], chunks[4], chunks[5]))
        }
        return (first, nil)
    }
}

// Submodule container only, no forward
public final class SelfAttention: Module {

    public let numHeads: Int

    @ModuleInfo(key: "qkv") public var qkv: Linear
    @ModuleInfo(key: "norm") public var norm: QKNorm
    @ModuleInfo(key: "proj") public var proj: Linear

    public init(_ dim: Int, numHeads: Int = 8) {
        self.numHeads = numHeads
        let headDim = dim / numHeads
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: false)
        self._norm.wrappedValue = QKNorm(headDim)
        self._proj.wrappedValue = Linear(dim, dim, bias: false)
        super.init()
    }
}

// Single-stream transformer block combining attention and MLP.
public final class SingleStreamBlock: Module {

    public let scale: Float
    public let hiddenSize: Int
    public let numHeads: Int
    public let mlpHiddenDim: Int
    public let mlpMultFactor: Int

    @ModuleInfo(key: "linear1") public var linear1: Linear
    @ModuleInfo(key: "linear2") public var linear2: Linear
    @ModuleInfo(key: "norm") public var norm: QKNorm
    @ModuleInfo(key: "pre_norm") public var preNorm: FastLayerNorm
    @ModuleInfo(key: "mlp_act") public var mlpAct: SiLUActivation

    public init(hiddenSize: Int, numHeads: Int, mlpRatio: Float = 4.0) {
        let headDim = hiddenSize / numHeads
        // head_dim**-0.5 is computed in float64, narrowed at the kernel
        self.scale = Float(pow(Double(headDim), -0.5))
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.mlpHiddenDim = Int(Float(hiddenSize) * mlpRatio)
        self.mlpMultFactor = 2

        self._linear1.wrappedValue = Linear(
            hiddenSize, hiddenSize * 3 + mlpHiddenDim * mlpMultFactor, bias: false)
        self._linear2.wrappedValue = Linear(hiddenSize + mlpHiddenDim, hiddenSize, bias: false)
        self._norm.wrappedValue = QKNorm(headDim)
        self._preNorm.wrappedValue = FastLayerNorm(hiddenSize, eps: 1e-6)
        self._mlpAct.wrappedValue = SiLUActivation()
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, _ pe: MLXArray, _ mod: ModulationTriple, safeAttn: Bool = false
    ) -> MLXArray {
        let (modShift, modScale, modGate) = mod
        let xMod = (1 + modScale) * preNorm(x) + modShift
        let parts = split(linear1(xMod), indices: [3 * hiddenSize], axis: -1)
        var qkv = parts[0]
        let mlp = parts[1]
        let (b, l) = (qkv.dim(0), qkv.dim(1))
        qkv = qkv.reshaped(b, l, 3, numHeads, -1).transposed(2, 0, 3, 1, 4)
        let (qn, kn) = norm(qkv[0], qkv[1], qkv[2])
        let attnOut = attention(qn, kn, qkv[2], pe, scale, safeAttn: safeAttn)
        let out = linear2(concatenated([attnOut, mlpAct(mlp)], axis: -1))
        return x + modGate * out
    }
}

// Double-stream transformer block processing image and text streams jointly.
public final class DoubleStreamBlock: Module {

    public let hiddenSize: Int
    public let numHeads: Int
    public let mlpMultFactor: Int
    public let scale: Float

    @ModuleInfo(key: "img_norm1") public var imgNorm1: FastLayerNorm
    @ModuleInfo(key: "img_attn") public var imgAttn: SelfAttention
    @ModuleInfo(key: "img_norm2") public var imgNorm2: FastLayerNorm
    // Plain list attribute; slot 1 is the parameter-free
    // activation, so weight keys are img_mlp.0 / img_mlp.2 (unflattened fills slot 1 with .none)
    @ModuleInfo(key: "img_mlp") public var imgMlp: [UnaryLayer]
    @ModuleInfo(key: "txt_norm1") public var txtNorm1: FastLayerNorm
    @ModuleInfo(key: "txt_attn") public var txtAttn: SelfAttention
    @ModuleInfo(key: "txt_norm2") public var txtNorm2: FastLayerNorm
    @ModuleInfo(key: "txt_mlp") public var txtMlp: [UnaryLayer]

    public init(hiddenSize: Int, numHeads: Int, mlpRatio: Float) {
        let mlpHiddenDim = Int(Float(hiddenSize) * mlpRatio)
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.mlpMultFactor = 2
        let headDim = hiddenSize / numHeads
        self.scale = Float(pow(Double(headDim), -0.5))

        self._imgNorm1.wrappedValue = FastLayerNorm(hiddenSize, eps: 1e-6)
        self._imgAttn.wrappedValue = SelfAttention(hiddenSize, numHeads: numHeads)
        self._imgNorm2.wrappedValue = FastLayerNorm(hiddenSize, eps: 1e-6)
        self._imgMlp.wrappedValue = [
            Linear(hiddenSize, mlpHiddenDim * 2, bias: false),
            SiLUActivation(),
            Linear(mlpHiddenDim, hiddenSize, bias: false),
        ]
        self._txtNorm1.wrappedValue = FastLayerNorm(hiddenSize, eps: 1e-6)
        self._txtAttn.wrappedValue = SelfAttention(hiddenSize, numHeads: numHeads)
        self._txtNorm2.wrappedValue = FastLayerNorm(hiddenSize, eps: 1e-6)
        self._txtMlp.wrappedValue = [
            Linear(hiddenSize, mlpHiddenDim * 2, bias: false),
            SiLUActivation(),
            Linear(mlpHiddenDim, hiddenSize, bias: false),
        ]
        super.init()
    }

    public func callAsFunction(
        _ img: MLXArray, _ txt: MLXArray, _ pe: MLXArray, _ peCtx: MLXArray,
        _ modImg: (ModulationTriple, ModulationTriple?),
        _ modTxt: (ModulationTriple, ModulationTriple?),
        safeAttn: Bool = false
    ) -> (MLXArray, MLXArray) {
        guard let imgMod2 = modImg.1, let txtMod2 = modTxt.1 else {
            // Unreachable: the transformer constructs these Modulations with double: true.
            fatalError("Flux2 DoubleStreamBlock requires double modulation")
        }
        let (imgMod1Shift, imgMod1Scale, imgMod1Gate) = modImg.0
        let (imgMod2Shift, imgMod2Scale, imgMod2Gate) = imgMod2
        let (txtMod1Shift, txtMod1Scale, txtMod1Gate) = modTxt.0
        let (txtMod2Shift, txtMod2Scale, txtMod2Gate) = txtMod2

        var img = img
        var txt = txt

        let imgModulated = (1 + imgMod1Scale) * imgNorm1(img) + imgMod1Shift
        var imgQkv = imgAttn.qkv(imgModulated)
        let (bi, li) = (imgQkv.dim(0), imgQkv.dim(1))
        imgQkv = imgQkv.reshaped(bi, li, 3, numHeads, -1).transposed(2, 0, 3, 1, 4)
        let imgV = imgQkv[2]
        let (imgQ, imgK) = imgAttn.norm(imgQkv[0], imgQkv[1], imgV)

        let txtModulated = (1 + txtMod1Scale) * txtNorm1(txt) + txtMod1Shift
        var txtQkv = txtAttn.qkv(txtModulated)
        let (bt, lt) = (txtQkv.dim(0), txtQkv.dim(1))
        txtQkv = txtQkv.reshaped(bt, lt, 3, numHeads, -1).transposed(2, 0, 3, 1, 4)
        let txtV = txtQkv[2]
        let (txtQ, txtK) = txtAttn.norm(txtQkv[0], txtQkv[1], txtV)

        let q = concatenated([txtQ, imgQ], axis: 2)
        let k = concatenated([txtK, imgK], axis: 2)
        let v = concatenated([txtV, imgV], axis: 2)
        let peAll = concatenated([peCtx, pe], axis: 2)

        let attnOut = attention(q, k, v, peAll, scale, safeAttn: safeAttn)
        let numTxt = txtQ.dim(2)
        let txtAttnOut = attnOut[0..., ..<numTxt]
        let imgAttnOut = attnOut[0..., numTxt...]

        img = img + imgMod1Gate * imgAttn.proj(imgAttnOut)
        let imgMlpIn = (1 + imgMod2Scale) * imgNorm2(img) + imgMod2Shift
        img = img + imgMod2Gate * imgMlp[2](imgMlp[1](imgMlp[0](imgMlpIn)))

        txt = txt + txtMod1Gate * txtAttn.proj(txtAttnOut)
        let txtMlpIn = (1 + txtMod2Scale) * txtNorm2(txt) + txtMod2Shift
        txt = txt + txtMod2Gate * txtMlp[2](txtMlp[1](txtMlp[0](txtMlpIn)))
        return (img, txt)
    }
}

// Final normalization and projection layer with adaptive modulation.
public final class LastLayer: Module {

    @ModuleInfo(key: "norm_final") public var normFinal: FastLayerNorm
    @ModuleInfo(key: "linear") public var linear: Linear
    // Slot 0 is parameter-free SiLU, weight key adaLN_modulation.1
    @ModuleInfo(key: "adaLN_modulation") public var adaLNModulation: [UnaryLayer]

    public init(hiddenSize: Int, outChannels: Int) {
        self._normFinal.wrappedValue = FastLayerNorm(hiddenSize, eps: 1e-6)
        self._linear.wrappedValue = Linear(hiddenSize, outChannels, bias: false)
        self._adaLNModulation.wrappedValue = [
            SiLU(),
            Linear(hiddenSize, 2 * hiddenSize, bias: false),
        ]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ vec: MLXArray) -> MLXArray {
        let mod = adaLNModulation[1](adaLNModulation[0](vec))
        let chunks = split(mod, parts: 2, axis: -1)
        var shift = chunks[0]
        var scale = chunks[1]
        if shift.ndim == 2 {
            shift = shift[0..., .newAxis, 0...]
            scale = scale[0..., .newAxis, 0...]
        }
        let out = (1 + scale) * normFinal(x) + shift
        return linear(out)
    }
}

// The Flux2 transformer (renamed per port convention).
public final class Flux2Transformer: Module {

    public let inChannels: Int
    public let outChannels: Int
    public let hiddenSize: Int
    public let numHeads: Int
    public let useGuidanceEmbed: Bool
    public var safeAttn: Bool = false

    @ModuleInfo(key: "pe_embedder") public var peEmbedder: EmbedND
    @ModuleInfo(key: "img_in") public var imgIn: Linear
    @ModuleInfo(key: "time_in") public var timeIn: MLPEmbedder
    @ModuleInfo(key: "txt_in") public var txtIn: Linear
    @ModuleInfo(key: "guidance_in") public var guidanceIn: MLPEmbedder?
    @ModuleInfo(key: "double_blocks") public var doubleBlocks: [DoubleStreamBlock]
    @ModuleInfo(key: "single_blocks") public var singleBlocks: [SingleStreamBlock]
    @ModuleInfo(key: "double_stream_modulation_img") public var doubleStreamModulationImg: Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") public var doubleStreamModulationTxt: Modulation
    @ModuleInfo(key: "single_stream_modulation") public var singleStreamModulation: Modulation
    @ModuleInfo(key: "final_layer") public var finalLayer: LastLayer

    public init(params: Flux2Config) throws {
        guard params.hiddenSize % params.numHeads == 0 else {
            throw Flux2Error.configMissing("hidden_size must be divisible by num_heads")
        }
        let peDim = params.hiddenSize / params.numHeads
        guard params.axesDim.reduce(0, +) == peDim else {
            throw Flux2Error.configMissing("axes_dim must sum to head_dim")
        }

        self.inChannels = params.inChannels
        self.outChannels = params.inChannels
        self.hiddenSize = params.hiddenSize
        self.numHeads = params.numHeads
        self.useGuidanceEmbed = params.useGuidanceEmbed

        self._peEmbedder.wrappedValue = EmbedND(
            dim: peDim, theta: params.theta, axesDim: params.axesDim)
        self._imgIn.wrappedValue = Linear(params.inChannels, params.hiddenSize, bias: false)
        self._timeIn.wrappedValue = MLPEmbedder(
            inDim: 256, hiddenDim: params.hiddenSize, disableBias: true)
        self._txtIn.wrappedValue = Linear(params.contextInDim, params.hiddenSize, bias: false)
        self._guidanceIn.wrappedValue =
            params.useGuidanceEmbed
            ? MLPEmbedder(inDim: 256, hiddenDim: params.hiddenSize, disableBias: true)
            : nil
        self._doubleBlocks.wrappedValue = (0 ..< params.depth).map { _ in
            DoubleStreamBlock(
                hiddenSize: params.hiddenSize, numHeads: params.numHeads, mlpRatio: params.mlpRatio)
        }
        self._singleBlocks.wrappedValue = (0 ..< params.depthSingleBlocks).map { _ in
            SingleStreamBlock(
                hiddenSize: params.hiddenSize, numHeads: params.numHeads, mlpRatio: params.mlpRatio)
        }
        self._doubleStreamModulationImg.wrappedValue = Modulation(
            params.hiddenSize, double: true, disableBias: true)
        self._doubleStreamModulationTxt.wrappedValue = Modulation(
            params.hiddenSize, double: true, disableBias: true)
        self._singleStreamModulation.wrappedValue = Modulation(
            params.hiddenSize, double: false, disableBias: true)
        self._finalLayer.wrappedValue = LastLayer(
            hiddenSize: params.hiddenSize, outChannels: params.inChannels)
        super.init()
    }

    /// Pre-embed text context. Call once and reuse across denoising steps.
    public func embedTxt(_ ctx: MLXArray) -> MLXArray {
        txtIn(ctx)
    }

    /// Pre-embed guidance. Call once and reuse across denoising steps.
    public func embedGuidance(_ guidance: MLXArray) -> MLXArray {
        guard let guidanceIn else {
            // Unreachable: only called on models constructed with useGuidanceEmbed == true.
            fatalError("Flux2Transformer.embedGuidance called without guidance_in")
        }
        let guidanceEmb = timestepEmbedding(guidance, 256)
        return guidanceIn(guidanceEmb)
    }

    public func callAsFunction(
        _ x: MLXArray,
        _ xIds: MLXArray,
        _ timesteps: MLXArray,
        _ ctx: MLXArray,
        _ ctxIds: MLXArray,
        _ guidance: MLXArray?,
        _ peX: MLXArray? = nil,
        _ peCtx: MLXArray? = nil,
        _ txtEmbedded: MLXArray? = nil,
        _ guidanceEmbedded: MLXArray? = nil
    ) -> MLXArray {
        let numTxtTokens = ctx.dim(1)
        let timestepEmb = timestepEmbedding(timesteps, 256)
        var vec = timeIn(timestepEmb)
        if useGuidanceEmbed, let guidance {
            if let guidanceEmbedded {
                vec = vec + guidanceEmbedded
            } else if let guidanceIn {
                let guidanceEmb = timestepEmbedding(guidance, 256)
                vec = vec + guidanceIn(guidanceEmb)
            }
        }

        let doubleBlockModImg = doubleStreamModulationImg(vec)
        let doubleBlockModTxt = doubleStreamModulationTxt(vec)
        let (singleBlockMod, _) = singleStreamModulation(vec)

        var img = imgIn(x)
        var txt = txtEmbedded ?? txtIn(ctx)

        let peXResolved = peX ?? peEmbedder(xIds)
        let peCtxResolved = peCtx ?? peEmbedder(ctxIds)

        for block in doubleBlocks {
            (img, txt) = block(
                img, txt, peXResolved, peCtxResolved, doubleBlockModImg, doubleBlockModTxt,
                safeAttn: safeAttn)
        }

        img = concatenated([txt, img], axis: 1)
        let pe = concatenated([peCtxResolved, peXResolved], axis: 2)

        for block in singleBlocks {
            img = block(img, pe, singleBlockMod, safeAttn: safeAttn)
        }

        img = img[0..., numTxtTokens..., 0...]
        img = finalLayer(img, vec)
        return img
    }
}
