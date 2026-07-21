// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// 2026-07-19 EDT | PERMANENT (Flux2Kit t2i port) — numerical parity with the reference implementation is the contract; do not refactor without re-running the parity harness.

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

public func swish(_ x: MLXArray) -> MLXArray {
    silu(x)
}

public final class ResnetBlock: Module {

    public let useShortcut: Bool

    @ModuleInfo(key: "norm1") public var norm1: GroupNorm
    @ModuleInfo(key: "conv1") public var conv1: Conv2d
    @ModuleInfo(key: "norm2") public var norm2: GroupNorm
    @ModuleInfo(key: "conv2") public var conv2: Conv2d
    @ModuleInfo(key: "nin_shortcut") public var ninShortcut: Conv2d?

    public init(inChannels: Int, outChannels: Int?, normGroups: Int) {
        let outChannels = outChannels ?? inChannels
        self.useShortcut = inChannels != outChannels

        self._norm1.wrappedValue = GroupNorm(
            groupCount: normGroups, dimensions: inChannels, eps: 1e-6, affine: true,
            pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1,
            padding: 1)
        self._norm2.wrappedValue = GroupNorm(
            groupCount: normGroups, dimensions: outChannels, eps: 1e-6, affine: true,
            pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1,
            padding: 1)
        // Shortcut conv only when channel counts differ
        self._ninShortcut.wrappedValue =
            useShortcut
            ? Conv2d(
                inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, stride: 1,
                padding: 0)
            : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        var h = norm1(x)
        h = swish(h)
        h = conv1(h)
        h = norm2(h)
        h = swish(h)
        h = conv2(h)
        if useShortcut {
            guard let ninShortcut = ninShortcut else {
                // Unreachable by construction (init creates the conv whenever useShortcut is true).
                fatalError("Flux2VAE.ResnetBlock: nin_shortcut missing despite use_shortcut")
            }
            x = ninShortcut(x)
        }
        return x + h
    }
}

public final class AttnBlock: Module {

    public let inChannels: Int
    /// Cache attention scale (`in_channels ** -0.5`,
    /// computed in double precision then narrowed to float32 at the kernel boundary).
    public let scale: Float

    @ModuleInfo(key: "norm") public var norm: GroupNorm
    @ModuleInfo(key: "q") public var q: Conv2d
    @ModuleInfo(key: "k") public var k: Conv2d
    @ModuleInfo(key: "v") public var v: Conv2d
    @ModuleInfo(key: "proj_out") public var projOut: Conv2d

    public init(inChannels: Int) {
        self.inChannels = inChannels
        self.scale = Float(pow(Double(inChannels), -0.5))
        // Group count is hardcoded 32 here (not norm_groups)
        self._norm.wrappedValue = GroupNorm(
            groupCount: 32, dimensions: inChannels, eps: 1e-6, affine: true,
            pytorchCompatible: true)
        self._q.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: inChannels, kernelSize: 1)
        self._k.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: inChannels, kernelSize: 1)
        self._v.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: inChannels, kernelSize: 1)
        self._projOut.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: inChannels, kernelSize: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm(x)
        var q = self.q(h)
        var k = self.k(h)
        var v = self.v(h)
        let b = q.dim(0)
        let hgt = q.dim(1)
        let wdt = q.dim(2)
        let c = q.dim(3)
        // Flatten spatial dims, insert single-head axis
        q = q.reshaped(b, hgt * wdt, c)[0..., .newAxis, 0..., 0...]
        k = k.reshaped(b, hgt * wdt, c)[0..., .newAxis, 0..., 0...]
        v = v.reshaped(b, hgt * wdt, c)[0..., .newAxis, 0..., 0...]
        var attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
        attn = attn.reshaped(b, hgt, wdt, c)
        return x + projOut(attn)
    }
}

public final class Downsample: Module {

    @ModuleInfo(key: "conv") public var conv: Conv2d

    public init(inChannels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: inChannels, kernelSize: 3, stride: 2,
            padding: 0)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Asymmetric (0,1) constant pad on H and W
        let widths: [IntOrPair] = [[0, 0], [0, 1], [0, 1], [0, 0]]
        let x = padded(x, widths: widths, mode: .constant)
        return conv(x)
    }
}

public final class Upsample: Module {

    @ModuleInfo(key: "conv") public var conv: Conv2d

    public init(inChannels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: inChannels, kernelSize: 3, stride: 1,
            padding: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let c = x.dim(3)
        // Nearest-neighbor 2x upsample via broadcast
        var y = x.reshaped(b, h, 1, w, 1, c)
        y = broadcast(y, to: [b, h, 2, w, 2, c])
        y = y.reshaped(b, h * 2, w * 2, c)
        return conv(y)
    }
}

/// Swift materialization of the anonymous `nn.Module()` used for `Encoder.down[i]`.
/// Weight keys: `block.{j}.*`, `downsample.conv.*`.
public final class EncoderDownBlock: Module {

    @ModuleInfo(key: "block") public var block: [ResnetBlock]
    @ModuleInfo(key: "downsample") public var downsample: Downsample?

    public init(block: [ResnetBlock], downsample: Downsample?) {
        self._block.wrappedValue = block
        self._downsample.wrappedValue = downsample
        super.init()
    }
}

/// Swift materialization of the anonymous `nn.Module()` used for `Decoder.up[i]`.
/// Weight keys: `block.{j}.*`, `upsample.conv.*`.
public final class DecoderUpBlock: Module {

    @ModuleInfo(key: "block") public var block: [ResnetBlock]
    @ModuleInfo(key: "upsample") public var upsample: Upsample?

    public init(block: [ResnetBlock], upsample: Upsample?) {
        self._block.wrappedValue = block
        self._upsample.wrappedValue = upsample
        super.init()
    }
}

/// Swift materialization of the anonymous `nn.Module()` used for `Encoder.mid` / `Decoder.mid`.
/// Weight keys: `block_1.*`, `attn_1.*`, `block_2.*`.
public final class VAEMidBlock: Module {

    @ModuleInfo(key: "block_1") public var block1: ResnetBlock
    @ModuleInfo(key: "attn_1") public var attn1: AttnBlock
    @ModuleInfo(key: "block_2") public var block2: ResnetBlock

    public init(block1: ResnetBlock, attn1: AttnBlock, block2: ResnetBlock) {
        self._block1.wrappedValue = block1
        self._attn1.wrappedValue = attn1
        self._block2.wrappedValue = block2
        super.init()
    }
}

public final class Encoder: Module {

    // quant_conv lives on the Encoder
    // (weight key "encoder.quant_conv.*")
    @ModuleInfo(key: "quant_conv") public var quantConv: Conv2d

    public let ch: Int
    public let numResolutions: Int
    public let numResBlocks: Int
    public let resolution: Int
    public let inChannels: Int

    @ModuleInfo(key: "conv_in") public var convIn: Conv2d
    @ModuleInfo(key: "down") public var down: [EncoderDownBlock]
    @ModuleInfo(key: "mid") public var mid: VAEMidBlock
    @ModuleInfo(key: "norm_out") public var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") public var convOut: Conv2d

    public init(
        resolution: Int,
        inChannels: Int,
        ch: Int,
        chMult: [Int],
        numResBlocks: Int,
        zChannels: Int,
        normGroups: Int
    ) {
        self._quantConv.wrappedValue = Conv2d(
            inputChannels: 2 * zChannels, outputChannels: 2 * zChannels, kernelSize: 1)
        self.ch = ch
        let numResolutions = chMult.count
        self.numResolutions = numResolutions
        self.numResBlocks = numResBlocks
        self.resolution = resolution
        self.inChannels = inChannels

        self._convIn.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: ch, kernelSize: 3, stride: 1, padding: 1)

        // in_ch_mult = (1,) + tuple(ch_mult)
        let inChMult = [1] + chMult
        var down = [EncoderDownBlock]()
        var blockIn = ch
        var currRes = resolution
        for iLevel in 0 ..< numResolutions {
            var block = [ResnetBlock]()
            blockIn = ch * inChMult[iLevel]
            let blockOut = ch * chMult[iLevel]
            for _ in 0 ..< numResBlocks {
                block.append(
                    ResnetBlock(inChannels: blockIn, outChannels: blockOut, normGroups: normGroups))
                blockIn = blockOut
            }
            var downsample: Downsample? = nil
            if iLevel != numResolutions - 1 {
                downsample = Downsample(inChannels: blockIn)
                currRes = currRes / 2
            }
            down.append(EncoderDownBlock(block: block, downsample: downsample))
        }
        _ = currRes  // Tracks curr_res but never reads it
        self._down.wrappedValue = down

        self._mid.wrappedValue = VAEMidBlock(
            block1: ResnetBlock(inChannels: blockIn, outChannels: blockIn, normGroups: normGroups),
            attn1: AttnBlock(inChannels: blockIn),
            block2: ResnetBlock(inChannels: blockIn, outChannels: blockIn, normGroups: normGroups))

        self._normOut.wrappedValue = GroupNorm(
            groupCount: normGroups, dimensions: blockIn, eps: 1e-6, affine: true,
            pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: blockIn, outputChannels: 2 * zChannels, kernelSize: 3, stride: 1,
            padding: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for iLevel in 0 ..< numResolutions {
            for iBlock in 0 ..< numResBlocks {
                h = down[iLevel].block[iBlock](h)
            }
            if iLevel != numResolutions - 1 {
                guard let downsample = down[iLevel].downsample else {
                    // Unreachable by construction (init creates a downsample for every non-final level).
                    fatalError("Flux2VAE.Encoder: missing downsample at level \(iLevel)")
                }
                h = downsample(h)
            }
        }

        h = mid.block1(h)
        h = mid.attn1(h)
        h = mid.block2(h)
        h = normOut(h)
        h = swish(h)
        h = convOut(h)
        h = quantConv(h)
        return h
    }
}

public final class Decoder: Module {

    // post_quant_conv lives on the Decoder
    // (weight key "decoder.post_quant_conv.*")
    @ModuleInfo(key: "post_quant_conv") public var postQuantConv: Conv2d

    public let ch: Int
    public let numResolutions: Int
    public let numResBlocks: Int
    public let resolution: Int
    public let inChannels: Int
    public let ffactor: Int

    @ModuleInfo(key: "conv_in") public var convIn: Conv2d
    @ModuleInfo(key: "mid") public var mid: VAEMidBlock
    @ModuleInfo(key: "up") public var up: [DecoderUpBlock]
    @ModuleInfo(key: "norm_out") public var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") public var convOut: Conv2d

    public init(
        ch: Int,
        outCh: Int,
        chMult: [Int],
        numResBlocks: Int,
        inChannels: Int,
        resolution: Int,
        zChannels: Int,
        normGroups: Int
    ) {
        self._postQuantConv.wrappedValue = Conv2d(
            inputChannels: zChannels, outputChannels: zChannels, kernelSize: 1)
        self.ch = ch
        let numResolutions = chMult.count
        self.numResolutions = numResolutions
        self.numResBlocks = numResBlocks
        self.resolution = resolution
        self.inChannels = inChannels
        self.ffactor = 1 << (numResolutions - 1)  // 2 ** (num_resolutions - 1)

        var blockIn = ch * chMult[numResolutions - 1]

        self._convIn.wrappedValue = Conv2d(
            inputChannels: zChannels, outputChannels: blockIn, kernelSize: 3, stride: 1, padding: 1)

        self._mid.wrappedValue = VAEMidBlock(
            block1: ResnetBlock(inChannels: blockIn, outChannels: blockIn, normGroups: normGroups),
            attn1: AttnBlock(inChannels: blockIn),
            block2: ResnetBlock(inChannels: blockIn, outChannels: blockIn, normGroups: normGroups))

        var up = [DecoderUpBlock]()
        for iLevel in stride(from: numResolutions - 1, through: 0, by: -1) {
            var block = [ResnetBlock]()
            let blockOut = ch * chMult[iLevel]
            for _ in 0 ..< (numResBlocks + 1) {
                block.append(
                    ResnetBlock(inChannels: blockIn, outChannels: blockOut, normGroups: normGroups))
                blockIn = blockOut
            }
            var upsample: Upsample? = nil
            if iLevel != 0 {
                upsample = Upsample(inChannels: blockIn)
            }
            up.append(DecoderUpBlock(block: block, upsample: upsample))
        }
        // Reverse to match weight indexing:
        // up[0]=finest, up[n-1]=coarsest
        self._up.wrappedValue = Array(up.reversed())

        self._normOut.wrappedValue = GroupNorm(
            groupCount: normGroups, dimensions: blockIn, eps: 1e-6, affine: true,
            pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: blockIn, outputChannels: outCh, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        let z = postQuantConv(z)
        var h = convIn(z)
        h = mid.block1(h)
        h = mid.attn1(h)
        h = mid.block2(h)

        // Iterate coarsest (up[n-1]) to finest (up[0])
        for iLevel in stride(from: numResolutions - 1, through: 0, by: -1) {
            for iBlock in 0 ..< (numResBlocks + 1) {
                h = up[iLevel].block[iBlock](h)
            }
            if iLevel != 0 {
                guard let upsample = up[iLevel].upsample else {
                    // Unreachable by construction (init creates an upsample for every level except 0).
                    fatalError("Flux2VAE.Decoder: missing upsample at level \(iLevel)")
                }
                h = upsample(h)
            }
        }

        h = normOut(h)
        h = swish(h)
        h = convOut(h)
        return h
    }
}

public final class AutoEncoder: Module {

    public let params: VAEConfig
    /// `var` (not `let`) because the pipeline mutates `vae.force_upcast` for the fp16 path.
    public var forceUpcast: Bool

    @ModuleInfo(key: "encoder") public var encoder: Encoder
    @ModuleInfo(key: "decoder") public var decoder: Decoder

    public let bnEps: Float
    public let bnMomentum: Float
    public let ps: (Int, Int)

    @ModuleInfo(key: "bn") public var bn: BatchNorm

    // Leading underscore keeps these lazy caches out of the MLXNN parameter tree
    // (Module.parameterIsValid filters "_"-prefixed keys), the
    // _inv_norm_scale / _inv_norm_mean exclusion.
    private var _invNormScale: MLXArray? = nil
    private var _invNormMean: MLXArray? = nil

    public init(params: VAEConfig) {
        self.params = params
        self.forceUpcast = params.forceUpcast
        self._encoder.wrappedValue = Encoder(
            resolution: params.resolution,
            inChannels: params.inChannels,
            ch: params.ch,
            chMult: params.chMult,
            numResBlocks: params.numResBlocks,
            zChannels: params.zChannels,
            normGroups: params.normNumGroups)
        // Swift requires keyword arguments here; order follows Decoder's
        // declaration order, semantics identical
        self._decoder.wrappedValue = Decoder(
            ch: params.ch,
            outCh: params.outCh,
            chMult: params.chMult,
            numResBlocks: params.numResBlocks,
            inChannels: params.inChannels,
            resolution: params.resolution,
            zChannels: params.zChannels,
            normGroups: params.normNumGroups)
        self.bnEps = Float(params.bnEps)
        self.bnMomentum = Float(params.bnMomentum)
        self.ps = params.ps
        // featureCount = math.prod(ps) * z_channels
        self._bn.wrappedValue = BatchNorm(
            featureCount: params.ps.0 * params.ps.1 * params.zChannels,
            eps: Float(params.bnEps),
            momentum: Float(params.bnMomentum),
            affine: false,
            trackRunningStats: true)
        super.init()
        bn.train(false)  // VAE is inference-only
    }

    public func normalize(_ z: MLXArray) -> MLXArray {
        bn(z)
    }

    // Running stats are read lazily on first call
    // and cached. Read through parameters() because MLXNN.BatchNorm's runningMean/runningVar
    // properties are internal to MLXNN.
    public func invNormalize(_ z: MLXArray) -> MLXArray {
        if _invNormScale == nil || _invNormMean == nil {
            let bnParameters = bn.parameters()
            guard let runningVar = bnParameters[unwrapping: "running_var"],
                let runningMean = bnParameters[unwrapping: "running_mean"]
            else {
                // Unreachable by construction (init passes trackRunningStats: true).
                fatalError("Flux2VAE.AutoEncoder: BatchNorm running stats are missing")
            }
            _invNormScale = sqrt(runningVar.reshaped(1, 1, 1, -1) + bnEps)
            _invNormMean = runningMean.reshaped(1, 1, 1, -1)
        }
        guard let scale = _invNormScale, let mean = _invNormMean else {
            fatalError("Flux2VAE.AutoEncoder: inverse-normalization cache unavailable")
        }
        return z * scale + mean
    }

    // Deterministic encode: takes the mean half of
    // the moments, patchifies (pi, pj) into channels, then batch-normalizes
    public func encode(_ x: MLXArray) -> MLXArray {
        var x = x
        let origDtype = x.dtype
        if forceUpcast && x.dtype != .float32 {
            x = x.asType(.float32)
        }
        let moments = encoder(x)
        var mean = moments[.ellipsis, 0 ..< params.zChannels]
        let b = mean.dim(0)
        let h = mean.dim(1)
        let w = mean.dim(2)
        let c = mean.dim(3)
        let (pi, pj) = ps
        // Per-channel patchify transpose order
        mean = mean.reshaped(b, h / pi, pi, w / pj, pj, c)
        mean = mean.transposed(0, 1, 3, 5, 2, 4)
        var z = mean.reshaped(b, h / pi, w / pj, c * pi * pj)
        z = normalize(z)
        if forceUpcast && z.dtype != origDtype {
            z = z.asType(origDtype)
        }
        return z
    }

    // Inverse-normalize then un-patchify channels
    // back into (pi, pj) spatial positions
    public func decode(_ z: MLXArray) -> MLXArray {
        var z = z
        let origDtype = z.dtype
        if forceUpcast && z.dtype != .float32 {
            z = z.asType(.float32)
        }
        z = invNormalize(z)
        let b = z.dim(0)
        let h = z.dim(1)
        let w = z.dim(2)
        let cp = z.dim(3)
        let (pi, pj) = ps
        let c = cp / (pi * pj)
        // Un-patchify transpose order
        z = z.reshaped(b, h, w, c, pi, pj)
        z = z.transposed(0, 1, 4, 2, 5, 3)
        z = z.reshaped(b, h * pi, w * pj, c)
        var out = decoder(z)
        if forceUpcast && out.dtype != origDtype {
            out = out.asType(origDtype)
        }
        return out
    }
}
