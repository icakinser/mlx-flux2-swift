// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// 2026-07-19 EDT | PERMANENT (Flux2Kit t2i port) — numerical parity with the reference implementation is the contract; do not refactor without re-running the parity harness.

import Foundation
import MLX
import MLXNN

// MARK: - Errors

/// Package-wide error type for Flux2Kit.
/// Declared here (config/weights module) per the porting contract; sibling modules throw these cases.
public enum Flux2Error: Error, LocalizedError {
    case loadFailed(String)
    case configMissing(String)
    case tokenizerFailed(String)
    case generationFailed(String)
    case duplicateWeightKeys(String)
    case missingWeights(String)

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let message): return "Load failed: \(message)"
        case .configMissing(let message): return "Config missing: \(message)"
        case .tokenizerFailed(let message): return "Tokenizer failed: \(message)"
        case .generationFailed(let message): return "Generation failed: \(message)"
        case .duplicateWeightKeys(let message): return "Duplicate weight keys: \(message)"
        case .missingWeights(let message): return "Missing weights: \(message)"
        }
    }
}

// MARK: - Config structs

/// Configuration for the Flux2 transformer.
public struct Flux2Config: Sendable {
    public var inChannels: Int
    public var contextInDim: Int
    public var hiddenSize: Int
    public var numHeads: Int
    public var depth: Int
    public var depthSingleBlocks: Int
    public var axesDim: [Int]
    public var theta: Float
    public var mlpRatio: Float
    public var useGuidanceEmbed: Bool

    public init(
        inChannels: Int,
        contextInDim: Int,
        hiddenSize: Int,
        numHeads: Int,
        depth: Int,
        depthSingleBlocks: Int,
        axesDim: [Int],
        theta: Float,
        mlpRatio: Float,
        useGuidanceEmbed: Bool
    ) {
        self.inChannels = inChannels
        self.contextInDim = contextInDim
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.depth = depth
        self.depthSingleBlocks = depthSingleBlocks
        self.axesDim = axesDim
        self.theta = theta
        self.mlpRatio = mlpRatio
        self.useGuidanceEmbed = useGuidanceEmbed
    }
}

/// Configuration for the VAE.
public struct VAEConfig: Sendable {
    public var resolution: Int
    public var inChannels: Int
    public var ch: Int
    public var outCh: Int
    public var chMult: [Int]
    public var numResBlocks: Int
    public var zChannels: Int
    public var normNumGroups: Int
    public var bnEps: Float
    public var bnMomentum: Float
    public var ps: (Int, Int)
    public var forceUpcast: Bool

    public init(
        resolution: Int,
        inChannels: Int,
        ch: Int,
        outCh: Int,
        chMult: [Int],
        numResBlocks: Int,
        zChannels: Int,
        normNumGroups: Int,
        bnEps: Float,
        bnMomentum: Float,
        ps: (Int, Int),
        forceUpcast: Bool
    ) {
        self.resolution = resolution
        self.inChannels = inChannels
        self.ch = ch
        self.outCh = outCh
        self.chMult = chMult
        self.numResBlocks = numResBlocks
        self.zChannels = zChannels
        self.normNumGroups = normNumGroups
        self.bnEps = bnEps
        self.bnMomentum = bnMomentum
        self.ps = ps
        self.forceUpcast = forceUpcast
    }
}

/// Configuration for the Qwen3 text encoder.
public struct Qwen3Config: Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool
    // Optional dictionary; only numeric entries are retained.
    // FLUX.2-klein-4B ships "rope_scaling": null and it is never read.
    public var ropeScaling: [String: Double]?

    public init(
        modelType: String,
        hiddenSize: Int,
        numHiddenLayers: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        vocabSize: Int,
        maxPositionEmbeddings: Int,
        ropeTheta: Float,
        tieWordEmbeddings: Bool,
        ropeScaling: [String: Double]?
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeScaling = ropeScaling
    }
}

// MARK: - JSON access helpers (int()/float()/bool() casts over the decoded JSON object)

private func loadJSONObject(_ path: URL) throws -> [String: Any] {
    let data: Data
    do {
        data = try Data(contentsOf: path)
    } catch {
        throw Flux2Error.loadFailed("Could not read config at \(path.path): \(error.localizedDescription)")
    }
    let parsed: Any
    do {
        parsed = try JSONSerialization.jsonObject(with: data)
    } catch {
        throw Flux2Error.loadFailed("Could not parse JSON at \(path.path): \(error.localizedDescription)")
    }
    guard let object = parsed as? [String: Any] else {
        throw Flux2Error.loadFailed("Config is not a JSON object: \(path.path)")
    }
    return object
}

private func intValue(_ data: [String: Any], _ key: String, _ path: URL) throws -> Int {
    // Int cast; a missing key becomes Flux2Error.configMissing
    guard let number = data[key] as? NSNumber else {
        throw Flux2Error.configMissing("\(key) in \(path.lastPathComponent)")
    }
    return number.intValue
}

private func floatValue(_ data: [String: Any], _ key: String, _ path: URL) throws -> Float {
    // Float cast
    guard let number = data[key] as? NSNumber else {
        throw Flux2Error.configMissing("\(key) in \(path.lastPathComponent)")
    }
    return number.floatValue
}

private func intArrayValue(_ data: [String: Any], _ key: String, _ path: URL) throws -> [Int] {
    // Int-array cast over the JSON list
    guard let raw = data[key] as? [Any] else {
        throw Flux2Error.configMissing("\(key) in \(path.lastPathComponent)")
    }
    var result: [Int] = []
    result.reserveCapacity(raw.count)
    for element in raw {
        guard let number = element as? NSNumber else {
            throw Flux2Error.configMissing("\(key) element in \(path.lastPathComponent)")
        }
        result.append(number.intValue)
    }
    return result
}

private func boolValue(_ data: [String: Any], _ key: String, default defaultValue: Bool) -> Bool {
    // Bool cast with a default
    guard let number = data[key] as? NSNumber else { return defaultValue }
    return number.boolValue
}

// MARK: - Config loaders

/// Loads the Flux2 transformer configuration from a JSON config file.
public func loadFlux2Config(_ path: URL) throws -> Flux2Config {
    let data = try loadJSONObject(path)
    let numHeads = try intValue(data, "num_attention_heads", path)
    let attentionHeadDim = try intValue(data, "attention_head_dim", path)
    return Flux2Config(
        inChannels: try intValue(data, "in_channels", path),
        contextInDim: try intValue(data, "joint_attention_dim", path),
        // hidden_size = int(num_attention_heads * attention_head_dim)
        hiddenSize: numHeads * attentionHeadDim,
        numHeads: numHeads,
        depth: try intValue(data, "num_layers", path),
        depthSingleBlocks: try intValue(data, "num_single_layers", path),
        axesDim: try intArrayValue(data, "axes_dims_rope", path),
        theta: try floatValue(data, "rope_theta", path),
        mlpRatio: try floatValue(data, "mlp_ratio", path),
        useGuidanceEmbed: boolValue(data, "guidance_embeds", default: false)
    )
}

/// Loads the VAE configuration from a JSON config file.
public func loadVaeConfig(_ path: URL) throws -> VAEConfig {
    let data = try loadJSONObject(path)
    let blockOutChannels = try intArrayValue(data, "block_out_channels", path)
    guard let firstBlockOut = blockOutChannels.first, firstBlockOut != 0 else {
        throw Flux2Error.configMissing("block_out_channels in \(path.lastPathComponent)")
    }
    let patchSize = try intArrayValue(data, "patch_size", path)
    guard patchSize.count >= 2 else {
        throw Flux2Error.configMissing("patch_size in \(path.lastPathComponent)")
    }
    return VAEConfig(
        resolution: try intValue(data, "sample_size", path),
        inChannels: try intValue(data, "in_channels", path),
        ch: firstBlockOut,
        outCh: try intValue(data, "out_channels", path),
        // ch_mult = [x // block_out_channels[0] for x in block_out_channels]
        chMult: blockOutChannels.map { $0 / firstBlockOut },
        numResBlocks: try intValue(data, "layers_per_block", path),
        zChannels: try intValue(data, "latent_channels", path),
        normNumGroups: try intValue(data, "norm_num_groups", path),
        bnEps: try floatValue(data, "batch_norm_eps", path),
        bnMomentum: try floatValue(data, "batch_norm_momentum", path),
        ps: (patchSize[0], patchSize[1]),
        forceUpcast: boolValue(data, "force_upcast", default: false)
    )
}

/// Loads the Qwen3 text encoder configuration from a JSON config file.
public func loadQwen3Config(_ path: URL) throws -> Qwen3Config {
    let data = try loadJSONObject(path)
    let hiddenSize = try intValue(data, "hidden_size", path)
    let numAttentionHeads = try intValue(data, "num_attention_heads", path)
    // head_dim = data["head_dim"] if present, else hidden_size // num_attention_heads
    let headDim: Int
    if let number = data["head_dim"] as? NSNumber {
        headDim = number.intValue
    } else {
        headDim = hiddenSize / numAttentionHeads
    }
    guard let modelType = data["model_type"] as? String else {
        throw Flux2Error.configMissing("model_type in \(path.lastPathComponent)")
    }
    // parity: rope_scaling = data.get("rope_scaling") — null maps to nil; numeric entries only
    var ropeScaling: [String: Double]? = nil
    if let rawScaling = data["rope_scaling"] as? [String: Any] {
        var parsed: [String: Double] = [:]
        for (key, value) in rawScaling {
            if let number = value as? NSNumber {
                parsed[key] = number.doubleValue
            }
        }
        ropeScaling = parsed
    }
    return Qwen3Config(
        modelType: modelType,
        hiddenSize: hiddenSize,
        numHiddenLayers: try intValue(data, "num_hidden_layers", path),
        intermediateSize: try intValue(data, "intermediate_size", path),
        numAttentionHeads: numAttentionHeads,
        numKeyValueHeads: try intValue(data, "num_key_value_heads", path),
        headDim: headDim,
        rmsNormEps: try floatValue(data, "rms_norm_eps", path),
        vocabSize: try intValue(data, "vocab_size", path),
        maxPositionEmbeddings: try intValue(data, "max_position_embeddings", path),
        ropeTheta: try floatValue(data, "rope_theta", path),
        tieWordEmbeddings: boolValue(data, "tie_word_embeddings", default: true),
        ropeScaling: ropeScaling
    )
}

// MARK: - Defaults

/// DEFAULT_REPO_ID
public let defaultRepoId = "black-forest-labs/FLUX.2-klein-4B"
/// WEIGHT_FILES (probe order matters)
public let weightFiles: [String] = [
    "flux-2-klein-4b-fp8.safetensors",
    "flux-2-klein-4b.safetensors",
    "flux-2-klein-base-4b.safetensors",
]
/// TOKENIZER_FALLBACK_DIR
public let tokenizerFallbackDir = "FLUX.2-klein-base-4B"
/// TEXT_ENCODER_MAX_LENGTH
public let textEncoderMaxLength = 512
/// TEXT_ENCODER_OUTPUT_LAYERS — layers (9, 18, 27)
public let textEncoderOutputLayers: [Int] = [9, 18, 27]
/// DEFAULT_WIDTH
public let defaultWidth = 512
/// DEFAULT_HEIGHT
public let defaultHeight = 512
/// DEFAULT_STEPS
public let defaultSteps = 4
/// DEFAULT_GUIDANCE
public let defaultGuidance: Float = 1.0
/// DEFAULT_DTYPE
public let defaultDtype = "bfloat16"
/// DEFAULT_QUANTIZE
public let defaultQuantize = "none"
/// DEFAULT_OUTPUT
public let defaultOutput = "output.png"
/// REF_IMAGE_LIMIT_PIXELS_SINGLE = 2048**2
public let refImageLimitPixelsSingle = 2048 * 2048
/// REF_IMAGE_LIMIT_PIXELS_MULTI = 1024**2
public let refImageLimitPixelsMulti = 1024 * 1024
/// REF_TIME_OFFSET_SCALE
public let refTimeOffsetScale = 10
