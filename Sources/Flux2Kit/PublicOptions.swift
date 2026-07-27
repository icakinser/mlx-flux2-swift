import CoreGraphics
import Foundation

/// Pipeline-level policy chosen once when loading weights.
public struct PipelineConfiguration: Sendable {
    public var repoId: String
    public var repoPath: URL?
    public var dtype: String
    public var quantize: String?
    public var safeAttention: Bool
    public var vaeFloat16: Bool
    public var compile: Bool
    public var residency: ResidencyPolicy
    public var cacheLimitMB: Int?
    public var memoryLimitMB: Int?
    public var reportMemory: Bool
    public var vaeTileLatent: Int?

    public init(
        repoId: String = defaultRepoId,
        repoPath: URL? = nil,
        dtype: String = defaultDtype,
        quantize: String? = nil,
        safeAttention: Bool = false,
        vaeFloat16: Bool = false,
        compile: Bool = false,
        residency: ResidencyPolicy = .keepResident,
        cacheLimitMB: Int? = nil,
        memoryLimitMB: Int? = nil,
        reportMemory: Bool = false,
        vaeTileLatent: Int? = nil
    ) {
        self.repoId = repoId
        self.repoPath = repoPath
        self.dtype = dtype
        self.quantize = quantize
        self.safeAttention = safeAttention
        self.vaeFloat16 = vaeFloat16
        self.compile = compile
        self.residency = residency
        self.cacheLimitMB = cacheLimitMB
        self.memoryLimitMB = memoryLimitMB
        self.reportMemory = reportMemory
        self.vaeTileLatent = vaeTileLatent
    }

    public static func lowMemory(repoPath: URL? = nil) -> PipelineConfiguration {
        PipelineConfiguration(
            repoPath: repoPath,
            quantize: "int4",
            vaeFloat16: true,
            residency: .unloadAfterUse,
            cacheLimitMB: 512)
    }
}

/// Settings shared by every denoising mode.
public struct DenoiseOptions: Sendable {
    public var numSteps: Int
    public var guidance: Double
    public var seed: UInt64?
    public var verbose: Bool
    public var evalFrequency: Int
    public var sampler: Sampler
    public var guidanceSchedule: GuidanceSchedule
    public var progress: (@Sendable (GenerationProgress) -> Void)?
    public var cancellation: GenerationCancellation?

    public init(
        numSteps: Int = defaultSteps,
        guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil,
        verbose: Bool = false,
        evalFrequency: Int = 1,
        sampler: Sampler = .euler,
        guidanceSchedule: GuidanceSchedule = .constant,
        progress: (@Sendable (GenerationProgress) -> Void)? = nil,
        cancellation: GenerationCancellation? = nil
    ) {
        self.numSteps = numSteps
        self.guidance = guidance
        self.seed = seed
        self.verbose = verbose
        self.evalFrequency = evalFrequency
        self.sampler = sampler
        self.guidanceSchedule = guidanceSchedule
        self.progress = progress
        self.cancellation = cancellation
    }
}

public struct ImageToImageOptions: Sendable {
    public var prompt: String
    public var strength: Double
    public var width: Int
    public var height: Int
    public var denoise: DenoiseOptions

    public init(
        prompt: String,
        strength: Double = 0.6,
        width: Int = defaultWidth,
        height: Int = defaultHeight,
        denoise: DenoiseOptions = DenoiseOptions()
    ) {
        self.prompt = prompt
        self.strength = strength
        self.width = width
        self.height = height
        self.denoise = denoise
    }
}

public struct InpaintOptions: Sendable {
    public var prompt: String
    public var strength: Double
    public var width: Int
    public var height: Int
    public var invertMask: Bool
    public var maskFeather: Int
    public var denoise: DenoiseOptions

    public init(
        prompt: String,
        strength: Double = 0.85,
        width: Int = defaultWidth,
        height: Int = defaultHeight,
        invertMask: Bool = false,
        maskFeather: Int = 1,
        denoise: DenoiseOptions = DenoiseOptions()
    ) {
        self.prompt = prompt
        self.strength = strength
        self.width = width
        self.height = height
        self.invertMask = invertMask
        self.maskFeather = maskFeather
        self.denoise = denoise
    }
}

public struct OutpaintOptions: Sendable {
    public var prompt: String
    public var left: Int
    public var right: Int
    public var top: Int
    public var bottom: Int
    public var strength: Double
    public var maskFeather: Int
    public var denoise: DenoiseOptions

    public init(
        prompt: String,
        left: Int,
        right: Int,
        top: Int,
        bottom: Int,
        strength: Double = 0.95,
        maskFeather: Int = 2,
        denoise: DenoiseOptions = DenoiseOptions()
    ) {
        self.prompt = prompt
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.strength = strength
        self.maskFeather = maskFeather
        self.denoise = denoise
    }
}

extension Flux2Pipeline {
    public convenience init(configuration: PipelineConfiguration) async throws {
        try await self.init(
            repoId: configuration.repoId,
            repoPath: configuration.repoPath,
            dtype: configuration.dtype,
            quantize: configuration.quantize,
            safeAttn: configuration.safeAttention,
            vaeFp16: configuration.vaeFloat16,
            compile: configuration.compile,
            residency: configuration.residency,
            cacheLimitMB: configuration.cacheLimitMB,
            memoryLimitMB: configuration.memoryLimitMB,
            memReport: configuration.reportMemory)
        self.vaeTileLatent = configuration.vaeTileLatent
    }

    public func generateImg2Img(
        _ options: ImageToImageOptions,
        source: CGImage,
        inputImages: [CGImage]? = nil
    ) throws -> CGImage {
        let denoise = options.denoise
        return try generateImg2Img(
            prompt: options.prompt,
            source: source,
            strength: options.strength,
            width: options.width,
            height: options.height,
            numSteps: denoise.numSteps,
            guidance: denoise.guidance,
            seed: denoise.seed,
            inputImages: inputImages,
            verbose: denoise.verbose,
            evalFreq: denoise.evalFrequency,
            sampler: denoise.sampler,
            guidanceSchedule: denoise.guidanceSchedule,
            progress: denoise.progress,
            cancellation: denoise.cancellation)
    }

    public func generateInpaint(
        _ options: InpaintOptions,
        source: CGImage,
        mask: CGImage,
        inputImages: [CGImage]? = nil
    ) throws -> CGImage {
        let denoise = options.denoise
        return try generateInpaint(
            prompt: options.prompt,
            source: source,
            mask: mask,
            strength: options.strength,
            width: options.width,
            height: options.height,
            numSteps: denoise.numSteps,
            guidance: denoise.guidance,
            seed: denoise.seed,
            inputImages: inputImages,
            invertMask: options.invertMask,
            maskFeather: options.maskFeather,
            verbose: denoise.verbose,
            evalFreq: denoise.evalFrequency,
            sampler: denoise.sampler,
            guidanceSchedule: denoise.guidanceSchedule,
            progress: denoise.progress,
            cancellation: denoise.cancellation)
    }

    public func generateOutpaint(
        _ options: OutpaintOptions,
        source: CGImage
    ) throws -> CGImage {
        let denoise = options.denoise
        return try generateOutpaint(
            source: source,
            prompt: options.prompt,
            left: options.left,
            right: options.right,
            top: options.top,
            bottom: options.bottom,
            strength: options.strength,
            numSteps: denoise.numSteps,
            guidance: denoise.guidance,
            seed: denoise.seed,
            maskFeather: options.maskFeather,
            verbose: denoise.verbose,
            evalFreq: denoise.evalFrequency,
            sampler: denoise.sampler,
            guidanceSchedule: denoise.guidanceSchedule,
            progress: denoise.progress,
            cancellation: denoise.cancellation)
    }
}
