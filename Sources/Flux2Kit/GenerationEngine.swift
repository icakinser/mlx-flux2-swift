import Foundation
import MLX

/// Thread-safe cooperative cancellation handle. Cancellation is observed before each denoise step;
/// model forwards already in flight finish before `Flux2Error.cancelled` is thrown.
public final class GenerationCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    public init() {}

    public func cancel() {
        lock.withLock { cancelled = true }
    }

    public var isCancelled: Bool {
        lock.withLock { cancelled }
    }
}

/// Shared internal request consumed by every generation mode after it has prepared its initial
/// latent and optional conditioning. Keeping this package-scoped prevents mode-specific public APIs
/// from exposing MLX arrays while ensuring they all execute the same sampler/compile/progress policy.
package struct DenoiseRequest {
    package let image: MLXArray
    package let imageIds: MLXArray
    package let text: MLXArray
    package let textIds: MLXArray
    package let timesteps: [Double]
    package let guidance: Double
    package let guidanceDistilled: Bool
    package let imageCondition: MLXArray?
    package let imageConditionIds: MLXArray?
    package let positionImage: MLXArray?
    package let positionText: MLXArray?
    package let postStep: ((Int, Double, MLXArray) -> MLXArray)?

    package init(
        image: MLXArray,
        imageIds: MLXArray,
        text: MLXArray,
        textIds: MLXArray,
        timesteps: [Double],
        guidance: Double,
        guidanceDistilled: Bool,
        imageCondition: MLXArray? = nil,
        imageConditionIds: MLXArray? = nil,
        positionImage: MLXArray? = nil,
        positionText: MLXArray? = nil,
        postStep: ((Int, Double, MLXArray) -> MLXArray)? = nil
    ) {
        self.image = image
        self.imageIds = imageIds
        self.text = text
        self.textIds = textIds
        self.timesteps = timesteps
        self.guidance = guidance
        self.guidanceDistilled = guidanceDistilled
        self.imageCondition = imageCondition
        self.imageConditionIds = imageConditionIds
        self.positionImage = positionImage
        self.positionText = positionText
        self.postStep = postStep
    }
}

package struct DenoiseExecutionOptions {
    package let sampler: Sampler
    package let guidanceSchedule: GuidanceSchedule
    package let evalFrequency: Int
    package let verbose: Bool
    package let progress: (@Sendable (GenerationProgress) -> Void)?
    package let cancellation: GenerationCancellation?

    package init(
        sampler: Sampler = .euler,
        guidanceSchedule: GuidanceSchedule = .constant,
        evalFrequency: Int = 1,
        verbose: Bool = false,
        progress: (@Sendable (GenerationProgress) -> Void)? = nil,
        cancellation: GenerationCancellation? = nil
    ) {
        self.sampler = sampler
        self.guidanceSchedule = guidanceSchedule
        self.evalFrequency = evalFrequency
        self.verbose = verbose
        self.progress = progress
        self.cancellation = cancellation
    }
}

package struct Flux2ModelFunctions {
    package let distilled: Flux2ModelFn
    package let cfg: Flux2ModelCfgFn
}

extension Flux2Pipeline {
    /// Build eager or compiled model closures once, using the pipeline-level compile policy.
    package func makeModelFunctions(_ model: Flux2Transformer) -> Flux2ModelFunctions {
        let identity = ObjectIdentifier(model)
        if cachedModelIdentity == identity, let cachedModelFunctions {
            return cachedModelFunctions
        }
        if useCompile {
            let compiledDistilled = compile { (args: [MLXArray]) -> [MLXArray] in
                [
                    model(
                        args[0], args[1], args[2], args[3], args[4], args[5],
                        args[6], args[7], args[8], args[9])
                ]
            }
            let distilled: Flux2ModelFn = {
                image, imageIds, timestep, text, textIds, guidance,
                positionImage, positionText, textEmbedded, guidanceEmbedded in
                compiledDistilled([
                    image,
                    imageIds,
                    timestep,
                    text,
                    textIds,
                    guidance ?? MLXArray(0),
                    positionImage ?? MLXArray(0),
                    positionText ?? MLXArray(0),
                    textEmbedded ?? MLXArray(0),
                    guidanceEmbedded ?? MLXArray(0),
                ])[0]
            }

            let compiledCfg = compile { (args: [MLXArray]) -> [MLXArray] in
                [
                    model(
                        args[0], args[1], args[2], args[3], args[4], nil,
                        args[5], args[6], args[7], nil)
                ]
            }
            let cfg: Flux2ModelCfgFn = {
                image, imageIds, timestep, text, textIds,
                positionImage, positionText, textEmbedded in
                compiledCfg([
                    image,
                    imageIds,
                    timestep,
                    text,
                    textIds,
                    positionImage,
                    positionText,
                    textEmbedded,
                ])[0]
            }
            let functions = Flux2ModelFunctions(distilled: distilled, cfg: cfg)
            cachedModelIdentity = identity
            cachedModelFunctions = functions
            return functions
        }

        let distilled: Flux2ModelFn = {
            [model] image, imageIds, timestep, text, textIds, guidance,
            positionImage, positionText, textEmbedded, guidanceEmbedded in
            model(
                image, imageIds, timestep, text, textIds, guidance,
                positionImage, positionText, textEmbedded, guidanceEmbedded)
        }
        let cfg: Flux2ModelCfgFn = {
            [model] image, imageIds, timestep, text, textIds,
            positionImage, positionText, textEmbedded in
            model(
                image, imageIds, timestep, text, textIds, nil,
                positionImage, positionText, textEmbedded, nil)
        }
        let functions = Flux2ModelFunctions(distilled: distilled, cfg: cfg)
        cachedModelIdentity = identity
        cachedModelFunctions = functions
        return functions
    }

    /// Single sampler dispatch used by text-to-image, img2img, inpaint, outpaint, and wrappers.
    package func executeDenoise(
        model: Flux2Transformer,
        request: DenoiseRequest,
        options: DenoiseExecutionOptions
    ) throws -> MLXArray {
        let denoiseSignpost = Flux2Signpost.begin("Denoise")
        defer { Flux2Signpost.end("Denoise", denoiseSignpost) }
        let functions = makeModelFunctions(model)
        let observesSteps = options.verbose || options.progress != nil
        let stepTimes = Flux2StepTimes()
        let totalSteps = request.timesteps.count - 1
        let logStep: (Int, Double, Double, MLXArray, MLXArray) -> Void = {
            step, current, next, _, _ in
            let elapsed = stepTimes.values.last ?? 0
            if options.verbose {
                print(
                    String(
                        format: "[%7.1fms] Step %d/%d  t=%.4f→%.4f",
                        elapsed * 1000,
                        step + 1,
                        totalSteps,
                        current,
                        next))
            }
            options.progress?(
                GenerationProgress(
                    step: step,
                    totalSteps: totalSteps,
                    currentTimestep: current,
                    nextTimestep: next,
                    elapsedMilliseconds: elapsed * 1000))
        }

        if request.guidanceDistilled {
            if options.sampler == .heun {
                return try denoiseHeun(
                    model,
                    request.image,
                    request.imageIds,
                    request.text,
                    request.textIds,
                    timesteps: request.timesteps,
                    guidance: request.guidance,
                    imgCondSeq: request.imageCondition,
                    imgCondSeqIds: request.imageConditionIds,
                    logFn: observesSteps ? logStep : nil,
                    peX: request.positionImage,
                    peCtx: request.positionText,
                    modelFn: functions.distilled,
                    stepTimes: observesSteps ? stepTimes : nil,
                    guidanceSchedule: options.guidanceSchedule,
                    evalFreq: options.evalFrequency,
                    shouldCancel: { options.cancellation?.isCancelled == true },
                    postStep: request.postStep)
            }
            return try denoise(
                model,
                request.image,
                request.imageIds,
                request.text,
                request.textIds,
                timesteps: request.timesteps,
                guidance: request.guidance,
                imgCondSeq: request.imageCondition,
                imgCondSeqIds: request.imageConditionIds,
                logFn: observesSteps ? logStep : nil,
                peX: request.positionImage,
                peCtx: request.positionText,
                modelFn: functions.distilled,
                stepTimes: observesSteps ? stepTimes : nil,
                guidanceSchedule: options.guidanceSchedule,
                evalFreq: options.evalFrequency,
                shouldCancel: { options.cancellation?.isCancelled == true },
                postStep: request.postStep)
        }

        return try denoiseCfg(
            model,
            request.image,
            request.imageIds,
            request.text,
            request.textIds,
            timesteps: request.timesteps,
            guidance: request.guidance,
            imgCondSeq: request.imageCondition,
            imgCondSeqIds: request.imageConditionIds,
            logFn: observesSteps ? logStep : nil,
            peX: request.positionImage,
            peCtx: request.positionText,
            modelFn: functions.distilled,
            modelFnCfg: functions.cfg,
            stepTimes: observesSteps ? stepTimes : nil,
            guidanceSchedule: options.guidanceSchedule,
            evalFreq: options.evalFrequency,
            shouldCancel: { options.cancellation?.isCancelled == true },
            postStep: request.postStep)
    }
}
