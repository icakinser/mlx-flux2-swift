// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// The Python implementation is a baseline, not a ceiling; numerical changes run through the image
// quality gates and release benchmark.

import CoreGraphics
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - Forward-function typealiases

/// The `model_fn` slot in `denoise()` / `denoise_cfg()`.
/// Matches `Flux2.__call__(x, x_ids, timesteps, ctx, ctx_ids, guidance, pe_x, pe_ctx, txt_embedded, guidance_embedded)`
/// The pipeline passes the (optionally compiled) full forward here.
package typealias Flux2ModelFn = (
    _ x: MLXArray, _ xIds: MLXArray, _ timesteps: MLXArray, _ ctx: MLXArray, _ ctxIds: MLXArray,
    _ guidance: MLXArray?, _ peX: MLXArray?, _ peCtx: MLXArray?, _ txtEmbedded: MLXArray?,
    _ guidanceEmbedded: MLXArray?
) -> MLXArray

/// The `model_fn_cfg` slot in `denoise_cfg()`.
/// `_cfg_forward(x, x_ids, t, ctx, ctx_ids, pe_x, pe_ctx, txt_embedded)`
/// (guidance is always None inside the wrapper).
package typealias Flux2ModelCfgFn = (
    _ x: MLXArray, _ xIds: MLXArray, _ timesteps: MLXArray, _ ctx: MLXArray, _ ctxIds: MLXArray,
    _ peX: MLXArray, _ peCtx: MLXArray, _ txtEmbedded: MLXArray
) -> MLXArray

/// Reference-semantics box for in-place mutation of the caller's
/// `step_times: [Float]`. Swift value-type arrays cannot
/// reproduce that aliasing through an optional defaulted parameter, hence the box.
package final class Flux2StepTimes {
    package var values: [Double]
    package init() { self.values = [] }
}

/// Denoising integrator. `euler` is the deterministic golden default (one forward pass per step).
/// `heun` is an additive second-order corrector that trades 2x forward passes for smoother
/// low-step output; it is opt-in and never changes the Euler path.
public enum Sampler: String, Sendable {
    case euler
    case heun
}

/// Per-step guidance policy. The default is constant and preserves the known-good klein recipe.
/// `linear(end:)` is an explicit experiment that interpolates from `guidance` to `end`.
public enum GuidanceSchedule: Sendable, Equatable {
    case constant
    case linear(end: Double)

    package func value(step: Int, totalSteps: Int, start: Double) -> Double {
        switch self {
        case .constant:
            return start
        case .linear(let end):
            guard totalSteps > 1 else { return start }
            let fraction = Double(step) / Double(totalSteps - 1)
            return start + fraction * (end - start)
        }
    }
}

// MARK: - Schedule

// Generalized time-dependent SNR shift.
package func generalizedTimeSnrShift(_ t: MLXArray, _ mu: Double, _ sigma: Double) -> MLXArray {
    // math.exp(mu) is a double; MLX weak-scalar promotion keeps the
    // array dtype (float32), which toArrays() reproduces exactly in mlx-swift.
    let expMu = Foundation.exp(mu)
    return expMu / (expMu + MLX.pow(1 / t - 1, sigma))
}

// Constants preserved exactly
package func computeEmpiricalMu(_ imageSeqLen: Int, _ numSteps: Int) -> Double {
    let a1 = 8.73809524e-05
    let b1 = 1.89833333
    let a2 = 0.00016927
    let b2 = 0.45666666
    if imageSeqLen > 4300 {
        return a2 * Double(imageSeqLen) + b2
    }
    let m200 = a2 * Double(imageSeqLen) + b2
    let m10 = a1 * Double(imageSeqLen) + b1
    let a = (m200 - m10) / 190.0
    let b = m200 - 200.0 * a
    return a * Double(numSteps) + b
}

// Builds the sampling schedule from the empirical mu shift.
package func getSchedule(_ numSteps: Int, _ imageSeqLen: Int) -> [Double] {
    let mu = computeEmpiricalMu(imageSeqLen, numSteps)
    var timesteps = MLX.linspace(Float(1.0), Float(0.0), count: numSteps + 1)  // mx.linspace default float32
    timesteps = generalizedTimeSnrShift(timesteps, mu, 1.0)
    // .tolist() yields doubles — float32 widened exactly to double
    return timesteps.asArray(Float.self).map { Double($0) }
}

// MARK: - Position-coordinate builders

// Builds position coordinates for text tokens.
package func prcTxt(_ x: MLXArray, _ tCoord: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let l = x.dim(0)
    let t = tCoord ?? MLX.arange(1)
    let h = MLX.arange(1)
    let w = MLX.arange(1)
    let lCoords = MLX.arange(l)
    let grids = MLX.meshGrid([t, h, w, lCoords], indexing: .ij)
    let (tt, hh, ww, ll) = (grids[0], grids[1], grids[2], grids[3])
    let ids = MLX.stacked([tt, hh, ww, ll], axis: -1).reshaped([-1, 4])
    return (x, ids)
}

// x is (C, H, W); tokens come out in raster order
package func prcImg(_ x: MLXArray, _ tCoord: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let c = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let t = tCoord ?? MLX.arange(1)
    let hCoords = MLX.arange(h)
    let wCoords = MLX.arange(w)
    let l = MLX.arange(1)
    let grids = MLX.meshGrid([t, hCoords, wCoords, l], indexing: .ij)
    let (tt, hh, ww, ll) = (grids[0], grids[1], grids[2], grids[3])
    let ids = MLX.stacked([tt, hh, ww, ll], axis: -1).reshaped([-1, 4])
    let tokens = x.transposed(1, 2, 0).reshaped([h * w, c])
    return (tokens, ids)
}

// Batched variant of prcImg.
package func batchedPrcImg(_ x: MLXArray, _ tCoord: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let batch = x.dim(0)
    let channels = x.dim(1)
    let height = x.dim(2)
    let width = x.dim(3)
    let tokens = x.transposed(0, 2, 3, 1).reshaped([batch, height * width, channels])

    // Spatial IDs are identical across the batch. Build them once, broadcast, then replace only
    // the temporal column when the caller supplies per-batch temporal coordinates.
    let baseIds = prcImg(x[0]).1
    let ids = MLX.broadcast(
        MLX.expandedDimensions(baseIds, axis: 0),
        to: [batch, height * width, 4])
    if let tCoord {
        let times = MLX.broadcast(tCoord.reshaped([batch, 1]), to: [batch, height * width])
        ids[0..., 0..., 0] = times
    }
    return (tokens, ids)
}

// Batched variant of prcTxt.
package func batchedPrcTxt(_ x: MLXArray, _ tCoord: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let batch = x.dim(0)
    let length = x.dim(1)
    let baseIds = prcTxt(x[0]).1
    let ids = MLX.broadcast(MLX.expandedDimensions(baseIds, axis: 0), to: [batch, length, 4])
    if let tCoord {
        let times = MLX.broadcast(tCoord.reshaped([batch, 1]), to: [batch, length])
        ids[0..., 0..., 0] = times
    }
    return (x, ids)
}

// List variant of prcImg for reference-image encoding.
package func listedPrcImg(_ xList: [MLXArray], _ tCoord: [MLXArray]? = nil) -> ([MLXArray], [MLXArray]) {
    var toks: [MLXArray] = []
    var ids: [MLXArray] = []
    for (i, x) in xList.enumerated() {
        let tI = tCoord?[i]
        let (tok, idx) = prcImg(x, tI)
        toks.append(tok)
        ids.append(idx)
    }
    return (toks, ids)
}

// MARK: - Reference-image encoding (kontext-editing hook)

/// Encode reference images for conditioning.
///
/// - Parameters:
///   - ae: VAE encoder
///   - imgCtx: List of reference images
///   - limitPixelsSingle: Max pixels for single image (default: 2048^2)
///   - limitPixelsMulti: Max pixels per image when multiple (default: 1024^2)
///
/// Reference temporal offsets are fixed at `10 + 10*i` (the offsets klein was trained with); see the
/// note at the position-id construction below.
package func encodeImageRefs(
    _ ae: AutoEncoder,
    _ imgCtx: [CGImage],
    limitPixelsSingle: Int = refImageLimitPixelsSingle,
    limitPixelsMulti: Int = refImageLimitPixelsMulti
) throws -> (MLXArray?, MLXArray?) {
    if imgCtx.isEmpty {
        return (nil, nil)
    }
    let limitPixels: Int
    if imgCtx.count > 1 {
        limitPixels = limitPixelsMulti
    } else {
        limitPixels = limitPixelsSingle
    }
    let imgCtxPrep = try imgCtx.map { try defaultPrep($0, limitPixels: limitPixels) }
    var encodedRefs: [MLXArray] = []
    for img in imgCtxPrep {
        var latent = try ae.encode(MLX.expandedDimensions(img, axis: 0))
        latent = latent.transposed(0, 3, 1, 2)[0]
        encodedRefs.append(latent)
    }
    // 2026-07-21 EDT | PERMANENT (edit-fidelity fix) — reference tokens go at temporal
    // offset 10, 20, 30 (the offset klein was TRAINED with), NOT the compact
    // 1, 2, 3. That compacting was done purely to avoid the host-sync in compressTime,
    // but at t=1 the RoPE position is nearly coincident with the canvas at t=0, so the
    // model cannot distinguish "reference" from "canvas" and strong edits (object removal,
    // recolor) collapse — while global edits survive. Verified against mflux, which uses
    // 10 + 10*i and removes objects correctly at 4 steps.
    // These reference ids feed RoPE during the forward pass only; the final latent scatter
    // operates on the canvas tokens (t=0), so this does not affect decode.
    let tOff = (0 ..< encodedRefs.count).map { MLXArray([Int32(10 + 10 * $0)]) }
    let (refTokList, refIdList) = listedPrcImg(encodedRefs, tOff)
    var refTokens = MLX.concatenated(refTokList, axis: 0)
    var refIds = MLX.concatenated(refIdList, axis: 0)
    refTokens = MLX.expandedDimensions(refTokens, axis: 0)
    refIds = MLX.expandedDimensions(refIds, axis: 0)
    return (refTokens, refIds)
}

// MARK: - Token scatter (latent reassembly)

// Scatters flattened tokens back to their spatial positions per batch entry.
package func scatterIds(_ x: MLXArray, _ xIds: MLXArray) -> [MLXArray] {
    var outList: [MLXArray] = []
    // zip(x, x_ids) iterates the leading (batch) axis of both arrays
    for i in 0 ..< x.dim(0) {
        let data = x[i]
        let pos = xIds[i]
        let tIds = pos[0..., 0].asType(.int32)
        let hIds = pos[0..., 1].asType(.int32)
        let wIds = pos[0..., 2].asType(.int32)

        // Query dimension bounds. Evaluating all four reductions together forces a single
        // host<->device sync per batch element instead of one per `.item()` read.
        let tMin = MLX.min(tIds)
        let tMax = MLX.max(tIds)
        let hMax = MLX.max(hIds)
        let wMax = MLX.max(wIds)
        MLX.eval(tMin, tMax, hMax, wMax)
        let tMinVal = tMin.item(Int.self)
        let tMaxVal = tMax.item(Int.self)
        let h = hMax.item(Int.self) + 1
        let w = wMax.item(Int.self) + 1
        let c = data.dim(1)

        if tMinVal == tMaxVal {
            // Fast path: single time step, tokens are already in raster order
            // from prcImg(), so we can reshape directly without scatter
            let out = data.reshaped([1, h, w, c]).transposed(3, 0, 1, 2)
            outList.append(MLX.expandedDimensions(out, axis: 0))
        } else {
            // Slow path: multiple time steps (reference images)
            let tIdsCmpr = compressTime(tIds)
            let t = MLX.max(tIdsCmpr).item(Int.self) + 1
            let flatIds = tIdsCmpr * (h * w) + hIds * w + wIds
            var out = MLX.zeros([t * h * w, c], dtype: data.dtype)
            let indices = MLX.broadcast(flatIds[0..., .newAxis], to: [flatIds.dim(0), c])
            out = MLX.putAlong(out, indices, values: data, axis: 0)
            let reassembled = out.reshaped([t, h, w, c]).transposed(3, 0, 1, 2)
            outList.append(MLX.expandedDimensions(reassembled, axis: 0))
        }
    }
    return outList
}

// Host-side remap of time IDs to a dense 0..n-1 range
package func compressTime(_ tIds: MLXArray) -> MLXArray {
    let tMin = MLX.min(tIds)
    let tMax = MLX.max(tIds)
    MLX.eval(tMin, tMax)
    if tMin.item(Int.self) == tMax.item(Int.self) {
        return MLX.zeros(like: tIds)
    }
    let tList = tIds.asArray(Int32.self).map { Int($0) }
    let uniq = Array(Set(tList)).sorted()
    var remap: [Int: Int] = [:]
    for (i, val) in uniq.enumerated() {
        remap[val] = i
    }
    // parity: remap is total over tList by construction (built from its unique values); ?? 0 is unreachable
    let mapped = tList.map { remap[$0] ?? 0 }
    return MLXArray(mapped.map { Int32($0) }).asType(tIds.dtype)
}

// MARK: - Denoise loops

// Single-forward (non-CFG) denoising loop.
package func denoise(
    _ model: Flux2Transformer,
    _ img: MLXArray,
    _ imgIds: MLXArray,
    _ txt: MLXArray,
    _ txtIds: MLXArray,
    timesteps: [Double],
    guidance: Double,
    imgCondSeq: MLXArray? = nil,
    imgCondSeqIds: MLXArray? = nil,
    logFn: ((Int, Double, Double, MLXArray, MLXArray) -> Void)? = nil,
    peX: MLXArray? = nil,
    peCtx: MLXArray? = nil,
    modelFn: Flux2ModelFn? = nil,
    stepTimes: Flux2StepTimes? = nil,
    txtEmbedded: MLXArray? = nil,
    guidanceEmbedded: MLXArray? = nil,
    guidanceSchedule: GuidanceSchedule = .constant,
    evalFreq: Int = 1,
    shouldCancel: (() -> Bool)? = nil,
    // Optional per-step transform applied to the latent AFTER the step update, before eval.
    // Defaults to nil (no-op) so the deterministic default path is unchanged. Used by inpainting to
    // re-blend the KEEP region each step. Receives (step, tPrev, img) and returns the new img.
    postStep: ((Int, Double, MLXArray) -> MLXArray)? = nil
) throws -> MLXArray {
    var img = img
    // model_fn defaults to the model itself
    let modelFn = modelFn ?? { x, xIds, t, ctx, ctxIds, g, pX, pCtx, txtEmb, gEmb in
        model(x, xIds, t, ctx, ctxIds, g, pX, pCtx, txtEmb, gEmb)
    }
    let imgInputIds: MLXArray
    // Branches on img_cond_seq and assumes ids are present with it
    if imgCondSeq != nil, let imgCondSeqIds {
        imgInputIds = MLX.concatenated([imgIds, imgCondSeqIds], axis: 1)
    } else {
        imgInputIds = imgIds
    }
    let peX = peX ?? model.peEmbedder(imgInputIds)
    let peCtx = peCtx ?? model.peEmbedder(txtIds)
    let txtEmbedded = txtEmbedded ?? model.embedTxt(txt)
    let numSteps = timesteps.count - 1
    // enumerate(zip(timesteps[:-1], timesteps[1:]))
    for (step, (tCurr, tPrev)) in zip(timesteps.dropLast(), timesteps.dropFirst()).enumerated() {
        if shouldCancel?() == true { throw Flux2Error.cancelled }
        let signpost = Flux2Signpost.beginStep(step)
        defer { Flux2Signpost.end("DenoiseStep", signpost) }
        let stepStart = ProcessInfo.processInfo.systemUptime  // monotonic clock
        let tVec = MLX.full([img.dim(0)], values: tCurr.asMLXArray(dtype: img.dtype), dtype: img.dtype)
        let stepGuidance = guidanceSchedule.value(step: step, totalSteps: numSteps, start: guidance)
        let guidanceVec = MLX.full(
            [img.dim(0)], values: stepGuidance.asMLXArray(dtype: img.dtype), dtype: img.dtype)
        let stepGuidanceEmbedded: MLXArray?
        if model.useGuidanceEmbed {
            stepGuidanceEmbedded = guidanceSchedule == .constant && guidanceEmbedded != nil
                ? guidanceEmbedded : model.embedGuidance(guidanceVec)
        } else {
            stepGuidanceEmbedded = nil
        }
        var imgInput = img
        if let imgCondSeq {
            imgInput = MLX.concatenated([imgInput, imgCondSeq], axis: 1)
        }
        var pred = modelFn(
            imgInput,
            imgInputIds,
            tVec,
            txt,
            txtIds,
            guidanceVec,
            peX,
            peCtx,
            txtEmbedded,
            stepGuidanceEmbedded
        )
        if imgCondSeq != nil {
            pred = pred[0..., ..<img.dim(1)]
        }
        // Pre-cast dt to model dtype to avoid potential float64 promotion
        // mx.array(t_prev - t_curr, dtype=img.dtype), double→dtype in one cast
        let dt = (tPrev - tCurr).asMLXArray(dtype: img.dtype)
        img = img + dt * pred
        if let postStep { img = postStep(step, tPrev, img) }
        if evalFreq <= 1 || (step + 1) % evalFreq == 0 || step == numSteps - 1 {
            MLX.eval(img)
        }
        let stepTime = ProcessInfo.processInfo.systemUptime - stepStart
        if let stepTimes {
            stepTimes.values.append(stepTime)
        }
        if let logFn {
            logFn(step, tCurr, tPrev, img, pred)
        }
    }
    return img
}

// 2026-07-26 EDT | PERMANENT — Heun-2 corrector for improved low-step quality
/// Heun (improved Euler / explicit trapezoid) integrator for the distilled guidance path. Same
/// signature and setup as `denoise`, but each step performs a predictor (Euler) forward pass plus a
/// corrector forward pass at `tPrev` using the Euler-predicted latent, then averages the two
/// velocities. This doubles the forward passes per step in exchange for smoother output at low step
/// counts. The `postStep` hook fires after the corrected update, matching `denoise`.
package func denoiseHeun(
    _ model: Flux2Transformer,
    _ img: MLXArray,
    _ imgIds: MLXArray,
    _ txt: MLXArray,
    _ txtIds: MLXArray,
    timesteps: [Double],
    guidance: Double,
    imgCondSeq: MLXArray? = nil,
    imgCondSeqIds: MLXArray? = nil,
    logFn: ((Int, Double, Double, MLXArray, MLXArray) -> Void)? = nil,
    peX: MLXArray? = nil,
    peCtx: MLXArray? = nil,
    modelFn: Flux2ModelFn? = nil,
    stepTimes: Flux2StepTimes? = nil,
    txtEmbedded: MLXArray? = nil,
    guidanceEmbedded: MLXArray? = nil,
    guidanceSchedule: GuidanceSchedule = .constant,
    evalFreq: Int = 1,
    shouldCancel: (() -> Bool)? = nil,
    postStep: ((Int, Double, MLXArray) -> MLXArray)? = nil
) throws -> MLXArray {
    var img = img
    let modelFn = modelFn ?? { x, xIds, t, ctx, ctxIds, g, pX, pCtx, txtEmb, gEmb in
        model(x, xIds, t, ctx, ctxIds, g, pX, pCtx, txtEmb, gEmb)
    }
    let imgInputIds: MLXArray
    if imgCondSeq != nil, let imgCondSeqIds {
        imgInputIds = MLX.concatenated([imgIds, imgCondSeqIds], axis: 1)
    } else {
        imgInputIds = imgIds
    }
    let peX = peX ?? model.peEmbedder(imgInputIds)
    let peCtx = peCtx ?? model.peEmbedder(txtIds)
    let txtEmbedded = txtEmbedded ?? model.embedTxt(txt)
    let numSteps = timesteps.count - 1
    for (step, (tCurr, tPrev)) in zip(timesteps.dropLast(), timesteps.dropFirst()).enumerated() {
        if shouldCancel?() == true { throw Flux2Error.cancelled }
        let signpost = Flux2Signpost.beginStep(step)
        defer { Flux2Signpost.end("DenoiseStep", signpost) }
        let stepStart = ProcessInfo.processInfo.systemUptime
        let tVec = MLX.full([img.dim(0)], values: tCurr.asMLXArray(dtype: img.dtype), dtype: img.dtype)
        let stepGuidance = guidanceSchedule.value(step: step, totalSteps: numSteps, start: guidance)
        let guidanceVec = MLX.full(
            [img.dim(0)], values: stepGuidance.asMLXArray(dtype: img.dtype), dtype: img.dtype)
        let stepGuidanceEmbedded: MLXArray?
        if model.useGuidanceEmbed {
            stepGuidanceEmbedded = guidanceSchedule == .constant && guidanceEmbedded != nil
                ? guidanceEmbedded : model.embedGuidance(guidanceVec)
        } else {
            stepGuidanceEmbedded = nil
        }
        var imgInput = img
        if let imgCondSeq {
            imgInput = MLX.concatenated([imgInput, imgCondSeq], axis: 1)
        }
        // Predictor velocity at tCurr.
        var pred1 = modelFn(
            imgInput, imgInputIds, tVec, txt, txtIds, guidanceVec, peX, peCtx, txtEmbedded,
            stepGuidanceEmbedded)
        if imgCondSeq != nil { pred1 = pred1[0..., ..<img.dim(1)] }
        let dt = (tPrev - tCurr).asMLXArray(dtype: img.dtype)
        // Euler-predicted latent at tPrev.
        let imgEuler = img + dt * pred1

        // Corrector velocity at tPrev, evaluated on the Euler-predicted latent.
        var pred2: MLXArray
        if step < numSteps - 1 {
            let tPrevVec = MLX.full(
                [img.dim(0)], values: tPrev.asMLXArray(dtype: img.dtype), dtype: img.dtype)
            let nextGuidance = guidanceSchedule.value(
                step: step + 1, totalSteps: numSteps, start: guidance)
            let nextGuidanceVec = MLX.full(
                [img.dim(0)], values: nextGuidance.asMLXArray(dtype: img.dtype), dtype: img.dtype)
            let nextGuidanceEmbedded = model.useGuidanceEmbed
                ? model.embedGuidance(nextGuidanceVec) : nil
            var imgEulerInput = imgEuler
            if let imgCondSeq {
                imgEulerInput = MLX.concatenated([imgEulerInput, imgCondSeq], axis: 1)
            }
            pred2 = modelFn(
                imgEulerInput, imgInputIds, tPrevVec, txt, txtIds, nextGuidanceVec, peX, peCtx,
                txtEmbedded, nextGuidanceEmbedded)
            if imgCondSeq != nil { pred2 = pred2[0..., ..<img.dim(1)] }
        } else {
            // Final step: Heun collapses to Euler (no corrector), matching standard practice.
            pred2 = pred1
        }
        // Trapezoid average of the two velocities.
        let pred = (pred1 + pred2) * 0.5
        img = img + dt * pred
        if let postStep { img = postStep(step, tPrev, img) }
        if evalFreq <= 1 || (step + 1) % evalFreq == 0 || step == numSteps - 1 {
            MLX.eval(img)
        }
        let stepTime = ProcessInfo.processInfo.systemUptime - stepStart
        if let stepTimes {
            stepTimes.values.append(stepTime)
        }
        if let logFn {
            logFn(step, tCurr, tPrev, img, pred)
        }
    }
    return img
}

// CFG (classifier-free guidance) denoising loop.
package func denoiseCfg(
    _ model: Flux2Transformer,
    _ img: MLXArray,
    _ imgIds: MLXArray,
    _ txt: MLXArray,
    _ txtIds: MLXArray,
    timesteps: [Double],
    guidance: Double,
    imgCondSeq: MLXArray? = nil,
    imgCondSeqIds: MLXArray? = nil,
    logFn: ((Int, Double, Double, MLXArray, MLXArray) -> Void)? = nil,
    peX: MLXArray? = nil,
    peCtx: MLXArray? = nil,
    modelFn: Flux2ModelFn? = nil,
    modelFnCfg: Flux2ModelCfgFn? = nil,
    stepTimes: Flux2StepTimes? = nil,
    txtEmbedded: MLXArray? = nil,
    guidanceSchedule: GuidanceSchedule = .constant,
    evalFreq: Int = 1,
    shouldCancel: (() -> Bool)? = nil,
    // See denoise(): optional per-step latent transform, applied to the doubled (uncond,cond)
    // batch after the step update. Defaults to nil (no-op). A (1,N,1) mask broadcasts over both
    // halves of the CFG batch, so the same blend applies to uncond and cond.
    postStep: ((Int, Double, MLXArray) -> MLXArray)? = nil
) throws -> MLXArray {
    // model_fn defaults to the model itself
    let resolvedModelFn = modelFn ?? { x, xIds, t, ctx, ctxIds, g, pX, pCtx, txtEmb, gEmb in
        model(x, xIds, t, ctx, ctxIds, g, pX, pCtx, txtEmb, gEmb)
    }
    // Double the batch (uncond, cond)
    var img = MLX.concatenated([img, img], axis: 0)
    let imgIds = MLX.concatenated([imgIds, imgIds], axis: 0)
    var imgCondSeq = imgCondSeq
    var imgCondSeqIds = imgCondSeqIds
    if let seq = imgCondSeq, let seqIds = imgCondSeqIds {
        imgCondSeq = MLX.concatenated([seq, seq], axis: 0)
        imgCondSeqIds = MLX.concatenated([seqIds, seqIds], axis: 0)
    }

    let imgInputIds: MLXArray
    if imgCondSeq != nil, let ids = imgCondSeqIds {
        imgInputIds = MLX.concatenated([imgIds, ids], axis: 1)
    } else {
        imgInputIds = imgIds
    }
    let peX = peX ?? model.peEmbedder(imgInputIds)
    let peCtx = peCtx ?? model.peEmbedder(txtIds)
    let txtEmbedded = txtEmbedded ?? model.embedTxt(txt)

    // Determine which forward function to use for CFG
    let callFnCfg: Flux2ModelCfgFn?
    let callFn: Flux2ModelFn
    if let modelFnCfg {
        // Use the dedicated CFG forward (compiled or not)
        callFnCfg = modelFnCfg
        callFn = resolvedModelFn  // parity: never invoked on this path
    } else if model.useGuidanceEmbed {
        // Fallback: use uncompiled model when guidance embed is enabled
        callFnCfg = nil
        callFn = { x, xIds, t, ctx, ctxIds, g, pX, pCtx, txtEmb, gEmb in
            model(x, xIds, t, ctx, ctxIds, g, pX, pCtx, txtEmb, gEmb)
        }
    } else {
        // No guidance embed, use standard model_fn
        callFnCfg = nil
        callFn = resolvedModelFn
    }

    let numSteps = timesteps.count - 1
    for (step, (tCurr, tPrev)) in zip(timesteps.dropLast(), timesteps.dropFirst()).enumerated() {
        if shouldCancel?() == true { throw Flux2Error.cancelled }
        let signpost = Flux2Signpost.beginStep(step)
        defer { Flux2Signpost.end("DenoiseStep", signpost) }
        let stepStart = ProcessInfo.processInfo.systemUptime  // monotonic clock
        let tVec = MLX.full([img.dim(0)], values: tCurr.asMLXArray(dtype: img.dtype), dtype: img.dtype)
        let stepGuidance = guidanceSchedule.value(step: step, totalSteps: numSteps, start: guidance)
        let guidanceArr = stepGuidance.asMLXArray(dtype: img.dtype)
        var imgInput = img
        if let seq = imgCondSeq {
            imgInput = MLX.concatenated([imgInput, seq], axis: 1)
        }

        var pred: MLXArray
        if let callFnCfg {
            // CFG wrapper takes fewer args (guidance is always None)
            pred = callFnCfg(imgInput, imgInputIds, tVec, txt, txtIds, peX, peCtx, txtEmbedded)
        } else {
            // guidance arg is None when the model embeds guidance,
            // else an explicit zeros vector over the doubled batch
            let guidanceArg: MLXArray? = model.useGuidanceEmbed
                ? nil
                : MLX.zeros([img.dim(0)], dtype: img.dtype)
            pred = callFn(
                imgInput,
                imgInputIds,
                tVec,
                txt,
                txtIds,
                guidanceArg,
                peX,
                peCtx,
                txtEmbedded,
                nil
            )
        }
        if imgCondSeq != nil {
            pred = pred[0..., ..<img.dim(1)]
        }
        let predParts = MLX.split(pred, parts: 2, axis: 0)
        let predUncond = predParts[0]
        let predCond = predParts[1]
        pred = predUncond + guidanceArr * (predCond - predUncond)
        pred = MLX.concatenated([pred, pred], axis: 0)
        // Pre-cast dt to model dtype to avoid potential float64 promotion
        let dt = (tPrev - tCurr).asMLXArray(dtype: img.dtype)
        img = img + dt * pred
        if let postStep { img = postStep(step, tPrev, img) }
        if evalFreq <= 1 || (step + 1) % evalFreq == 0 || step == numSteps - 1 {
            MLX.eval(img)
        }
        let stepTime = ProcessInfo.processInfo.systemUptime - stepStart
        if let stepTimes {
            stepTimes.values.append(stepTime)
        }
        if let logFn {
            logFn(step, tCurr, tPrev, img, pred)
        }
    }

    return MLX.split(img, parts: 2, axis: 0)[0]
}
