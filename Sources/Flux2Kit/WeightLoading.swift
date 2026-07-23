// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT). Weight conversion + loading.
// 2026-07-19 EDT | PERMANENT (Flux2Kit t2i port) — numerical parity with the reference implementation is the contract; do not refactor without re-running the parity harness.

import Foundation
import MLX
import MLXNN

// MARK: - Key validation

private func requireKeys(_ weights: [String: MLXArray], _ keys: [String], _ context: String) throws {
    // Require the given keys to be present.
    let missing = keys.filter { weights[$0] == nil }
    if !missing.isEmpty {
        let examples = Array(missing.prefix(5))
        throw Flux2Error.missingWeights("\(context): missing \(missing.count) key(s), e.g. \(examples)")
    }
}

// MARK: - Transformer weight conversion

/// Converts diffusers-format FLUX.2 transformer weights to this implementation's key layout.
/// Output keys exactly match the native `flux-2-klein-4b.safetensors` checkpoint naming and are
/// the module-parameter-tree contract (verified 2026-07-19 against the on-disk checkpoint header).
public func convertFlux2DiffusersWeights(_ weights: [String: MLXArray], _ cfg: Flux2Config) throws -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]

    func take(_ key: String) throws -> MLXArray {
        guard let w = weights[key] else {
            throw Flux2Error.missingWeights(key)
        }
        return w
    }

    // Rename table
    let rename: [(String, String)] = [
        ("x_embedder.weight", "img_in.weight"),
        ("context_embedder.weight", "txt_in.weight"),
        ("time_guidance_embed.timestep_embedder.linear_1.weight", "time_in.in_layer.weight"),
        ("time_guidance_embed.timestep_embedder.linear_2.weight", "time_in.out_layer.weight"),
        ("double_stream_modulation_img.linear.weight", "double_stream_modulation_img.lin.weight"),
        ("double_stream_modulation_txt.linear.weight", "double_stream_modulation_txt.lin.weight"),
        ("single_stream_modulation.linear.weight", "single_stream_modulation.lin.weight"),
        ("norm_out.linear.weight", "final_layer.adaLN_modulation.1.weight"),
        ("proj_out.weight", "final_layer.linear.weight"),
    ]
    for (src, dst) in rename {
        if let w = weights[src] {
            out[dst] = w
        }
    }

    for i in 0..<cfg.depth {
        let base = "transformer_blocks.\(i).attn"
        let requiredKeys = [
            "\(base).to_q.weight", "\(base).to_k.weight", "\(base).to_v.weight",
            "\(base).add_q_proj.weight", "\(base).add_k_proj.weight", "\(base).add_v_proj.weight",
            "\(base).to_out.0.weight", "\(base).to_add_out.weight",
            "\(base).norm_q.weight", "\(base).norm_k.weight",
            "\(base).norm_added_q.weight", "\(base).norm_added_k.weight",
            "transformer_blocks.\(i).ff.linear_in.weight", "transformer_blocks.\(i).ff.linear_out.weight",
            "transformer_blocks.\(i).ff_context.linear_in.weight", "transformer_blocks.\(i).ff_context.linear_out.weight",
        ]
        try requireKeys(weights, requiredKeys, "double_blocks.\(i)")

        // QKV fused along axis 0 in q,k,v order
        let q = try take("\(base).to_q.weight")
        let k = try take("\(base).to_k.weight")
        let v = try take("\(base).to_v.weight")
        out["double_blocks.\(i).img_attn.qkv.weight"] = concatenated([q, k, v], axis: 0)
        let aq = try take("\(base).add_q_proj.weight")
        let ak = try take("\(base).add_k_proj.weight")
        let av = try take("\(base).add_v_proj.weight")
        out["double_blocks.\(i).txt_attn.qkv.weight"] = concatenated([aq, ak, av], axis: 0)

        out["double_blocks.\(i).img_attn.proj.weight"] = try take("\(base).to_out.0.weight")
        out["double_blocks.\(i).txt_attn.proj.weight"] = try take("\(base).to_add_out.weight")

        out["double_blocks.\(i).img_attn.norm.query_norm.scale"] = try take("\(base).norm_q.weight")
        out["double_blocks.\(i).img_attn.norm.key_norm.scale"] = try take("\(base).norm_k.weight")
        out["double_blocks.\(i).txt_attn.norm.query_norm.scale"] = try take("\(base).norm_added_q.weight")
        out["double_blocks.\(i).txt_attn.norm.key_norm.scale"] = try take("\(base).norm_added_k.weight")

        out["double_blocks.\(i).img_mlp.0.weight"] = try take("transformer_blocks.\(i).ff.linear_in.weight")
        out["double_blocks.\(i).img_mlp.2.weight"] = try take("transformer_blocks.\(i).ff.linear_out.weight")
        out["double_blocks.\(i).txt_mlp.0.weight"] = try take("transformer_blocks.\(i).ff_context.linear_in.weight")
        out["double_blocks.\(i).txt_mlp.2.weight"] = try take("transformer_blocks.\(i).ff_context.linear_out.weight")
    }

    for i in 0..<cfg.depthSingleBlocks {
        let base = "single_transformer_blocks.\(i).attn"
        let requiredKeys = [
            "\(base).to_qkv_mlp_proj.weight", "\(base).to_out.weight",
            "\(base).norm_q.weight", "\(base).norm_k.weight",
        ]
        try requireKeys(weights, requiredKeys, "single_blocks.\(i)")

        out["single_blocks.\(i).linear1.weight"] = try take("\(base).to_qkv_mlp_proj.weight")
        out["single_blocks.\(i).linear2.weight"] = try take("\(base).to_out.weight")
        out["single_blocks.\(i).norm.query_norm.scale"] = try take("\(base).norm_q.weight")
        out["single_blocks.\(i).norm.key_norm.scale"] = try take("\(base).norm_k.weight")
    }

    return out
}

// MARK: - VAE weight conversion

/// Convert VAE weights from diffusers format.
public func convertVaeDiffusersWeights(_ weights: [String: MLXArray], _ cfg: VAEConfig? = nil) -> [String: MLXArray] {
    // Use config if provided, otherwise fall back to typical FLUX VAE structure
    let numBlocks: Int
    let numResnetsDown: Int
    let numResnetsUp: Int
    if let cfg {
        numBlocks = cfg.chMult.count
        numResnetsDown = cfg.numResBlocks
        numResnetsUp = cfg.numResBlocks + 1
    } else {
        numBlocks = 4
        numResnetsDown = 2
        numResnetsUp = 3
    }

    var out: [String: MLXArray] = [:]

    func rename(_ src: String, _ dst: String) {
        if let w = weights[src] {
            out[dst] = w
        }
    }

    rename("bn.running_mean", "bn.running_mean")
    rename("bn.running_var", "bn.running_var")

    rename("encoder.conv_in.weight", "encoder.conv_in.weight")
    rename("encoder.conv_in.bias", "encoder.conv_in.bias")
    rename("encoder.conv_norm_out.weight", "encoder.norm_out.weight")
    rename("encoder.conv_norm_out.bias", "encoder.norm_out.bias")
    rename("encoder.conv_out.weight", "encoder.conv_out.weight")
    rename("encoder.conv_out.bias", "encoder.conv_out.bias")

    rename("decoder.conv_in.weight", "decoder.conv_in.weight")
    rename("decoder.conv_in.bias", "decoder.conv_in.bias")
    rename("decoder.conv_norm_out.weight", "decoder.norm_out.weight")
    rename("decoder.conv_norm_out.bias", "decoder.norm_out.bias")
    rename("decoder.conv_out.weight", "decoder.conv_out.weight")
    rename("decoder.conv_out.bias", "decoder.conv_out.bias")

    rename("quant_conv.weight", "encoder.quant_conv.weight")
    rename("quant_conv.bias", "encoder.quant_conv.bias")
    rename("post_quant_conv.weight", "decoder.post_quant_conv.weight")
    rename("post_quant_conv.bias", "decoder.post_quant_conv.bias")

    // Encoder down blocks
    for i in 0..<numBlocks {
        for j in 0..<numResnetsDown {
            let prefix = "encoder.down_blocks.\(i).resnets.\(j)"
            let dst = "encoder.down.\(i).block.\(j)"
            rename("\(prefix).conv1.weight", "\(dst).conv1.weight")
            rename("\(prefix).conv1.bias", "\(dst).conv1.bias")
            rename("\(prefix).conv2.weight", "\(dst).conv2.weight")
            rename("\(prefix).conv2.bias", "\(dst).conv2.bias")
            rename("\(prefix).norm1.weight", "\(dst).norm1.weight")
            rename("\(prefix).norm1.bias", "\(dst).norm1.bias")
            rename("\(prefix).norm2.weight", "\(dst).norm2.weight")
            rename("\(prefix).norm2.bias", "\(dst).norm2.bias")
            rename("\(prefix).conv_shortcut.weight", "\(dst).nin_shortcut.weight")
            rename("\(prefix).conv_shortcut.bias", "\(dst).nin_shortcut.bias")
        }
        if i != numBlocks - 1 {
            rename(
                "encoder.down_blocks.\(i).downsamplers.0.conv.weight",
                "encoder.down.\(i).downsample.conv.weight"
            )
            rename(
                "encoder.down_blocks.\(i).downsamplers.0.conv.bias",
                "encoder.down.\(i).downsample.conv.bias"
            )
        }
    }

    // Encoder mid resnets
    for j in 0..<2 {
        let src = "encoder.mid_block.resnets.\(j)"
        let dst = j == 0 ? "encoder.mid.block_1" : "encoder.mid.block_2"
        rename("\(src).conv1.weight", "\(dst).conv1.weight")
        rename("\(src).conv1.bias", "\(dst).conv1.bias")
        rename("\(src).conv2.weight", "\(dst).conv2.weight")
        rename("\(src).conv2.bias", "\(dst).conv2.bias")
        rename("\(src).norm1.weight", "\(dst).norm1.weight")
        rename("\(src).norm1.bias", "\(dst).norm1.bias")
        rename("\(src).norm2.weight", "\(dst).norm2.weight")
        rename("\(src).norm2.bias", "\(dst).norm2.bias")
        rename("\(src).conv_shortcut.weight", "\(dst).nin_shortcut.weight")
        rename("\(src).conv_shortcut.bias", "\(dst).nin_shortcut.bias")
    }

    // Encoder mid attention
    let encAttn = "encoder.mid_block.attentions.0"
    let encAttnDst = "encoder.mid.attn_1"
    rename("\(encAttn).group_norm.weight", "\(encAttnDst).norm.weight")
    rename("\(encAttn).group_norm.bias", "\(encAttnDst).norm.bias")
    rename("\(encAttn).to_q.weight", "\(encAttnDst).q.weight")
    rename("\(encAttn).to_q.bias", "\(encAttnDst).q.bias")
    rename("\(encAttn).to_k.weight", "\(encAttnDst).k.weight")
    rename("\(encAttn).to_k.bias", "\(encAttnDst).k.bias")
    rename("\(encAttn).to_v.weight", "\(encAttnDst).v.weight")
    rename("\(encAttn).to_v.bias", "\(encAttnDst).v.bias")
    rename("\(encAttn).to_out.0.weight", "\(encAttnDst).proj_out.weight")
    rename("\(encAttn).to_out.0.bias", "\(encAttnDst).proj_out.bias")

    // Decoder up blocks — note the index reversal dst_i = num_blocks - 1 - i
    for i in 0..<numBlocks {
        let dstI = numBlocks - 1 - i
        for j in 0..<numResnetsUp {
            let src = "decoder.up_blocks.\(i).resnets.\(j)"
            let dst = "decoder.up.\(dstI).block.\(j)"
            rename("\(src).conv1.weight", "\(dst).conv1.weight")
            rename("\(src).conv1.bias", "\(dst).conv1.bias")
            rename("\(src).conv2.weight", "\(dst).conv2.weight")
            rename("\(src).conv2.bias", "\(dst).conv2.bias")
            rename("\(src).norm1.weight", "\(dst).norm1.weight")
            rename("\(src).norm1.bias", "\(dst).norm1.bias")
            rename("\(src).norm2.weight", "\(dst).norm2.weight")
            rename("\(src).norm2.bias", "\(dst).norm2.bias")
            rename("\(src).conv_shortcut.weight", "\(dst).nin_shortcut.weight")
            rename("\(src).conv_shortcut.bias", "\(dst).nin_shortcut.bias")
        }
        if i != numBlocks - 1 {
            rename(
                "decoder.up_blocks.\(i).upsamplers.0.conv.weight",
                "decoder.up.\(dstI).upsample.conv.weight"
            )
            rename(
                "decoder.up_blocks.\(i).upsamplers.0.conv.bias",
                "decoder.up.\(dstI).upsample.conv.bias"
            )
        }
    }

    // Decoder mid resnets
    for j in 0..<2 {
        let src = "decoder.mid_block.resnets.\(j)"
        let dst = j == 0 ? "decoder.mid.block_1" : "decoder.mid.block_2"
        rename("\(src).conv1.weight", "\(dst).conv1.weight")
        rename("\(src).conv1.bias", "\(dst).conv1.bias")
        rename("\(src).conv2.weight", "\(dst).conv2.weight")
        rename("\(src).conv2.bias", "\(dst).conv2.bias")
        rename("\(src).norm1.weight", "\(dst).norm1.weight")
        rename("\(src).norm1.bias", "\(dst).norm1.bias")
        rename("\(src).norm2.weight", "\(dst).norm2.weight")
        rename("\(src).norm2.bias", "\(dst).norm2.bias")
        rename("\(src).conv_shortcut.weight", "\(dst).nin_shortcut.weight")
        rename("\(src).conv_shortcut.bias", "\(dst).nin_shortcut.bias")
    }

    // Decoder mid attention
    let decAttn = "decoder.mid_block.attentions.0"
    let decAttnDst = "decoder.mid.attn_1"
    rename("\(decAttn).group_norm.weight", "\(decAttnDst).norm.weight")
    rename("\(decAttn).group_norm.bias", "\(decAttnDst).norm.bias")
    rename("\(decAttn).to_q.weight", "\(decAttnDst).q.weight")
    rename("\(decAttn).to_q.bias", "\(decAttnDst).q.bias")
    rename("\(decAttn).to_k.weight", "\(decAttnDst).k.weight")
    rename("\(decAttn).to_k.bias", "\(decAttnDst).k.bias")
    rename("\(decAttn).to_v.weight", "\(decAttnDst).v.weight")
    rename("\(decAttn).to_v.bias", "\(decAttnDst).v.bias")
    rename("\(decAttn).to_out.0.weight", "\(decAttnDst).proj_out.weight")
    rename("\(decAttn).to_out.0.bias", "\(decAttnDst).proj_out.bias")

    return out
}

// MARK: - Repo path resolution

/// The HF-cache branch is an explicit local-files-only
/// reimplementation of huggingface_hub.snapshot_download (never touches the network).
public func resolveRepoPath(_ repoId: String, _ localPath: URL? = nil, revision: String? = nil) throws -> URL {
    let fm = FileManager.default
    if let localPath {
        if !fm.fileExists(atPath: localPath.path) {
            throw Flux2Error.loadFailed("Local repo path does not exist: \(localPath.path)")
        }
        return localPath
    }
    let repoName = repoId.components(separatedBy: "/").last ?? repoId
    let cwdCandidate = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent(repoName)
    if fm.fileExists(atPath: cwdCandidate.path) {
        return cwdCandidate
    }
    return try resolveHuggingFaceSnapshot(repoId: repoId, revision: revision)
}

/// Explicit Swift stand-in for huggingface_hub.snapshot_download(repo_id, local_files_only=True):
/// resolves HF_HUB_CACHE / HF_HOME/hub / ~/.cache/huggingface/hub, models--{org}--{name},
/// refs/{revision} -> commit, snapshots/{commit}. Throws Flux2Error.loadFailed when absent.
private func resolveHuggingFaceSnapshot(repoId: String, revision: String?) throws -> URL {
    let fm = FileManager.default
    let env = ProcessInfo.processInfo.environment
    let hfHome = env["HF_HOME"].map { URL(fileURLWithPath: $0) }
        ?? fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface")
    let hubDir = env["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
        ?? hfHome.appendingPathComponent("hub")
    let repoDir = hubDir.appendingPathComponent("models--" + repoId.replacingOccurrences(of: "/", with: "--"))
    let rev = revision ?? "main"

    let snapshots = repoDir.appendingPathComponent("snapshots")
    // revision may itself name a snapshot directory (commit hash)
    let direct = snapshots.appendingPathComponent(rev)
    if fm.fileExists(atPath: direct.path) {
        return direct
    }
    let refPath = repoDir.appendingPathComponent("refs").appendingPathComponent(rev)
    guard
        let commit = (try? String(contentsOf: refPath, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines),
        !commit.isEmpty
    else {
        throw Flux2Error.loadFailed("Could not resolve local snapshot for \(repoId) (revision \(rev)) under \(repoDir.path)")
    }
    let snapshot = snapshots.appendingPathComponent(commit)
    guard fm.fileExists(atPath: snapshot.path) else {
        throw Flux2Error.loadFailed("Snapshot missing for \(repoId)@\(commit) at \(snapshot.path)")
    }
    return snapshot
}

// MARK: - Safetensors loading

/// Loads safetensors, merging files and raising on duplicate keys.
public func loadSafetensors(_ paths: [URL]) throws -> [String: MLXArray] {
    var weights: [String: MLXArray] = [:]
    for path in paths {
        let w = try MLX.loadArrays(url: path)
        let dupes = Set(weights.keys).intersection(w.keys)
        if !dupes.isEmpty {
            let examples = Array(dupes.sorted().prefix(5))
            throw Flux2Error.duplicateWeightKeys("Duplicate weight keys in \(path.lastPathComponent): \(examples)...")
        }
        weights.merge(w) { _, new in new }
    }
    return weights
}

/// Sorted non-recursive glob of *.safetensors; empty on missing dir.
public func listSafetensors(_ dirPath: URL) -> [URL] {
    let fm = FileManager.default
    guard let contents = try? fm.contentsOfDirectory(at: dirPath, includingPropertiesForKeys: nil) else {
        return []
    }
    return contents
        .filter { $0.pathExtension == "safetensors" && !$0.lastPathComponent.hasPrefix(".") }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

// MARK: - Sharded text-encoder resolution

/// Resolves the safetensors shard files in a diffusers model directory.
/// Prefers the supplied index json (parses weight_map, dedupes + sorts shard names, verifies
/// each exists); falls back to the glob order (*.safetensors, then model-*.safetensors).
public func resolveShardPaths(
    _ directory: URL,
    indexFileName: String = "model.safetensors.index.json"
) throws -> [URL] {
    let fm = FileManager.default
    let indexURL = directory.appendingPathComponent(indexFileName)
    if fm.fileExists(atPath: indexURL.path) {
        let data: Data
        do {
            data = try Data(contentsOf: indexURL)
        } catch {
            throw Flux2Error.loadFailed("Could not read \(indexURL.path): \(error.localizedDescription)")
        }
        guard
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let weightMap = object["weight_map"] as? [String: String]
        else {
            throw Flux2Error.loadFailed("Malformed safetensors index: \(indexURL.path)")
        }
        let shardNames = Set(weightMap.values).sorted()
        var urls: [URL] = []
        urls.reserveCapacity(shardNames.count)
        for name in shardNames {
            let url = directory.appendingPathComponent(name)
            guard fm.fileExists(atPath: url.path) else {
                throw Flux2Error.loadFailed("Missing shard \(name) referenced by \(indexURL.lastPathComponent)")
            }
            urls.append(url)
        }
        if !urls.isEmpty {
            return urls
        }
    }

    // Fallback glob order
    var shardPaths = listSafetensors(directory)
    if shardPaths.isEmpty {
        if let contents = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
            shardPaths = contents
                .filter { $0.lastPathComponent.hasPrefix("model-") && $0.lastPathComponent.hasSuffix(".safetensors") }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
    }
    guard !shardPaths.isEmpty else {
        throw Flux2Error.loadFailed("Could not locate safetensors shards in \(directory.path)")
    }
    return shardPaths
}

/// Loads and merges all shards in a directory (e.g. text_encoder/ with model-00001-of-00002 + index json).
/// Merge loop with duplicate-key checking.
public func loadShardedSafetensors(_ directory: URL) throws -> [String: MLXArray] {
    try loadSafetensors(resolveShardPaths(directory))
}

// MARK: - Aligned loading

/// Applies a flat key-path dictionary to a module's parameter tree.
/// strict=true enforces: missing model keys, unused dict keys,
/// and shape mismatches all throw. strict=false applies without verification.
public func applyWeights(_ module: Module, _ weights: [String: MLXArray], strict: Bool = true) throws {
    let verify: Module.VerifyUpdate = strict ? .all : .none
    try module.update(parameters: ModuleParameters.unflattened(weights), verify: verify)
}

/// Aligns weight shapes/dtypes to the module's parameters, transposing conv weights as needed, then loads them.
public func alignAndLoad(_ module: Module, _ weights: [String: MLXArray], strict: Bool = true) throws {
    let params = module.parameters().flattened()
    var out: [String: MLXArray] = [:]
    for (name, target) in params {
        guard var w = weights[name] else {
            continue
        }
        if w.shape != target.shape {
            if w.ndim == 4 && target.ndim == 4 {
                // OIHW -> OHWI conv transpose
                if w.shape[0] == target.shape[0] && w.shape[1] == target.shape[3] {
                    w = w.transposed(0, 2, 3, 1)
                }
            }
        }
        if w.dtype != target.dtype {
            w = w.asType(target.dtype)
        }
        out[name] = w
    }
    try applyWeights(module, out, strict: strict)
}

/// Aligns torch-layout weights (2D linear, OIHW conv) to the module's parameters, then loads them.
public func alignAndLoadFromTorch(_ module: Module, _ weights: [String: MLXArray], strict: Bool = true) throws {
    let params = module.parameters().flattened()
    var out: [String: MLXArray] = [:]
    for (name, target) in params {
        guard var w = weights[name] else {
            continue
        }
        if w.ndim == 2 {
            // 2D torch weight -> 1x1 NHWC conv: reshape (O,I) -> (O,I,1,1), then the 4D
            // branch below transposes to (O,1,1,I)
            if target.ndim == 4 && target.shape[1] == 1 && target.shape[2] == 1 {
                if w.shape == [target.shape[0], target.shape[3]] {
                    w = w.reshaped(target.shape[0], target.shape[3], 1, 1)
                }
            }
            // Re-check ndim == 2 because the reshape above may have made it 4D
            if w.ndim == 2 && w.shape != target.shape && w.T.shape == target.shape {
                w = w.T
            }
        }
        if w.ndim == 4 && target.ndim == 4 {
            if w.shape[0] == target.shape[0] && w.shape[1] == target.shape[3] {
                w = w.transposed(0, 2, 3, 1)
            }
        }
        if w.dtype != target.dtype {
            w = w.asType(target.dtype)
        }
        out[name] = w
    }
    try applyWeights(module, out, strict: strict)
}

// MARK: - QKV fusion for the Qwen3 text encoder

/// Fuse separate Q/K/V projection weights into single QKV weight.
/// model.layers.N.self_attn.{q,k,v}_proj.weight -> model.layers.N.self_attn.qkv_proj.weight
public func fuseQkvWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    // Matches "model.layers.(\d+).self_attn.q_proj.weight", anchored at
    // the start only, via a manual prefix/digit parse
    var layerIndices: Set<Int> = []
    let layersPrefix = "model.layers."
    for key in weights.keys {
        guard key.hasPrefix(layersPrefix) else { continue }
        let rest = key.dropFirst(layersPrefix.count)
        let digits = rest.prefix { $0.isNumber }
        guard
            !digits.isEmpty,
            rest.dropFirst(digits.count).hasPrefix(".self_attn.q_proj.weight"),
            let idx = Int(digits)
        else { continue }
        layerIndices.insert(idx)
    }

    if layerIndices.isEmpty {
        return weights  // No q_proj found, return unchanged
    }

    var out: [String: MLXArray] = [:]
    var fusedPrefixes: Set<String> = []

    for idx in layerIndices.sorted() {
        let prefix = "model.layers.\(idx).self_attn"
        let qKey = "\(prefix).q_proj.weight"
        let kKey = "\(prefix).k_proj.weight"
        let vKey = "\(prefix).v_proj.weight"

        if let qW = weights[qKey], let kW = weights[kKey], let vW = weights[vKey] {
            // Concatenate along axis 0 (out_features), q,k,v order
            out["\(prefix).qkv_proj.weight"] = concatenated([qW, kW, vW], axis: 0)
            fusedPrefixes.insert(prefix)
        }
    }

    // Copy all other weights, skipping the individual q/k/v that were fused
    for (key, value) in weights {
        var skip = false
        for prefix in fusedPrefixes {
            if key == "\(prefix).q_proj.weight" || key == "\(prefix).k_proj.weight" || key == "\(prefix).v_proj.weight" {
                skip = true
                break
            }
        }
        if !skip {
            out[key] = value
        }
    }

    return out
}
