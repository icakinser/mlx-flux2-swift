// Flux2Kit — opt-in weight download from the Hugging Face Hub (via swift-transformers' HubApi).
// Never runs implicitly; a caller must ask for it. FLUX.2 [klein] is large (~15 GB) and may be
// license-gated on the Hub — pass an HF token (or set HF_TOKEN) if the repo requires accepting terms.

import Foundation
import Hub

public enum WeightDownloadError: Error, CustomStringConvertible {
    case failed(String)

    public var description: String {
        switch self {
        case .failed(let m): return m
        }
    }
}

/// Download the FLUX.2 snapshot from the Hugging Face Hub into the local HF cache and return the
/// local snapshot directory (suitable to pass as `repoPath`). Reports 0...1 progress.
///
/// - Parameters:
///   - repoId: Hub repo (default: the FLUX.2 [klein] 4B repo).
///   - revision: git revision / branch (default "main").
///   - hfToken: auth token; defaults to `HF_TOKEN` in the environment. Needed for gated repos.
///   - progress: called with fraction complete in [0, 1].
@discardableResult
public func downloadFluxSnapshot(
    repoId: String = defaultRepoId,
    revision: String = "main",
    hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"],
    progress: @Sendable @escaping (Double) -> Void = { _ in }
) async throws -> URL {
    let api = HubApi(hfToken: hfToken)
    do {
        let url = try await api.snapshot(from: Hub.Repo(id: repoId), revision: revision) { p in
            progress(p.fractionCompleted)
        }
        return url
    } catch {
        throw WeightDownloadError.failed(
            "Could not download \(repoId): \(error). If the repo is license-gated, accept the terms "
                + "at https://huggingface.co/\(repoId) and set HF_TOKEN to a token with access.")
    }
}

/// Human-readable guidance shown when weights can't be located, so the user knows how to proceed.
public func weightsHelpMessage(repoId: String = defaultRepoId) -> String {
    """
    FLUX.2 weights were not found. Options:
      • Let the app fetch them:   flux2kit-cli --download   (needs ~15 GB free; HF_TOKEN if gated)
      • Or download yourself:     huggingface-cli download \(repoId)
        then point at them:       FLUX2_REPO=/path/to/snapshot flux2kit-cli ...
      • Accept the model license first at: https://huggingface.co/\(repoId)
    """
}
