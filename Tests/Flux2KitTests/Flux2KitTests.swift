// Flux2Kit — golden parity tests against the reference implementation.
// Golden values were captured once from the reference and are checked in here (see
// Fixtures/README.md). CPU-only: no MLX GPU ops (swift test runs without the Cmlx
// metallib), so these cover the tokenizer/template contract, not tensors.

import Foundation
import Testing
@testable import Flux2Kit

// Path to the FLUX.2 diffusers snapshot. Set FLUX2_REPO to run the tokenizer
// parity tests; otherwise they self-skip via the `tokenizerAvailable` guard.
private let repoPath = URL(fileURLWithPath:
    ProcessInfo.processInfo.environment["FLUX2_REPO"] ?? "./Models/FLUX-2")

private var tokenizerAvailable: Bool {
    FileManager.default.fileExists(
        atPath: repoPath.appendingPathComponent("tokenizer/tokenizer.json").path)
}

@Test func chatTemplateMatchesReferenceRender() async throws {
    // Golden: repr of the reference Qwen3 chat template applied to 'TESTPROMPT'.
    let expected = "<|im_start|>user\nTESTPROMPT<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    try #require(tokenizerAvailable, "FLUX-2 tokenizer not on disk; skipping")
    let tok = try await Qwen3Tokenizer.fromRepo(repoPath)
    #expect(tok.applyChatTemplate("TESTPROMPT") == expected)
}

@Test func specialTokenIdsMatchGolden() async throws {
    try #require(tokenizerAvailable, "FLUX-2 tokenizer not on disk; skipping")
    let tok = try await Qwen3Tokenizer.fromRepo(repoPath)
    // Golden: pad_id/eos_id from the reference tokenizer.
    #expect(tok.padId == 151643)
    #expect(tok.eosId == 151645)
}

@Test func tokenIdsMatchGolden() async throws {
    try #require(tokenizerAvailable, "FLUX-2 tokenizer not on disk; skipping")
    let tok = try await Qwen3Tokenizer.fromRepo(repoPath)
    // Golden: reference encode_batch(['a red bicycle'], max_length=512) unpadded prefix.
    let goldenPrefix = [
        151644, 872, 198, 64, 2518, 34986, 151645, 198, 151644, 77091, 198, 151667, 271, 151668,
        271,
    ]
    let text = tok.applyChatTemplate("a red bicycle")
    let ids = tok.tokenizer.encode(text: text, addSpecialTokens: false)
    #expect(Array(ids.prefix(goldenPrefix.count)) == goldenPrefix)
    // The reference pads to ceil(15/64)*64 = 64 with pad_id; mirror the arithmetic (host-side check).
    let targetLen = min(512, ((ids.count + 63) / 64) * 64)
    #expect(targetLen == 64)
}
