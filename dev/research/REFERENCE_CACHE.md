# Reference-image cache research

**Decision:** No-go for transformer K/V caching in the current architecture.

## What is already cached per generation

The reference path does not redo static work per denoise step:

- `encodeImageRefs` prepares and VAE-encodes each reference once.
- Reference tokens and IDs are concatenated once.
- `Flux2Pipeline.generate` computes image/reference RoPE (`peX`) once.
- Text RoPE (`peCtx`) and text projection are computed outside the denoise loop.

This means the obvious safe caching work is already present.

## Why block-level K/V reuse is unsafe

FLUX.2 does not process reference tokens through an isolated, static cross-attention branch. Canvas
and reference tokens enter the same image sequence. Double-stream blocks update the full image
sequence, then single-stream blocks concatenate text and image tokens and update them jointly.
Although the input reference latent is fixed, its hidden state after each block depends on the
current noisy canvas state. K/V values therefore change at every denoise step and cannot be reused
without changing model semantics.

A correct cache would require an architectural split or approximation that the checkpoint was not
trained for. That is a quality experiment, not a transparent performance optimization.

## Go/no-go criterion

Do not ship a K/V cache until a prototype:

1. demonstrates at least 10% release-build speedup on a multi-reference workload, and
2. stays below the soft 5/255 mean-difference gate across the image suite.

No such safe prototype exists in the current block structure, so this board item concludes as a
documented no-go. Future work should focus on lower-risk wins (batch position vectorization,
compile improvements in MLX, and reference VAE residency/caching across repeated app requests).
