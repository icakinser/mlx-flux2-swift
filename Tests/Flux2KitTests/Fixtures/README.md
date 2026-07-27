# Image quality fixtures

- `ref_bike_s42.png` — Flux2Kit's deterministic Euler golden for the prompt
  `"a red bicycle leaning against a stone wall, golden hour"`, generated with
  `--repo <Models/FLUX-2> -s 42 -w 512 -H 512 -t 4 --guidance 1.0` (bf16,
  unquantized).
- `ref_bike_heun_s42.png` — Flux2Kit's deterministic Heun-2 golden for the same workload.
- `edit_img2img_s43.png` — source-conditioned img2img at strength 0.6.
- `edit_inpaint_s44.png` — box-mask inpaint at strength 0.85.
- `edit_outpaint_s45.png` — 32-pixel extension on every side (576×576 output).
- `ref_condition_s46.png` — single-reference conditioning.
- `ref_multi_condition_s48.png` — two-reference conditioning.
- `manifest.json` — reproducible parameters, model revision, and threshold for every fixture.

The Python reference is now a baseline rather than a permanent feature ceiling. The opt-in image
suite applies two explicit quality contracts:

- **strict:** deterministic default Euler output, mean absolute channel difference ≤ 0.5/255.
- **soft:** optimized or alternative paths such as `mx.compile`, mean difference ≤ 5/255.

Refresh these fixtures intentionally with `FLUX2_REPO=... Scripts/refresh_goldens.sh`, review the
image diffs, then run
`FLUX2_RUN_IMAGE_TESTS=1 FLUX2_REPO=... swift test --filter ImageQualityTests`.
