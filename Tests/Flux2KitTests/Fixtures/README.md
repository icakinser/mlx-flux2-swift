# Parity fixtures

- `ref_bike_s42.png` — golden output from the reference implementation for the prompt
  `"a red bicycle leaning against a stone wall, golden hour"`, generated with
  `--repo <Models/FLUX-2> -s 42 -w 512 -h 512 -t 4` (bf16, unquantized). The Swift
  port must reproduce this image from the same seed; visible divergence = a parity
  bug, not taste.
