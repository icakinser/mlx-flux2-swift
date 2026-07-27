# Flux2Kit Release Checklist

Do not create a tag until every required gate is green.

## Required gates

- `swift test` passes.
- `FLUX2_RUN_MLX_TESTS=1 FLUX2_REPO=... swift test` passes.
- `FLUX2_RUN_IMAGE_TESTS=1 FLUX2_REPO=... swift test --filter ImageQualityTests` passes.
- `Scripts/smoke_consumer.sh --local` passes before push.
- The remote consumer smoke job passes against the exact release revision.
- `Scripts/bench.sh` has been run on the release runner and the result reviewed.
- `git diff --check` and `bash -n Scripts/*.sh` pass.
- `Docs/Library.md`, `Docs/ConsumerSetup.md`, examples, and release notes match the shipped API.

The `Release readiness` GitHub workflow automates these checks on a tag or manual dispatch. Its
image-quality and benchmark job requires an Apple Silicon self-hosted runner with `FLUX2_REPO`.

## Versioning

- Patch: compatible bug fixes and documentation.
- Minor: additive stable APIs, new opt-in generation modes, or newly stable experimental behavior.
- Major: removal or incompatible change to stable public APIs.

Package-scoped implementation types are not API. Experimental options are documented explicitly and
may change in a minor release while preserving the deterministic default path.

## Tagging

After reviewing workflow artifacts:

1. Update `Docs/ReleaseNotes-Next.md` with the final version and date.
2. Create the signed/version tag manually.
3. Publish the notes and benchmark artifact with the GitHub release.
4. Verify a clean remote SwiftPM consumer against the published tag.
