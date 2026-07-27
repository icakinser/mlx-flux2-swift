#!/usr/bin/env bash
#
# Release benchmark for the library's major generation paths.
#
# Usage:
#   FLUX2_REPO=/path/to/FLUX.2-klein-4B Scripts/bench.sh
#   FLUX2_REPO=... BENCH_SIZE=256 BENCH_OUTPUT=/tmp/results.md Scripts/bench.sh

set -euo pipefail

cd "$(dirname "$0")/.."

: "${FLUX2_REPO:?Set FLUX2_REPO to a FLUX.2-klein-4B snapshot}"

SIZE="${BENCH_SIZE:-512}"
STEPS="${BENCH_STEPS:-4}"
OUTPUT="${BENCH_OUTPUT:-dev/bench/BASELINE.md}"
PROMPT="${BENCH_PROMPT:-a red bicycle leaning against a stone wall, golden hour}"
SEED="${BENCH_SEED:-42}"
BIN=".build/release/flux2kit-cli"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/flux2kit-bench.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

swift build -c release
if [[ ! -f ".build/default.metallib" ]]; then
    Scripts/setup_metallib.sh
fi

mkdir -p "$(dirname "$OUTPUT")"
RESULTS="$WORK/results.tsv"
printf "case\tload\ttext\tdenoise\tvae\timage\ttotal\n" > "$RESULTS"

extract_ms() {
    local pattern="$1"
    local log="$2"
    awk -v pattern="$pattern" '
        index($0, pattern) {
            value=$0
            sub(/^[^[]*\[/, "", value)
            sub(/ms\].*$/, "", value)
            gsub(/[[:space:]]/, "", value)
            printf "%s", value
            exit
        }
    ' "$log"
}

run_case() {
    local name="$1"
    shift
    local log="$WORK/$name.log"
    echo "==> $name"
    "$BIN" \
        --repo "$FLUX2_REPO" \
        --prompt "$PROMPT" \
        --width "$SIZE" \
        --height "$SIZE" \
        --steps "$STEPS" \
        --guidance 1.0 \
        --seed "$SEED" \
        --verbose \
        --output "$WORK/$name.png" \
        "$@" > "$log" 2>&1
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$name" \
        "$(extract_ms "Pipeline load" "$log")" \
        "$(extract_ms "Text encode:" "$log")" \
        "$(extract_ms "Denoise total" "$log")" \
        "$(extract_ms "VAE decode" "$log")" \
        "$(extract_ms "To image" "$log")" \
        "$(extract_ms "TOTAL" "$log")" >> "$RESULTS"
}

# Two eager runs distinguish first-run kernel/cache costs from steady state.
run_case "eager-first" --sampler euler
run_case "eager-warm" --sampler euler
run_case "compile" --compile --sampler euler
run_case "heun" --sampler heun

{
    echo "# Flux2Kit Release Benchmark"
    echo
    echo "**Generated:** $(date -u '+%Y-%m-%d %H:%M:%S UTC')  "
    echo "**Host:** $(uname -m), $(sw_vers -productName) $(sw_vers -productVersion)  "
    echo "**Workload:** ${SIZE}×${SIZE}, ${STEPS} steps, guidance 1.0, seed ${SEED}  "
    echo "**Build:** Swift release"
    echo
    echo "| Path | Pipeline load | Text encode | Denoise | VAE decode | To image | Generation total |"
    echo "|---|---:|---:|---:|---:|---:|---:|"
    awk -F '\t' 'NR > 1 {
        printf "| `%s` | %s ms | %s ms | %s ms | %s ms | %s ms | %s ms |\n",
            $1, $2, $3, $4, $5, $6, $7
    }' "$RESULTS"
    echo
    echo "The CLI is a measurement harness. App integrations should benchmark their own resolution,"
    echo 'residency, and repetition pattern. `keepResident` favors repeated generation;'
    echo '`unloadAfterUse` favors minimum between-call memory.'
} > "$OUTPUT"

echo "Wrote $OUTPUT"
