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
printf "case\tload\ttext\tdenoise\tvae\timage\ttotal\twall\n" > "$RESULTS"

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
    /usr/bin/time -p "$BIN" \
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
    local wall
    wall="$(awk '/^real / { printf "%.1f", $2 * 1000; exit }' "$log")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$name" \
        "$(extract_ms "Pipeline load" "$log")" \
        "$(extract_ms "Text encode:" "$log")" \
        "$(extract_ms "Denoise total" "$log")" \
        "$(extract_ms "VAE decode" "$log")" \
        "$(extract_ms "To image" "$log")" \
        "$(extract_ms "TOTAL" "$log")" \
        "$wall" >> "$RESULTS"
}

# Two eager runs distinguish first-run kernel/cache costs from steady state.
run_case "eager-first" --sampler euler
run_case "eager-warm" --sampler euler
run_case "eager-eval4" --sampler euler --eval-freq 4
run_case "compile" --compile --sampler euler
run_case "heun" --sampler heun
run_case "img2img" \
    --img2img --source "Tests/Flux2KitTests/Fixtures/ref_bike_s42.png" \
    --strength 0.6 --sampler euler
run_case "inpaint" \
    --source "Tests/Flux2KitTests/Fixtures/ref_bike_s42.png" \
    --edit "a wooden crate against the stone wall" \
    --mask-box "136,152,240,224" --strength 0.85 --sampler euler
run_case "outpaint" \
    --source "Tests/Flux2KitTests/Fixtures/ref_bike_s42.png" \
    --outpaint "32,32,32,32" --sampler euler
run_case "reference" \
    --input "Tests/Flux2KitTests/Fixtures/ref_bike_s42.png" --sampler euler

{
    echo "# Flux2Kit Release Benchmark"
    echo
    echo "**Generated:** $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "**Host:** $(uname -m), $(sw_vers -productName) $(sw_vers -productVersion)"
    echo "**Workload:** ${SIZE}×${SIZE}, ${STEPS} steps, guidance 1.0, seed ${SEED}"
    echo "**Build:** Swift release"
    echo
    echo "| Path | Pipeline load | Text encode | Denoise | VAE decode | To image | Generation total | Wall time |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|"
    awk -F '\t' 'NR > 1 {
        for (i=2; i<=8; i++) if ($i == "") $i = "—"
        printf "| `%s` | %s ms | %s ms | %s ms | %s ms | %s ms | %s ms | %s ms |\n",
            $1, $2, $3, $4, $5, $6, $7, $8
    }' "$RESULTS"
    echo
    echo "## Policy review"
    echo
    awk -F '\t' '
        $1 == "eager-warm" { eager=$7 }
        $1 == "eager-eval4" { eval4=$7 }
        $1 == "compile" { compiled=$7 }
        END {
            if (eager > 0 && compiled > 0)
                printf "- Compile versus warm eager: %+.1f%% generation time.\n", (compiled/eager-1)*100
            if (eager > 0 && eval4 > 0)
                printf "- Eval-frequency 4 versus warm eager: %+.1f%% generation time.\n", (eval4/eager-1)*100
        }
    ' "$RESULTS"
    echo "- Defaults change only when a quality-gated path improves its target workload by at least 5%."
    echo "- Reference-conditioning optimizations require at least 10%."
    echo
    echo "The CLI is a measurement harness. App integrations should benchmark their own resolution,"
    echo 'residency, and repetition pattern. `keepResident` favors repeated generation;'
    echo '`unloadAfterUse` favors minimum between-call memory.'
} > "$OUTPUT"

echo "Wrote $OUTPUT"
