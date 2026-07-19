#!/bin/bash
# E26: skip vs substitute A/B (the E25 survivor), then the n>=3 repeat bench.
# Banked comparisons: battery_<dom>_m0 (anchors), battery_<dom>_m0.5
# (substitute mode). New: skip mode at m0.5 all 5 domains + m0.25 code/reason.
# PREDICTION (logged pre-run): skip reduces the m0.5 reasoning damage
# (substitute +9.7% warm) because a skip injects nothing where a substitute
# injects a wrong expert's signal at real weight; effect at m0.25 ~ noise
# (swaps too rare). skip_fills counter must be >0 in every skip rung.
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
CODE=$(sed -n 's/^CODE=//p' /dev/null); # passages inlined below
source /dev/stdin <<'PASSAGES'
CODE='def quicksort(items):
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    smaller = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    larger = [x for x in items if x > pivot]
    return quicksort(smaller) + equal + quicksort(larger)

# The recursion depth is bounded by the number of distinct pivots chosen. On an
# already-sorted input with a middle pivot the partitions stay balanced, so the
# call tree has logarithmic height and the total work is n log n comparisons.'
REASON='A train leaves the station at nine in the morning traveling forty miles per hour.
A second train leaves the same station two hours later traveling sixty miles per hour
on a parallel track in the same direction. To find when the faster train catches the
slower one, note the first train has a head start of eighty miles. The gap closes at
twenty miles per hour, the difference of their speeds, so it takes four hours after
the second train departs. Therefore the faster train catches up at three in the afternoon.'
CHATDOM='I have been having trouble sleeping for the past few weeks and I am not sure why.
Some nights I lie awake for hours with my thoughts racing about work, and other nights I
fall asleep quickly but wake up at three and cannot drift off again. I have tried cutting
back on coffee after noon and it helped a little, but the problem has not gone away. What
practical steps would you suggest I try before considering anything more drastic?'
MULTI='La ciudad amaneció cubierta por una niebla espesa que apenas dejaba ver el final de
la calle. Los comerciantes abrían sus puertas en silencio, y el aroma del pan recién
hecho se mezclaba con la humedad del aire. Poco a poco, la gente salía de sus casas y el
murmullo cotidiano regresaba a las plazas, como si la niebla nunca hubiera existido.'
PROSE='The old lighthouse keeper climbed the spiral stairs each evening at dusk, carrying
the small brass lamp his father had left him. The sea below churned against the rocks
with a patience older than any human memory, and he had long since stopped fearing it.
What he feared instead was the silence of the tower on nights when no ships passed.'
PASSAGES

nll() { # domain margin skipw text
    pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
    [ "${pct:-99}" -lt 40 ] && sleep 45 || true
    echo "== $1 m=$2 skip=$3 =="
    env LLMSTREAM_NLL=1 LLMSTREAM_SLOTS=8 LLMSTREAM_MARGIN=$2 LLMSTREAM_PREFETCH=0 \
        LLMSTREAM_SKIP_W=$3 \
        ./csrc/stream_run "$MODEL" 0 "$4" 1 > "results/skipab_$1_m$2_w$3.txt" 2>&1
    grep -E 'avg_nll|skip_fills|router_agreement' "results/skipab_$1_m$2_w$3.txt" | sed 's/^/  /'
}
for d in CODE REASON CHATDOM MULTI PROSE; do
    eval t=\$$d
    nll "$(echo $d | tr A-Z a-z)" 0.5 0.25 "$t"
done
nll code   0.25 0.25 "$CODE"
nll reason 0.25 0.25 "$REASON"
echo "SKIP A/B DONE - starting repeat bench"
bash scripts/bench_repeat.sh 3
