#!/bin/bash
# Domain battery (audit finding 1): the m0.25 "no-compromise" claim rested on
# ONE memorized 144-tok prose passage. This scores teacher-forced NLL across
# 5 domains x margins {0, 0.25, 0.5} with harmony CHAT + warm-split, so the
# quality claim is settled on chat/code/reasoning/multilingual, not just prose.
# NO chat template here: gpt-oss is assistant-loss-trained, so teacher-forcing
# USER-turn text measures uncalibrated territory (measured: ppl 2166-18k on
# plain text through the template vs 2.8-3 raw). Raw scoring matches the
# original anchor methodology; CHAT belongs to generation evals only.
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf

# fixed passages, ~200-400 tok each, NON-famous where possible.
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

nll() { # label margin text
    pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
    [ "${pct:-99}" -lt 40 ] && sleep 45 || true
    echo "== $1 m=$2 =="
    LLMSTREAM_NLL=1 LLMSTREAM_SLOTS=8 LLMSTREAM_MARGIN=$2 LLMSTREAM_PREFETCH=0 \
        ./csrc/stream_run "$MODEL" 0 "$3" 1 > "results/battery_$1_m$2.txt" 2>&1
    grep -E 'avg_nll|router_agreement' "results/battery_$1_m$2.txt" | sed "s/^/  /"
}

for m in 0 0.25 0.5; do
    nll code   $m "$CODE"
    nll reason $m "$REASON"
    nll chat   $m "$CHATDOM"
    nll multi  $m "$MULTI"
    nll prose  $m "$PROSE"
done
echo "BATTERY COMPLETE"
