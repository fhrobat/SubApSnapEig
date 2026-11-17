#!/bin/bash
# Run all the experiments of Section 8, one at a time, on an otherwise idle machine.
#
# Before each experiment the script waits until the 1-minute load average is
# below MAXLOAD, so that the timings are not polluted by other jobs. While the
# experiment runs, the load average is written every 30 s in
# results/<name>/load.log (it includes the NTHREADS threads of MATLAB itself),
# and the MATLAB output goes to results/<name>/run.log.
#
# Usage:  ./run_all.sh [experiment ...]
#   experiments: exponential differential pagerank kernel graph transformer theory
#   (default: all; 'theory' writes the data of Figures 1-3 in theory/)
# Environment variables:
#   MATLAB    MATLAB executable                       (default: matlab)
#   NTHREADS  number of computational threads of MATLAB (default: MATLAB's choice)
#   MAXLOAD   maximum load average before starting     (default: 2)

MATLAB=${MATLAB:-matlab}
MAXLOAD=${MAXLOAD:-2}
EXPERIMENTS=${*:-exponential differential pagerank kernel graph transformer theory}
cd "$(dirname "$0")"

for ex in $EXPERIMENTS; do
    mkdir -p results/$ex
    while (( $(awk -v m="$MAXLOAD" '{print ($1 > m)}' /proc/loadavg) )); do
        echo "$(date '+%F %T')  load $(cut -d' ' -f1 /proc/loadavg) > $MAXLOAD, waiting before $ex"
        sleep 60
    done

    case $ex in
        transformer) cmd="run_experiment_svd" ;;
        theory)      cmd="addpath theory; theory(0.8765); theory(1.008765)" ;;
        *)           cmd="run_experiment('$ex')" ;;
    esac
    if [ -n "$NTHREADS" ]; then cmd="maxNumCompThreads($NTHREADS); $cmd"; fi

    ( while true; do echo "$(date '+%T') $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 30; done ) \
        > results/$ex/load.log &
    monitor=$!
    echo "$(date '+%F %T')  start $ex (NTHREADS=${NTHREADS:-default})"
    { echo "host $(hostname), $(date '+%F %T'), NTHREADS=${NTHREADS:-default}"; uptime; } > results/$ex/run.log
    "$MATLAB" -batch "$cmd" >> results/$ex/run.log 2>&1
    kill $monitor
    echo "$(date '+%F %T')  end $ex"
done
