#!/bin/bash
#SBATCH --job-name=h3-autoresearch-100
#SBATCH --partition=h200
#SBATCH --nodelist=gpu-h200x8-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cj/h3-cuda/autoresearch-100/search-telemetry-%j.log

# The original search.sbatch and frozen worker/judge are run unchanged. The
# observer is staged outside seed/ and uses a separate CPU-only container.
set -euo pipefail
H3_STAGE=/var/tmp/h3-search-100
H3_SHARED=/home/cj/h3-cuda/autoresearch-100
H3_IMAGE=/data0/apptainer/pytorch_291_cu128.sif
H3_OUTBOX="$H3_SHARED/telemetry"
H3_OBSERVER="$H3_STAGE/telemetry/telemetry_observer.py"
H3_ARCHIVER="$H3_STAGE/telemetry/telemetry_archive.py"
H3_RUN_PID=
H3_OBSERVER_PID=
H3_CLEANED=0
H3_PRODUCER_PATTERN="$H3_STAGE/(seed|candidates)/"

mkdir -p "$H3_STAGE/telemetry" "$H3_OUTBOX"
install -m 0644 "$H3_SHARED/telemetry_observer.py" "$H3_OBSERVER"
install -m 0644 "$H3_SHARED/telemetry_archive.py" "$H3_ARCHIVER"
H3_CPU=(apptainer exec --bind /data0:/data0 --bind /var/tmp:/var/tmp "$H3_IMAGE" python3)
H3_OBSERVER_ARGS=(--stage "$H3_STAGE" --shared-results "$H3_SHARED/results"
    --outbox "$H3_OUTBOX" --session-id "${SLURM_JOB_ID:-manual}")

h3_observe() {
    "${H3_CPU[@]}" "$H3_OBSERVER" "$@" "${H3_OBSERVER_ARGS[@]}"
}

h3_drain_and_archive() {
    h3_observe drain --producer-stopped || return
    "${H3_CPU[@]}" "$H3_ARCHIVER" --stage "$H3_STAGE" --outbox "$H3_OUTBOX" \
        --destination "$H3_SHARED/telemetry-packages" --producer-stopped
}

h3_producers_present() {
    # Only test for processes referring to this frozen seed/candidate tree.
    # Do not print process command lines or kill another session's processes.
    pgrep -f "$H3_PRODUCER_PATTERN" >/dev/null
}

h3_cleanup() {
    local H3_REASON="$1"
    if [ "$H3_CLEANED" = 1 ]; then return; fi
    H3_CLEANED=1
    if [ -n "$H3_RUN_PID" ]; then
        # The producer was started in its own session. On signals/failure,
        # stop only that process group; never signal unrelated GPU jobs.
        kill -TERM -- "-$H3_RUN_PID" 2>/dev/null || true
        for H3_WAIT in $(seq 1 10); do
            if ! kill -0 -- "-$H3_RUN_PID" 2>/dev/null; then break; fi
            sleep 1
        done
        kill -KILL -- "-$H3_RUN_PID" 2>/dev/null || true
        wait "$H3_RUN_PID" 2>/dev/null || true
    fi
    if [ -n "$H3_OBSERVER_PID" ]; then
        kill -TERM "$H3_OBSERVER_PID" 2>/dev/null || true
        wait "$H3_OBSERVER_PID" 2>/dev/null || true
    fi
    if h3_producers_present; then
        echo "Telemetry drain deferred: a producer still refers to the frozen tree ($H3_REASON)."
        echo "Run drain --producer-stopped after the producer group has stopped and before the next resume."
        return 1
    fi
    # Full evidence packaging runs only after producers stop, outside inference.
    # Neither observer nor archive helper claims an exact historical stop time.
    h3_drain_and_archive
}

trap 'H3_EXIT=$?; h3_cleanup exit || true; exit "$H3_EXIT"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# A predecessor can have been SIGKILLed, including its observer. Recover its
# interrupted metadata before the original worker renames the retry directory.
if h3_producers_present; then
    echo "Refusing resume/drain while a previous producer still uses $H3_STAGE."
    exit 1
fi
h3_drain_and_archive

setsid bash "$H3_SHARED/search.sbatch" &
H3_RUN_PID=$!
# Launch the container directly so this PID owns the observer process, rather
# than a background shell function that can leave the lock holder orphaned.
"${H3_CPU[@]}" "$H3_OBSERVER" watch --worker-pid "$H3_RUN_PID" --poll-seconds 2 \
    "${H3_OBSERVER_ARGS[@]}" &
H3_OBSERVER_PID=$!
set +e
H3_FIRST_EXIT_PID=
wait -n -p H3_FIRST_EXIT_PID "$H3_RUN_PID" "$H3_OBSERVER_PID"
H3_EXIT=$?
set -e
if [ "$H3_FIRST_EXIT_PID" = "$H3_OBSERVER_PID" ]; then
    if kill -0 "$H3_RUN_PID" 2>/dev/null; then
        echo "Telemetry observer exited before producer; stopping the owned producer group."
        H3_EXIT=1
    else
        set +e
        wait "$H3_RUN_PID"
        H3_EXIT=$?
        set -e
    fi
fi
h3_cleanup "producer exit $H3_EXIT" || H3_EXIT=1
exit "$H3_EXIT"
