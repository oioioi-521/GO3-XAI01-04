#!/usr/bin/env bash
# Run inside the vendor container, from the repository root.
set -euo pipefail
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PWD/results/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME="$PWD/.cache/torch"
devices=${XAI_DEVICES:?Set XAI_DEVICES to the assigned physical device IDs}
IFS=',' read -r -a cards <<< "$devices"
workers=${#cards[@]}
if (( workers < 2 || workers > 8 )); then
    printf 'This scaling suite requires 2–8 assigned devices.\n' >&2
    exit 2
fi
images=$((workers * 2))
root="results/dcu-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$root"
printf '%s\n' "$root" > results/LATEST_DCU_BENCHMARK
python -m pip freeze > "$root/environment.txt"
run() {
    local label=$1
    shift
    python scripts/benchmark_accelerators.py "$@" --output "$root/$label"
}
# Actual pretrained networks and real frozen debug images; full forward/backward.
for model in resnet50 densenet121 vgg16; do
    for method in gradcam ig occlusion kernelshap rise; do
        run "gate-$model-$method" --method "$method" --model "$model" \
            --images 1 --devices "${cards[0]}" --morf
    done
done
for method in kernelshap rise; do
    run "$method-1card" --method "$method" --images "$images" --devices "${cards[0]}"
    run "$method-${workers}card" --method "$method" --images "$images" \
        --devices "$devices" --workers "$workers"
done
# Same KernelSHAP coalitions/budget, larger forward batch. Keep maps to check drift.
run kernelshap-b128-1card --method kernelshap --batch 128 --images "$images" --devices "${cards[0]}"
run "kernelshap-b128-${workers}card" --method kernelshap --batch 128 --images "$images" \
    --devices "$devices" --workers "$workers"
# Bounded sustained workload, all reserved devices, no detached orphan workers.
run stress --method kernelshap --batch 128 --images "$images" --repeats 4 \
    --devices "$devices" --workers "$workers"
run sustained-stress --method kernelshap --batch 128 --images "$images" --repeats 40 \
    --devices "$devices" --workers "$workers"
printf 'complete\n' > "$root/COMPLETE"
printf 'All benchmark runs complete: %s\n' "$root"
