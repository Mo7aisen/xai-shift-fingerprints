#!/bin/bash
# Launch GPU jobs once enough free memory is available.

set -euo pipefail

while true; do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [ "${FREE_MEM}" -gt 36000 ]; then
        echo "GPU free (${FREE_MEM} MiB), launching jobs..."
        python scripts/run_energy_ood_baseline.py --batch-size 16 --num-workers 4
        python scripts/run_rise_segmentation.py --max-samples 200 --n-masks 1000 --mask-batch 32
        python scripts/generate_publication_figures_final.py
        break
    fi
    echo "Waiting for GPU... free=${FREE_MEM} MiB"
    sleep 300
done
