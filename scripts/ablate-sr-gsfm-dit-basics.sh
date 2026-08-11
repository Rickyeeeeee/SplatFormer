#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

gpu_id=${GPU_ID:-1}
ablation_root=${ABLATION_OUTPUT_ROOT:-/project2/ricky/experiments/0811-arch-abla/overfit_sr_gsfm_noemd_512}
launcher=scripts/overfit-sr-gsfm-noemd-on-objaverse.sh
features=means,opacities,features_dc,features_rest,scales,quats
common_args=(10000 10000 1000 1000 4 1 "$features")

run_ablation() {
    local name=$1
    local shuffle_eval=$2
    local drop_path=$3
    local turn_off_bn=$4
    local output_root="${ablation_root}/${name}"

    if [[ -e "$output_root" ]]; then
        echo "Refusing to overwrite existing ablation output: $output_root" >&2
        return 1
    fi

    echo "Starting $name: shuffle_eval=$shuffle_eval drop_path=$drop_path turn_off_bn=$turn_off_bn"
    OUTPUT_ROOT="$output_root" \
    MIX_SCHEDULE=fm-only \
    GPU_ID="$gpu_id" \
    PTV3_SHUFFLE_ORDERS=True \
    PTV3_SHUFFLE_ORDERS_EVAL="$shuffle_eval" \
    PTV3_DROP_PATH="$drop_path" \
    PTV3_TURN_OFF_BN="$turn_off_bn" \
        bash "$launcher" "${common_args[@]}"
}

# Each single-option run changes exactly one setting from the current baseline.
run_ablation deterministic_eval False 0.3 False
run_ablation no_drop_path True 0.0 False
run_ablation no_batch_norm True 0.3 True

# Combined run enables all three DiT-like settings.
run_ablation all_three False 0.0 True
