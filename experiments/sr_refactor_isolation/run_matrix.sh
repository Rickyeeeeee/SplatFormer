#!/usr/bin/env bash
# Standalone launcher for the SR refactor isolation matrix.

set -euo pipefail

SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SUITE_DIR}/../.." && pwd)

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/project2/ricky/experiments/0822-refactor-isolation}
SCENE_NAME=${SCENE_NAME:-3e288ee8aced4a0797e66d53536112b1}
CONFIRM_SCENE_NAME=${CONFIRM_SCENE_NAME:-914d43018c9b4cd681e31d56cb889563}
NATIVE_DATA_ROOT=${NATIVE_DATA_ROOT:-/project/ricky/splatformer-sr-data-new}
LEGACY_DATA_ROOT=${LEGACY_DATA_ROOT:-/project2/ricky/splatformer-data/test-set-512/objaverse}
CONDA_BIN=${CONDA_BIN:-conda}
SR_CONDA_ENV=${SR_CONDA_ENV:-3dgs-sr}
NERFSTUDIO_CONDA_ENV=${NERFSTUDIO_CONDA_ENV:-splatformer}
SR_PYTHON_BIN=${SR_PYTHON_BIN:-python}
NS_TRAIN_BIN=${NS_TRAIN_BIN:-ns-train}
GPU_ID=${GPU_ID:-0}
VERIFY_RENDER=${VERIFY_RENDER:-1}
MAX_RENDER_VIEWS=${MAX_RENDER_VIEWS:-0}
DRY_RUN=${DRY_RUN:-0}
RESTART_INCOMPLETE=${RESTART_INCOMPLETE:-0}
CONTINUE_AFTER_FAILURE=${CONTINUE_AFTER_FAILURE:-0}

LEGACY_COMMIT=f77538b
POINTCEPT_COMMIT=bbb4102
INPUT_RESOLUTION=128
TARGET_RESOLUTION=512
INPUT_FACTOR=4
TARGET_FACTOR=1
FULL_STEPS=3000
FULL_MATCHING_STEPS=2000
SMOKE_STEPS=20
SMOKE_MATCHING_STEPS=20
MATCHING_IMAGES=32
ATTRIBUTES=means,scales,opacities,quats,features_dc,features_rest
CONTROL_REFERENCE_PSNR=${CONTROL_REFERENCE_PSNR:-27.0791}

LEGACY_EXPORT="${EXPERIMENT_ROOT}/work/legacy_source_${LEGACY_COMMIT}"
PRODUCER_ROOT="${EXPERIMENT_ROOT}/producers/${SCENE_NAME}/nerfstudio_native"
FIXTURE_ROOT="${EXPERIMENT_ROOT}/fixtures/${SCENE_NAME}"
METADATA_ROOT="${EXPERIMENT_ROOT}/metadata/${SCENE_NAME}"
SCENE_LIST="${METADATA_ROOT}/scenes.csv"

native_colmap_scene() {
    local resolution=$1
    printf '%s/test-set/objaverse/%s/colmap/%s' \
        "${NATIVE_DATA_ROOT}" "${resolution}" "${SCENE_NAME}"
}

native_gsplat_checkpoint() {
    local resolution=$1
    printf '%s/test-set/objaverse/%s/gsplat/%s/ckpts/ckpt_14999_rank0.pt' \
        "${NATIVE_DATA_ROOT}" "${resolution}" "${SCENE_NAME}"
}

native_gsplat_stats() {
    local resolution=$1
    printf '%s/test-set/objaverse/%s/gsplat/%s/stats/val_step14999.json' \
        "${NATIVE_DATA_ROOT}" "${resolution}" "${SCENE_NAME}"
}

producer_scene_dir() {
    local resolution=$1
    printf '%s/%s/nerfstudio/%s/splatfacto' \
        "${PRODUCER_ROOT}" "${resolution}" "${SCENE_NAME}"
}

producer_checkpoint() {
    local resolution=$1
    printf '%s/nerfstudio_models/step-000015001.ckpt' \
        "$(producer_scene_dir "${resolution}")"
}

producer_camera() {
    local resolution=$1
    printf '%s/camera_for-3d-denoise.pkl' \
        "$(producer_scene_dir "${resolution}")"
}

require_path() {
    local path=$1
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
}

write_scene_list() {
    mkdir -p "${METADATA_ROOT}"
    if [[ -f "${SCENE_LIST}" ]]; then
        local content
        content=$(tr -d '\r' < "${SCENE_LIST}")
        if [[ "${content}" == $'scene_id\n'"${SCENE_NAME}" ]]; then
            return
        fi
        if [[ "${content}" != "${SCENE_NAME}" ]]; then
            echo "Refusing to replace mismatched scene list: ${SCENE_LIST}" >&2
            exit 1
        fi
        echo "[migrate] Adding scene_id header to: ${SCENE_LIST}"
    fi
    local temporary_scene_list="${SCENE_LIST}.tmp.$$"
    printf 'scene_id\n%s\n' "${SCENE_NAME}" > "${temporary_scene_list}"
    mv "${temporary_scene_list}" "${SCENE_LIST}"
}

ensure_clean_export() {
    local marker="${LEGACY_EXPORT}/.sr_refactor_isolation_export"
    if [[ -f "${marker}" ]]; then
        local marker_main=""
        local marker_pointcept=""
        while IFS="=" read -r key value; do
            case "${key}" in
                main) marker_main="${value}" ;;
                pointcept) marker_pointcept="${value}" ;;
            esac
        done < "${marker}"
        if [[ "${marker_main}" != "${LEGACY_COMMIT}" || \
              "${marker_pointcept}" != "${POINTCEPT_COMMIT}" ]]; then
            echo "Legacy export marker does not match requested commits: ${marker}" >&2
            exit 1
        fi
        echo "[skip] Clean legacy export already exists: ${LEGACY_EXPORT}"
        return
    fi
    if [[ -d "${LEGACY_EXPORT}" ]] && [[ -n "$(find "${LEGACY_EXPORT}" -mindepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite unmarked legacy export: ${LEGACY_EXPORT}" >&2
        exit 1
    fi
    mkdir -p "${LEGACY_EXPORT}"
    echo "[prepare] Exporting ${LEGACY_COMMIT} to ${LEGACY_EXPORT}"
    git -C "${REPO_ROOT}" archive "${LEGACY_COMMIT}" | tar -x -C "${LEGACY_EXPORT}"
    mkdir -p "${LEGACY_EXPORT}/Pointcept"
    git -C "${REPO_ROOT}/Pointcept" archive "${POINTCEPT_COMMIT}" | \
        tar -x -C "${LEGACY_EXPORT}/Pointcept"
    printf 'main=%s\npointcept=%s\n' "${LEGACY_COMMIT}" "${POINTCEPT_COMMIT}" > "${marker}"
}

train_nerfstudio_resolution() {
    local resolution=$1
    local colmap_scene
    local checkpoint
    local camera
    local output_dir
    local producer_dir
    colmap_scene=$(native_colmap_scene "${resolution}")
    checkpoint=$(producer_checkpoint "${resolution}")
    camera=$(producer_camera "${resolution}")
    output_dir="${PRODUCER_ROOT}/${resolution}/nerfstudio"
    producer_dir="${PRODUCER_ROOT}/${resolution}"
    require_path "${colmap_scene}/images"
    if [[ -f "${checkpoint}" && -f "${camera}" ]]; then
        echo "[skip] Nerfstudio ${resolution} producer exists: ${checkpoint}"
        return
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "Refusing to overwrite incomplete Nerfstudio producer: ${output_dir}" >&2
        exit 1
    fi
    mkdir -p "${producer_dir}"
    local command=(
        "${CONDA_BIN}" run --no-capture-output -n "${NERFSTUDIO_CONDA_ENV}"
        "${NS_TRAIN_BIN}" splatfacto
        --logging.local-writer.enable=False
        --logging.profiler=none
        "--pipeline.datamanager.data=${colmap_scene}"
        --pipeline.model.sh_degree=1
        --pipeline.save_img=False
        --pipeline.datamanager.images-on-gpu=True
        --pipeline.datamanager.cache-images=gpu
        --pipeline.model.stop-split-at=10000
        --test_after_train True
        "--output_dir=${output_dir}"
        "--experiment-name=${SCENE_NAME}"
        --relative-model-dir=nerfstudio_models
        --vis viewer
        --steps_per_eval_image=100000
        --steps_per_eval_all_images=1000000
        --max_num_iterations=30000
        --save_only_latest_checkpoint False
        --steps_per_save=100000
        --save_last_checkpoint True
        --early_stop_steps=15000
        --save_only_gs_params True
        --viewer.quit-on-train-completion True
        colmap
        --downscale_factor=1
        --downscale-rounding-mode=floor
        --load_3D_points True
        --eval-mode all
        --auto_scale_poses=False
        --orientation_method=none
        --center_method=none
        --load_bbox True
        --num_points_from_bbox 50000
        --assume_colmap_world_coordinate_convention False
    )
    echo "[producer] Nerfstudio native ${resolution} for ${SCENE_NAME}"
    printf '%q ' env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "${command[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == 1 ]]; then
        return
    fi
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 | \
        tee "${producer_dir}/producer.log"
    require_path "${checkpoint}"
    require_path "${camera}"
}

prepare_fixture() {
    local name=$1
    local source_format=$2
    local fixture="${FIXTURE_ROOT}/${name}"
    if [[ -f "${fixture}/verification.json" ]]; then
        echo "[skip] Verified fixture exists: ${fixture}"
        return
    fi
    local input_checkpoint
    local target_checkpoint
    local extra=()
    if [[ "${source_format}" == nerfstudio ]]; then
        input_checkpoint=$(producer_checkpoint "${INPUT_RESOLUTION}")
        target_checkpoint=$(producer_checkpoint "${TARGET_RESOLUTION}")
    else
        input_checkpoint=$(native_gsplat_checkpoint "${INPUT_RESOLUTION}")
        target_checkpoint=$(native_gsplat_checkpoint "${TARGET_RESOLUTION}")
        extra+=(
            "--input-reference-stats=$(native_gsplat_stats "${INPUT_RESOLUTION}")"
            "--target-reference-stats=$(native_gsplat_stats "${TARGET_RESOLUTION}")"
        )
    fi
    require_path "${input_checkpoint}"
    require_path "${target_checkpoint}"
    require_path "$(producer_camera "${INPUT_RESOLUTION}")"
    require_path "$(producer_camera "${TARGET_RESOLUTION}")"
    local command=(
        "${CONDA_BIN}" run --no-capture-output -n "${SR_CONDA_ENV}"
        "${SR_PYTHON_BIN}" "${SUITE_DIR}/prepare_compat_fixture.py"
        "--scene-name=${SCENE_NAME}"
        "--source-format=${source_format}"
        "--input-checkpoint=${input_checkpoint}"
        "--target-checkpoint=${target_checkpoint}"
        "--input-camera-metadata=$(producer_camera "${INPUT_RESOLUTION}")"
        "--target-camera-metadata=$(producer_camera "${TARGET_RESOLUTION}")"
        "--input-colmap-scene=$(native_colmap_scene "${INPUT_RESOLUTION}")"
        "--target-colmap-scene=$(native_colmap_scene "${TARGET_RESOLUTION}")"
        "--output-root=${fixture}"
        "--input-factor=${INPUT_FACTOR}"
        "--target-factor=${TARGET_FACTOR}"
        "--max-render-views=${MAX_RENDER_VIEWS}"
        "${extra[@]}"
    )
    if [[ "${VERIFY_RENDER}" == 1 ]]; then
        command+=(--verify-render "--render-device=cuda")
    fi
    echo "[fixture] ${name}: ${source_format}"
    printf '%q ' env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "${command[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == 0 ]]; then
        env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" | \
            tee "${fixture}.prepare.log"
    fi
}

prepare_all() {
    mkdir -p "${EXPERIMENT_ROOT}" "${FIXTURE_ROOT}" "${METADATA_ROOT}"
    write_scene_list
    ensure_clean_export
    train_nerfstudio_resolution "${INPUT_RESOLUTION}"
    train_nerfstudio_resolution "${TARGET_RESOLUTION}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        echo "[dry-run] Fixture creation requires producer outputs and is omitted."
        return
    fi
    prepare_fixture E1 nerfstudio
    prepare_fixture E2 gsplat
}

run_root_for_mode() {
    local mode=$1
    if [[ "${mode}" == smoke ]]; then
        printf '%s/smoke_runs/%s' "${EXPERIMENT_ROOT}" "${SCENE_NAME}"
    else
        printf '%s/runs/%s' "${EXPERIMENT_ROOT}" "${SCENE_NAME}"
    fi
}

record_run() {
    local run_dir=$1
    local mode=$2
    local name=$3
    local consumer=$4
    local backend=$5
    shift 5
    printf '%q ' "$@" > "${run_dir}/launch_command.txt"
    printf '\n' >> "${run_dir}/launch_command.txt"
    printf '{\n  "version": 1,\n  "run": "%s",\n  "mode": "%s",\n  "scene_name": "%s",\n  "consumer": "%s",\n  "backend": "%s",\n  "seed": 42,\n  "attributes": "%s",\n  "matching_images_per_step": %s\n}\n' \
        "${name}" "${mode}" "${SCENE_NAME}" "${consumer}" "${backend}" \
        "${ATTRIBUTES}" "${MATCHING_IMAGES}" > "${run_dir}/run_manifest.json"
}

prepare_run_dir() {
    local run_dir=$1
    local cache_dir=$2
    local mode=$3
    local name=$4
    if [[ -f "${run_dir}/eval_final/metrics.json" ]]; then
        echo "[skip] Completed run exists: ${run_dir}"
        return 1
    fi
    if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
        if [[ "${RESTART_INCOMPLETE}" != 1 ]]; then
            echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
            echo "Re-run with RESTART_INCOMPLETE=1 to archive it and restart cleanly." >&2
            exit 1
        fi
        local archive_dir
        archive_dir="${EXPERIMENT_ROOT}/interrupted/${mode}/${SCENE_NAME}/${name}/$(date -u +%Y%m%dT%H%M%SZ)-$$"
        mkdir -p "${archive_dir}"
        mv "${run_dir}" "${archive_dir}/run"
        if [[ -d "${cache_dir}" ]] && [[ -n "$(find "${cache_dir}" -mindepth 1 -print -quit)" ]]; then
            mv "${cache_dir}" "${archive_dir}/matching_cache"
        fi
        echo "[archive] Incomplete ${name} run moved to: ${archive_dir}"
    fi
    mkdir -p "${run_dir}"
    return 0
}

legacy_roots_for_run() {
    local name=$1
    if [[ "${name}" == E0 ]]; then
        printf '%s\n%s\n' "${LEGACY_DATA_ROOT}/nerfstudio" "${LEGACY_DATA_ROOT}/colmap"
    elif [[ "${name}" == E1 ]]; then
        printf '%s\n%s\n' "${FIXTURE_ROOT}/E1/nerfstudio" "${FIXTURE_ROOT}/E1/colmap"
    else
        printf '%s\n%s\n' "${FIXTURE_ROOT}/E2/nerfstudio" "${FIXTURE_ROOT}/E2/colmap"
    fi
}

run_legacy() {
    local mode=$1
    local name=$2
    local steps=$3
    local matching_steps=$4
    local run_dir=$5
    local cache_dir=$6
    local roots
    mapfile -t roots < <(legacy_roots_for_run "${name}")
    local command=(
        "${CONDA_BIN}" run --no-capture-output -n "${SR_CONDA_ENV}"
        "${SR_PYTHON_BIN}" overfit-sr-mse.py
        "--output_dir=${run_dir}"
        "--scene_name=${SCENE_NAME}"
        --alignment=fit_lr_to_hr
        --attribute_init=aligned
        "--input_factor=${INPUT_FACTOR}"
        "--target_factor=${TARGET_FACTOR}"
        --post_activate_loss=true
        "--alignment_cache_root=${cache_dir}"
        --force_alignment_fit=true
        --gin_file=configs/model/ptv3.gin
        --gin_file=configs/overfit/sr_mse.gin
        --gin_param="FeaturePredictor.output_features_type='res'"
        --gin_param="FeaturePredictor.max_scale_normalized=0.01"
        --gin_param="FeaturePredictor.output_features=['means','scales','opacities','quats','features_dc','features_rest']"
        "--gin_param=total_steps=${steps}"
        "--gin_param=training.save_interval=${steps}"
        "--gin_param=training.eval_interval=${steps}"
        "--gin_param=training.log_image_interval=${steps}"
        "--gin_param=matching_total_steps=${matching_steps}"
        "--gin_param=matching_fit.image_per_step=${MATCHING_IMAGES}"
        --gin_param="matching_fit.image_l1_loss_weight=1.0"
        --gin_param="matching_fit.lpips_loss_weight=1.0"
        --gin_param="set_seed.seed=42"
        --gin_param="train_dataset/SplatFactoMultiLevelDataset.factors=[1,4]"
        "--gin_param=train_dataset/SplatFactoMultiLevelDataset.nerfstudio_folder='${roots[0]}'"
        "--gin_param=train_dataset/SplatFactoMultiLevelDataset.colmap_folder='${roots[1]}'"
    )
    record_run "${run_dir}" "${mode}" "${name}" "${LEGACY_COMMIT}" legacy "${command[@]}"
    echo "[${mode}] ${name}: clean ${LEGACY_COMMIT} legacy consumer"
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "${command[@]}"
        printf '\n'
        return
    fi
    (
        cd "${LEGACY_EXPORT}"
        env TORCH_CUDNN_V8_API_DISABLED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            "${command[@]}"
    ) 2>&1 | tee "${run_dir}/launcher.log"
}

run_current() {
    local mode=$1
    local name=$2
    local steps=$3
    local matching_steps=$4
    local run_dir=$5
    local cache_dir=$6
    local backend=gsplat_native
    local environment=(
        TORCH_CUDNN_V8_API_DISABLED=1
        "CUDA_VISIBLE_DEVICES=${GPU_ID}"
        SR_ABLATION_BACKEND=gsplat_native
    )
    if [[ "${name}" == E4 ]]; then
        backend=nerfstudio_factor
        environment=(
            TORCH_CUDNN_V8_API_DISABLED=1
            "CUDA_VISIBLE_DEVICES=${GPU_ID}"
            SR_ABLATION_BACKEND=nerfstudio_factor
            "SR_ABLATION_NERFSTUDIO_ROOT=${FIXTURE_ROOT}/E2/nerfstudio"
            "SR_ABLATION_COLMAP_ROOT=${FIXTURE_ROOT}/E2/colmap"
            "SR_ABLATION_INPUT_FACTOR=${INPUT_FACTOR}"
            "SR_ABLATION_TARGET_FACTOR=${TARGET_FACTOR}"
            "SR_ABLATION_INPUT_RESOLUTION=${INPUT_RESOLUTION}"
            "SR_ABLATION_TARGET_RESOLUTION=${TARGET_RESOLUTION}"
        )
    fi
    local command=(
        "${CONDA_BIN}" run --no-capture-output -n "${SR_CONDA_ENV}"
        "${SR_PYTHON_BIN}" "${SUITE_DIR}/overfit_backend_ablation.py"
        "--output_dir=${run_dir}"
        "--scene_name=${SCENE_NAME}"
        --alignment=fit_lr_to_hr
        --attribute_init=aligned
        "--input_resolution=${INPUT_RESOLUTION}"
        "--target_resolution=${TARGET_RESOLUTION}"
        --post_activate_loss=true
        "--matching_cache_root=${cache_dir}"
        --force_matching_fit=true
        --gin_file="${REPO_ROOT}/configs/model/ptv3.gin"
        --gin_file="${REPO_ROOT}/configs/overfit/sr_mse.gin"
        --gin_file="${REPO_ROOT}/configs/dataset/objaverse-sr.gin"
        --gin_param="dataset_root='${NATIVE_DATA_ROOT}'"
        --gin_param="test_scene_list='${SCENE_LIST}'"
        --gin_param="FeaturePredictor.output_features_type='res'"
        --gin_param="FeaturePredictor.max_scale_normalized=0.01"
        --gin_param="FeaturePredictor.output_features=['means','scales','opacities','quats','features_dc','features_rest']"
        "--gin_param=total_steps=${steps}"
        "--gin_param=training.save_interval=${steps}"
        "--gin_param=training.eval_interval=${steps}"
        "--gin_param=training.log_image_interval=${steps}"
        "--gin_param=matching_total_steps=${matching_steps}"
        "--gin_param=matching_fit.image_per_step=${MATCHING_IMAGES}"
        --gin_param="matching_fit.image_l1_loss_weight=1.0"
        --gin_param="matching_fit.lpips_loss_weight=1.0"
        --gin_param="set_seed.seed=42"
    )
    record_run "${run_dir}" "${mode}" "${name}" current "${backend}" \
        env "${environment[@]}" "${command[@]}"
    echo "[${mode}] ${name}: current consumer, ${backend} backend"
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' env "${environment[@]}" "${command[@]}"
        printf '\n'
        return
    fi
    env "${environment[@]}" "${command[@]}" 2>&1 | tee "${run_dir}/launcher.log"
}

run_one() {
    local mode=$1
    local name=$2
    local steps
    local matching_steps
    if [[ "${mode}" == smoke ]]; then
        steps=${SMOKE_STEPS}
        matching_steps=${SMOKE_MATCHING_STEPS}
    else
        steps=${FULL_STEPS}
        matching_steps=${FULL_MATCHING_STEPS}
    fi
    local run_dir="$(run_root_for_mode "${mode}")/${name}"
    local cache_dir="${EXPERIMENT_ROOT}/matching_cache/${mode}/${SCENE_NAME}/${name}"
    if ! prepare_run_dir "${run_dir}" "${cache_dir}" "${mode}" "${name}"; then
        return
    fi
    mkdir -p "${cache_dir}"
    case "${name}" in
        E0|E1|E2) run_legacy "${mode}" "${name}" "${steps}" "${matching_steps}" "${run_dir}" "${cache_dir}" ;;
        E3|E4) run_current "${mode}" "${name}" "${steps}" "${matching_steps}" "${run_dir}" "${cache_dir}" ;;
        *) echo "Unknown run: ${name}" >&2; exit 2 ;;
    esac
}

summarize() {
    "${CONDA_BIN}" run --no-capture-output -n "${SR_CONDA_ENV}" \
        "${SR_PYTHON_BIN}" "${SUITE_DIR}/summarize_results.py" \
        "--experiment-root=${EXPERIMENT_ROOT}" \
        "--scene-name=${SCENE_NAME}" \
        "--control-reference-psnr=${CONTROL_REFERENCE_PSNR}"
}

current_decision() {
    "${CONDA_BIN}" run --no-capture-output -n "${SR_CONDA_ENV}" \
        "${SR_PYTHON_BIN}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["decision"]["classification"])' \
        "${EXPERIMENT_ROOT}/summaries/${SCENE_NAME}/summary.json"
}

run_smoke_matrix() {
    prepare_all
    for name in E0 E1 E2 E3; do
        run_one smoke "${name}"
    done
}

run_full_matrix() {
    prepare_all
    local decision=incomplete
    for name in E0 E1 E2 E3; do
        run_one smoke "${name}"
        run_one full "${name}"
        if [[ "${DRY_RUN}" == 1 ]]; then
            continue
        fi
        summarize
        decision=$(current_decision)
        case "${decision}" in
            invalid_control_environment)
                echo "[stop] ${decision}"
                return
                ;;
            native_128_rendering|custom_gsplat_producer)
                if [[ "${CONTINUE_AFTER_FAILURE}" == 1 ]]; then
                    echo "[continue] ${decision}; CONTINUE_AFTER_FAILURE=1"
                else
                    echo "[stop] ${decision}"
                    echo "Re-run with CONTINUE_AFTER_FAILURE=1 to run the remaining E2/E3 diagnostics."
                    return
                fi
                ;;
        esac
    done
    if [[ "${DRY_RUN}" == 1 ]]; then
        echo "[dry-run] E4 would run only when the completed summary requests it."
        return
    fi
    summarize
    decision=$(current_decision)
    if [[ "${decision}" == e4_required ]]; then
        run_one smoke E4
        run_one full E4
        summarize
    else
        echo "[skip] E4 not required by decision: ${decision}"
    fi
}

run_all_matrix() {
    prepare_all
    for name in E0 E1 E2 E3 E4; do
        run_one smoke "${name}"
        run_one full "${name}"
        if [[ "${DRY_RUN}" == 0 ]]; then
            summarize
        fi
    done
}

usage() {
    cat <<EOF
Usage: $(basename "$0") COMMAND

Commands:
  prepare    Create clean legacy export, native Nerfstudio producers, and fixtures.
  smoke      Run the 20-step smoke matrix E0-E3.
  full       Smoke-test before each full run; stop/apply E4 using decision rules.
  all        Run smoke and full E0-E4 regardless of intermediate decisions.
  summarize Produce JSON/Markdown summaries without launching training.
  confirm    Run the same guarded matrix on ${CONFIRM_SCENE_NAME}.

Environment routing: Nerfstudio uses ${NERFSTUDIO_CONDA_ENV}; gsplat and SR use
${SR_CONDA_ENV}.

Useful overrides: EXPERIMENT_ROOT, SCENE_NAME, GPU_ID, CONDA_BIN,
NERFSTUDIO_CONDA_ENV, SR_CONDA_ENV, SR_PYTHON_BIN, NS_TRAIN_BIN,
NATIVE_DATA_ROOT, LEGACY_DATA_ROOT, VERIFY_RENDER, MAX_RENDER_VIEWS, DRY_RUN,
RESTART_INCOMPLETE, CONTINUE_AFTER_FAILURE.
EOF
}

command=${1:-}
case "${command}" in
    prepare) prepare_all ;;
    smoke) run_smoke_matrix ;;
    full) run_full_matrix ;;
    all) run_all_matrix ;;
    summarize) summarize ;;
    confirm)
        if [[ "${SCENE_NAME}" == "${CONFIRM_SCENE_NAME}" ]]; then
            echo "SCENE_NAME already equals CONFIRM_SCENE_NAME; refusing recursive confirm." >&2
            exit 2
        fi
        exec env SCENE_NAME="${CONFIRM_SCENE_NAME}" "$0" full
        ;;
    *) usage; [[ -z "${command}" ]] || exit 2 ;;
esac
