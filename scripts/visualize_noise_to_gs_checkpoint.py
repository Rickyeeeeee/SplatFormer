#!/usr/bin/env python3
"""Inspect a trained noise-to-GS flow sample in Viser.

Example:
  python scripts/visualize_noise_to_gs_checkpoint.py \
    --checkpoint output_overfit_noise_to_gs/run/checkpoints/model_last.pth \
    --predictor dit

The checkpoint's run directory must contain config.gin unless --config is set.
DiT defaults to its configured maximum Gaussian count; other predictors need
--gs_count. GT-coordinate DiT checkpoints need --scene_name for reference means.
In Viser, set the two seeds and Flow steps, then select Noise interpolation to
generate six blends. The interpolation-rate slider switches cached samples.
"""

import argparse
import colorsys
from pathlib import Path
import sys
import threading
import time

import gin
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.GS_SR import SplatFactoSRDataset
from models.diffusion_gaussian_predictor import DiffusionGaussianPredictor
from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy Gin keys.
from models.gaussian_dit_predictor import GaussianDiTPredictor
from sr import noise_to_gs
from sr.noise_to_gs import flow_matching
from utils.gaussian_viewer import GsplatBackgroundRenderer, _rgb_from_gaussians, _validate_gsplat_available
from utils.gs_normalization import GaussianStandardizer


PREDICTORS = {"ptv3": GSFlowPredictor, "dipt": DiffusionGaussianPredictor,
              "gdpt": DiffusionGaussianPredictor, "dit": GaussianDiTPredictor}
INTERPOLATION_RATES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Saved config.gin; defaults to the checkpoint's run directory.")
    parser.add_argument("--predictor", choices=tuple(PREDICTORS), default="dit")
    parser.add_argument("--scene_name", help="Reference scene for GT-coordinate DiT or paired evaluation sampling.")
    parser.add_argument("--gs_count", type=int, help="Number of Gaussians to generate; defaults to DiT max_gaussians.")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--gin_param", action="append", default=[])
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--train_scene_list", type=Path)
    parser.add_argument("--test_scene_list", type=Path)
    parser.add_argument("--source_gs_root", type=Path)
    parser.add_argument("--target_gs_root", type=Path)
    parser.add_argument("--statistics_path", type=Path)
    parser.add_argument("--noise_seed", type=int, help="Override the saved evaluation seed.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--initial_view_mode", choices=("pointcloud", "gsplat"), default="pointcloud")
    parser.add_argument("--initial_color_mode", choices=("gs", "index", "morton"), default="gs",
                        help="Use 'index' for fixed sampling slots; 'morton' is a legacy alias.")
    parser.add_argument("--point_size", type=float, default=0.003)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--render_interval", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true", help="Load and sample without starting Viser.")
    args = parser.parse_args(argv)
    if args.point_size <= 0 or args.render_height < 1 or args.render_interval < 0 or not np.isfinite((args.point_size, args.render_interval)).all():
        parser.error("Point size and render height must be positive; render interval must be nonnegative")
    if args.port < 1 or args.port > 65535:
        parser.error("Port must be between 1 and 65535")
    if args.gs_count is not None and args.gs_count < 1:
        parser.error("gs_count must be positive")
    args.config = args.config or args.checkpoint.parent.parent / "config.gin"
    return args


def load_experiment(args):
    # Parse the run's operative config, retaining explicit path overrides for moved datasets.
    gin.clear_config()
    bindings = list(args.gin_param)
    if args.dataset_root is not None:
        bindings.append(f"dataset_root={str(args.dataset_root)!r}")
    for name in ("train_scene_list", "test_scene_list"):
        path = getattr(args, name)
        if path is not None:
            bindings.append(f"{name}={str(path)!r}")
    gin.parse_config_files_and_bindings([str(args.config)], bindings, skip_unknown=True)
    flow_cfg = flow_matching()
    statistics_path = args.statistics_path or Path(flow_cfg["gs_statistics_path"])
    model = PREDICTORS[args.predictor]().to(args.device)
    noise_to_gs.validate_coupling(model, flow_cfg["coupling"])
    if args.predictor in ("dipt", "gdpt") and model.backbone_type.lower() != args.predictor:
        raise ValueError(f"Predictor {args.predictor} requires its matching backbone Gin config")
    if model.backbone_type == "DIT" and model.coordinate_mode == "gt" and not args.scene_name:
        raise ValueError("GT-coordinate DiT needs --scene_name to supply reference means")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    standardizer = GaussianStandardizer(statistics_path, flow_cfg["normalization_variance_floor"],
                                        sh_degree=0 if model.sh_degree == 0 else None)
    scene_idx = 0
    reference_means = None
    count = args.gs_count or getattr(model, "max_gaussians", None)
    if args.scene_name:
        # Scene loading is only needed for its native count, seed offset, or GT reference means.
        dataset = SplatFactoSRDataset.from_gin_scope(
            f"{args.split}_dataset", load_src_gs=False, load_tgt_gs=True,
            load_src_images=False, load_tgt_images=False, split_across_gpus=False,
        )
        scene_idx = dataset.scene_index(args.scene_name)
        for resolution, root in ((dataset.src_resolution, args.source_gs_root),
                                 (dataset.tgt_resolution, args.target_gs_root)):
            if root is not None:
                dataset.folders[scene_idx]["resolution_paths"][resolution]["gsplat_dir"] = str(
                    root / f"{args.split}-set" / dataset.dataset_name / str(resolution) / "gsplat" / args.scene_name
                )
        target = dataset.load_scene(scene_idx, sample_views=False)["data"][dataset.tgt_resolution]["gs_params"]
        count = len(target["means"])
        if args.gs_count is not None and args.gs_count != count:
            raise ValueError(f"gs_count={args.gs_count} does not match scene count {count}")
        if model.backbone_type == "DIT" and model.coordinate_mode == "gt":
            if model.sh_degree == 0:
                target = {**target, "features_rest": target["features_rest"][:, :0, :]}
            reference_means = standardizer.encode(target)["means"].to(args.device)
    if count is None:
        raise ValueError(f"Predictor {args.predictor} needs --gs_count when no scene is selected")
    if hasattr(model, "max_gaussians") and count > model.max_gaussians:
        raise ValueError(f"Requested {count} Gaussians; model maximum is {model.max_gaussians}")
    seed = flow_cfg["eval_noise_seed"] if args.noise_seed is None else args.noise_seed
    return model, standardizer, count, scene_idx, reference_means, flow_cfg, seed


@torch.no_grad()
def record_sample(model, count, scene_idx, standardizer, steps, t_eps, seed, reference_means=None,
                  initial_noise=None, coupling="random"):
    """Record physical Gaussian states using the same seeded Euler rollout as evaluation."""
    device = next(model.parameters()).device
    noise_to_gs.validate_coupling(model, coupling)
    if initial_noise is None:
        generator = torch.Generator(device=device).manual_seed(int(seed) + int(scene_idx))
        initial_noise = noise_to_gs.order_noise(noise_to_gs.sample_noise(count, standardizer, device, generator), coupling)
    states = []
    was_training = model.training
    model.eval()
    try:
        noise_to_gs.rollout(model, initial_noise, scene_idx, steps, standardizer, t_eps, reference_means,
                            on_state=lambda _, state: states.append({key: value.detach().cpu()
                                                                     for key, value in standardizer.decode(state).items()}))
    finally:
        model.train(was_training)
    return states


@torch.no_grad()
def record_interpolated_samples(model, count, scene_idx, standardizer, steps, t_eps, seed_a, seed_b,
                                reference_means=None, on_progress=None, coupling="random"):
    """Sample six linear blends of two fixed standardized noise tensors."""
    device = next(model.parameters()).device
    generator_a = torch.Generator(device=device).manual_seed(int(seed_a) + int(scene_idx))
    generator_b = torch.Generator(device=device).manual_seed(int(seed_b) + int(scene_idx))
    noise_to_gs.validate_coupling(model, coupling)
    noise_a = noise_to_gs.order_noise(noise_to_gs.sample_noise(count, standardizer, device, generator_a), coupling)
    noise_b = noise_to_gs.order_noise(noise_to_gs.sample_noise(count, standardizer, device, generator_b), coupling)
    trajectories = []
    for index, rate in enumerate(INTERPOLATION_RATES):
        blended = {key: (1 - rate) * noise_a[key] + rate * noise_b[key] for key in noise_a}
        trajectories.append(record_sample(model, count, scene_idx, standardizer, steps, t_eps, seed_a,
                                          reference_means, initial_noise=blended, coupling=coupling))
        if on_progress is not None:
            on_progress(index + 1, len(INTERPOLATION_RATES))
    return trajectories


def sequence_index_colors(count):
    """Give each sampling tensor slot a fixed color, independent of its position."""
    gradient = [colorsys.hsv_to_rgb(0.75 * (1 - index / max(count - 1, 1)), 0.85, 1.0)
                for index in range(count)]
    return np.round(np.asarray(gradient) * 255).astype(np.uint8)


def sequence_colorbar(colors, width=256, height=16):
    indices = np.rint(np.linspace(0, len(colors) - 1, width)).astype(np.int64)
    image = np.broadcast_to(colors[indices][None], (height, width, 3)).copy()
    ticks = np.unique(np.rint(np.linspace(0, len(colors) - 1, 5)).astype(np.int64))
    return image, "Index (left → right): " + " · ".join(f"**{index}**" for index in ticks)


class CheckpointViewer:
    def __init__(self, args, model, standardizer, count, scene_idx, reference_means, flow_cfg, seed, states):
        import viser

        self.args, self.model, self.standardizer = args, model, standardizer
        self.count, self.scene_idx = count, scene_idx
        self.reference_means, self.flow_cfg = reference_means, flow_cfg
        self.active_seed, self.states = seed, states
        self.single_states, self.interpolated_states = states, None
        self.active_mode, self.active_steps = "Single", len(states) - 1
        self.interpolation_seeds = None
        self.active_checkpoint = args.checkpoint
        self.sequence_colors = sequence_index_colors(count)
        self.lock = threading.RLock()
        self.sampling = False
        self.message = "Ready"
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.set_up_direction("+z")
        self.state = {"view_mode": args.initial_view_mode}
        render_cfg = {"device": args.device, "height": args.render_height, "interval": args.render_interval,
                      "near_plane": 0.01, "far_plane": 100.0, "radius_clip": 0.0, "eps2d": 0.3,
                      "rasterize_mode": "classic", "background": np.zeros(3, dtype=np.float32)}
        self.renderer = GsplatBackgroundRenderer(self.server, states[0], self.state, render_cfg)
        self.point_handle = None
        gui = self.gui = self.server.gui
        self.status = gui.add_markdown("")
        self.step = gui.add_slider("Flow step", 0, self.active_steps, 1, 0)
        gui.add_button("Decrease step").on_click(self.decrease_step)
        gui.add_button("Increase step").on_click(self.increase_step)
        self.mode = gui.add_dropdown("Sampling mode", ("Single", "Noise interpolation"), initial_value="Single")
        self.seed = gui.add_number("Noise seed", initial_value=seed, step=1)
        self.seed_b = gui.add_number("Second noise seed (interpolation)", initial_value=seed + 1, step=1)
        self.flow_steps = gui.add_number("Flow steps", initial_value=self.active_steps, min=1, step=1)
        self.resample_button = gui.add_button("Resample")
        self.rate = gui.add_slider("Interpolation rate", 0.0, 1.0, 0.2, 0.0, visible=False)
        self.checkpoint_input = gui.add_text("Checkpoint path", initial_value=str(args.checkpoint))
        self.reload_button = gui.add_button("Load checkpoint")
        self.view_mode = gui.add_dropdown("Rendering", ("Point cloud", "GS"),
                                          initial_value="Point cloud" if args.initial_view_mode == "pointcloud" else "GS")
        self.color_mode = gui.add_dropdown("Coloring", ("GS color", "Sequence index"),
                                           initial_value="GS color" if args.initial_color_mode == "gs" else "Sequence index")
        bar_image, bar_labels = sequence_colorbar(self.sequence_colors)
        self.colorbar = gui.add_image(bar_image, label="Sequence index", format="png")
        self.colorbar_labels = gui.add_markdown(bar_labels)
        self.point_size = gui.add_number("Point size", initial_value=args.point_size, min=1e-6, step=0.001)
        self.step.on_update(self.refresh)
        self.mode.on_update(self.change_sample_mode)
        self.rate.on_update(self.change_interpolation_rate)
        self.view_mode.on_update(self.change_view_mode)
        self.color_mode.on_update(self.refresh)
        self.point_size.on_update(self.refresh)
        self.resample_button.on_click(self.resample)
        self.reload_button.on_click(self.reload_checkpoint)
        self.server.on_client_connect(self.on_client_connect)
        self.refresh(None)

    def refresh(self, _event):
        with self.lock:
            current = self.states[int(self.step.value)]
            indices = np.arange(self.count)
            sequence = self.color_mode.value == "Sequence index"
            self.rate.visible = self.active_mode == "Noise interpolation"
            self.colorbar.visible = self.colorbar_labels.visible = sequence
            colors = self.sequence_colors if sequence else _rgb_from_gaussians(current, indices)
            self.renderer.update_scene(current, self.sequence_colors if sequence else None)
            if self.point_handle is not None:
                self.point_handle.remove()
            self.point_handle = self.server.scene.add_point_cloud(
                "/sample", points=current["means"].numpy(), colors=colors,
                point_size=float(self.point_size.value), point_shape="circle",
                visible=self.state["view_mode"] == "pointcloud",
            )
            if self.active_mode == "Noise interpolation":
                rate = INTERPOLATION_RATES[round(float(self.rate.value) * 5)]
                sample_info = f"Seeds **{self.interpolation_seeds[0]} → {self.interpolation_seeds[1]}** · α=**{rate:.1f}**"
            else:
                sample_info = f"Seed **{self.active_seed}**"
            self.status.content = (f"**{self.args.scene_name or 'Noise only'}** · {self.count:,} Gaussians · "
                                   f"checkpoint `{self.active_checkpoint}` · config `{self.args.config}`  \n"
                                   f"{sample_info} · step **{self.step.value}/{self.active_steps}** · {self.message}")
        if self.state["view_mode"] == "gsplat":
            self.renderer.render_all()

    def set_active_states(self, states, step):
        steps = len(states) - 1
        previous_steps = self.active_steps
        self.states, self.active_steps = states, steps
        if steps != previous_steps:
            # Legacy Viser sliders cannot change bounds, so replace the control in place.
            old = self.step
            self.step = self.gui.add_slider("Flow step", 0, steps, 1, step, order=old.order)
            old.remove()
            self.step.on_update(self.refresh)
        else:
            self.step.value = step
        self.refresh(None)

    def change_sample_mode(self, _event):
        requested = self.mode.value
        if requested == self.active_mode:
            self.refresh(None)
        elif requested == "Single":
            self.active_mode = "Single"
            self.set_active_states(self.single_states, len(self.single_states) - 1)
        elif self.interpolated_states is not None:
            self.active_mode = "Noise interpolation"
            self.set_active_states(self.interpolated_states[round(float(self.rate.value) * 5)],
                                   len(self.interpolated_states[0]) - 1)
        else:
            self.start_sampling(reload_checkpoint=False)

    def change_interpolation_rate(self, _event):
        if self.active_mode == "Noise interpolation":
            index = round(float(self.rate.value) * 5)
            self.set_active_states(self.interpolated_states[index], int(self.step.value))

    def change_view_mode(self, _event):
        requested = "gsplat" if self.view_mode.value == "GS" else "pointcloud"
        if requested == "gsplat":
            try:
                _validate_gsplat_available(self.args.device)
            except RuntimeError as error:
                self.message = f"GS rendering unavailable: {error}"
                self.view_mode.value = "Point cloud"
                return
        self.state["view_mode"] = requested
        if requested == "pointcloud":
            self.renderer.clear_all()
        self.refresh(None)

    def decrease_step(self, _event):
        self.step.value = max(0, int(self.step.value) - 1)

    def increase_step(self, _event):
        self.step.value = min(self.active_steps, int(self.step.value) + 1)

    def resample(self, _event):
        self.start_sampling(reload_checkpoint=False)

    def reload_checkpoint(self, _event):
        self.start_sampling(reload_checkpoint=True)

    def start_sampling(self, reload_checkpoint):
        with self.lock:
            if self.sampling:
                return
            steps = int(self.flow_steps.value)
            if steps < 1:
                self.message = "Flow steps must be positive"
                self.refresh(None)
                return
            self.sampling = True
            seed_a, seed_b = int(self.seed.value), int(self.seed_b.value)
            mode = self.mode.value
            path = Path(self.checkpoint_input.value) if reload_checkpoint else self.active_checkpoint
            self.message = f"Loading {path}…" if reload_checkpoint else f"Sampling {mode.lower()}…"
            self.resample_button.disabled = True
            self.reload_button.disabled = True
            self.mode.disabled = True
            self.refresh(None)

        def worker():
            try:
                model = self.model
                if reload_checkpoint:
                    # Keep the active model and sample until the replacement has loaded and sampled.
                    model = type(self.model)()
                    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
                    model = model.to(self.args.device).eval()
                if mode == "Noise interpolation":
                    def progress(done, total):
                        with self.lock:
                            self.status.content = f"Sampling interpolation **{done}/{total}**; previous result remains visible."

                    trajectories = record_interpolated_samples(
                        model, self.count, self.scene_idx, self.standardizer, steps, self.flow_cfg["flow_t_eps"],
                        seed_a, seed_b, self.reference_means, on_progress=progress, coupling=self.flow_cfg["coupling"]
                    )
                    states = trajectories[round(float(self.rate.value) * 5)]
                else:
                    states = record_sample(model, self.count, self.scene_idx, self.standardizer,
                                           steps, self.flow_cfg["flow_t_eps"], seed_a, self.reference_means,
                                           coupling=self.flow_cfg["coupling"])
                with self.lock:
                    self.model, self.active_checkpoint = model, path
                    self.active_seed = seed_a
                    if mode == "Noise interpolation":
                        self.interpolated_states, self.interpolation_seeds = trajectories, (seed_a, seed_b)
                        if reload_checkpoint:
                            self.single_states = trajectories[0]
                    else:
                        self.single_states = states
                        if reload_checkpoint:
                            self.interpolated_states = None
                    self.active_mode = mode
                    self.message = "Ready"
                    self.set_active_states(states, steps)
            except Exception as error:
                with self.lock:
                    operation = "Checkpoint load" if reload_checkpoint else "Sampling"
                    self.message = f"{operation} failed: {error}"
                    if self.mode.value != self.active_mode:
                        self.mode.value = self.active_mode
                    self.refresh(None)
            finally:
                with self.lock:
                    self.sampling = False
                    self.resample_button.disabled = False
                    self.reload_button.disabled = False
                    self.mode.disabled = False

        threading.Thread(target=worker, daemon=True).start()

    def on_client_connect(self, client):
        @client.camera.on_update
        def _(_camera):
            self.renderer.render_client(client)
        if self.state["view_mode"] == "gsplat":
            self.renderer.render_client(client, force=True)


def main(argv=None):
    args = parse_args(argv)
    if args.initial_view_mode == "gsplat" and not args.dry_run:
        _validate_gsplat_available(args.device)
    model, standardizer, count, scene_idx, reference_means, flow_cfg, seed = load_experiment(args)
    states = record_sample(model, count, scene_idx, standardizer, flow_cfg["flow_steps"],
                           flow_cfg["flow_t_eps"], seed, reference_means, coupling=flow_cfg["coupling"])
    print(f"Loaded {args.checkpoint}; sampled {len(states)} states ({count:,} Gaussians)", flush=True)
    if args.dry_run:
        return
    viewer = CheckpointViewer(args, model, standardizer, count, scene_idx, reference_means, flow_cfg, seed, states)
    print(f"Viser server running at http://{args.host}:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.server.stop()


if __name__ == "__main__":
    main()
