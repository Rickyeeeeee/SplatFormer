import gin
import torch

from utils import gs_utils
from utils.optimizers import build_3DGSoptimizer


class GSPath:
    MODE_OPTIMIZED_TARGET_DISCRETE = "optimized_target_discrete"
    MODE_PROGRESSIVE_SEGMENT = "progressive_segment"

    def __init__(
        self,
        path_mode=MODE_OPTIMIZED_TARGET_DISCRETE,
        flow_num_timesteps=10,
        flow_segment_steps=10,
        logger=None,
    ):
        if path_mode not in [self.MODE_OPTIMIZED_TARGET_DISCRETE, self.MODE_PROGRESSIVE_SEGMENT]:
            raise ValueError(f"Unsupported path_mode: {path_mode}")
        if flow_num_timesteps <= 0:
            raise ValueError("flow_num_timesteps must be positive")
        if flow_segment_steps <= 0:
            raise ValueError("flow_segment_steps must be positive")

        self.path_mode = path_mode
        self.flow_num_timesteps = int(flow_num_timesteps)
        self.flow_segment_steps = int(flow_segment_steps)
        self.logger = logger

    @staticmethod
    def flow_float_keys(input_gs, target_gs, output_features):
        del target_gs
        keys = []
        for key in output_features:
            if key not in input_gs:
                continue
            if not torch.is_tensor(input_gs[key]):
                continue
            if input_gs[key].is_floating_point():
                keys.append(key)
        if len(keys) == 0:
            raise ValueError("No floating output features are present in input GS")
        return keys

    @staticmethod
    def detach_gs(gs):
        out = {}
        for key, value in gs.items():
            out[key] = value.detach().clone() if torch.is_tensor(value) else value
        return out

    @staticmethod
    def clone_trainable_gs(input_gs, flow_keys):
        state = {}
        trainable = {}
        flow_key_set = set(flow_keys)
        for key, value in input_gs.items():
            if torch.is_tensor(value):
                cloned = value.detach().clone()
                if key in flow_key_set and cloned.is_floating_point():
                    cloned.requires_grad_(True)
                    trainable[key] = cloned
                state[key] = cloned
            else:
                state[key] = value
        return state, trainable

    @staticmethod
    def render_l1_loss(gs, images, cameras):
        pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs, cameras)
        loss = 0
        for pred_img, gt_img in zip(pred_imgs, images):
            gt_rgb = gt_img[..., :3]
            if gt_img.shape[-1] == 4:
                mask = gt_img[..., 3:].to(pred_img.dtype)
                loss = loss + ((pred_img - gt_rgb) * mask).abs().mean()
            else:
                loss = loss + (pred_img - gt_rgb).abs().mean()
        return loss / max(len(pred_imgs), 1)

    def optimize_gs(self, input_gs, images, cameras, flow_keys, optim_steps, log_prefix="flow_optim"):
        if optim_steps <= 0:
            raise ValueError("optim_steps must be positive")

        state, trainable = self.clone_trainable_gs(input_gs, flow_keys)
        if len(trainable) == 0:
            raise ValueError("No trainable GS tensors selected for trajectory optimization")
        with gin.config_scope("flow_optim"):
            optimizer = build_3DGSoptimizer(trainable)

        last_loss = None
        for opt_step in range(1, optim_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = self.render_l1_loss(state, images, cameras)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()
            if self.logger is not None and (opt_step == optim_steps or opt_step % self.flow_segment_steps == 0):
                self.logger.info(f"{log_prefix} step={opt_step}/{optim_steps} image_l1={last_loss.item():.6f}")

        return self.detach_gs(state), last_loss

    def sample_from_timestep(self, device):
        timestep_idx_tensor = torch.randint(self.flow_num_timesteps, (1,), device=device)
        timestep_idx = int(timestep_idx_tensor.item())
        t = timestep_idx_tensor.to(dtype=torch.float32) / float(self.flow_num_timesteps)
        return t, timestep_idx

    def sample_timestep(self, device):
        return self.sample_from_timestep(device)[0]

    def timestep_index(self, t):
        if torch.is_tensor(t):
            t_value = float(t.detach().reshape(-1)[0].item())
        else:
            t_value = float(t)
        timestep_idx = int(round(t_value * float(self.flow_num_timesteps)))
        return max(0, min(timestep_idx, self.flow_num_timesteps - 1))

    @staticmethod
    def interpolate_gs(input_gs, target_gs, t, flow_keys):
        t_value = torch.as_tensor(t, device=input_gs["means"].device, dtype=torch.float32).reshape(())
        out = {}
        flow_key_set = set(flow_keys)
        for key, value in input_gs.items():
            if key in flow_key_set:
                out[key] = value + t_value * (target_gs[key] - value)
            else:
                out[key] = value
        return out

    def sample_path(self, data_dict, t):
        input_gs = data_dict["input_gs"]
        images = data_dict["images"]
        cameras = data_dict["cameras"]
        flow_keys = data_dict["flow_keys"]
        segment_idx = self.timestep_index(t)

        if self.path_mode == self.MODE_OPTIMIZED_TARGET_DISCRETE:
            if "optimized_target_gs" not in data_dict:
                optimized_target_gs, gs_loss = self.optimize_gs(
                    input_gs,
                    images,
                    cameras,
                    flow_keys=flow_keys,
                    optim_steps=self.flow_num_timesteps * self.flow_segment_steps,
                    log_prefix="optimized_target_discrete/gs",
                )
                data_dict["optimized_target_gs"] = optimized_target_gs
                data_dict["last_gs_loss"] = gs_loss

            optimized_target_gs = data_dict["optimized_target_gs"]
            start_t = float(segment_idx) / float(self.flow_num_timesteps)
            target_t = float(segment_idx + 1) / float(self.flow_num_timesteps)
            start_gs = self.interpolate_gs(input_gs, optimized_target_gs, start_t, flow_keys)
            target_gs = self.interpolate_gs(input_gs, optimized_target_gs, target_t, flow_keys)
            return start_gs, target_gs

        else:
            current_idx = 0
            current_gs = self.detach_gs(input_gs)
            gs_loss = None
            while current_idx < segment_idx:
                current_gs, gs_loss = self.optimize_gs(
                    current_gs,
                    images,
                    cameras,
                    flow_keys=flow_keys,
                    optim_steps=self.flow_segment_steps,
                    log_prefix=f"progressive_segment/gs_segment_{current_idx}",
                )
                current_idx += 1

            start_gs = current_gs
            target_gs, gs_loss = self.optimize_gs(
                start_gs,
                images,
                cameras,
                flow_keys=flow_keys,
                optim_steps=self.flow_segment_steps,
                log_prefix=f"progressive_segment/gs_segment_{segment_idx}",
            )
            data_dict["last_gs_loss"] = gs_loss
            return start_gs, target_gs

    @staticmethod
    def integration_timestep(step, num_steps, device, mode="left"):
        dt = 1.0 / float(num_steps)
        if mode == "left":
            t_value = float(step) * dt
        elif mode == "midpoint":
            t_value = (float(step) + 0.5) * dt
        else:
            raise ValueError(f"Unsupported flow integration timestep mode: {mode}")
        return torch.tensor([t_value], device=device)

    def integrate_flow(self, model, input_gs, num_steps=1, timestep_mode="left"):
        if num_steps <= 0:
            raise ValueError("flow_eval_steps must be positive")
        current_gs = input_gs
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t = self.integration_timestep(step, num_steps, input_gs["means"].device, mode=timestep_mode)
            pred_velocity = model(
                batch_normalized_gs=[current_gs],
                timestep=t,
            )[0]
            next_gs = dict(current_gs)
            for key in getattr(model, "output_features", []):
                if key in pred_velocity and key in current_gs and torch.is_tensor(current_gs[key]):
                    next_gs[key] = current_gs[key] + dt * pred_velocity[key]
            current_gs = next_gs
        return current_gs
