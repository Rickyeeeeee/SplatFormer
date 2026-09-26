"""Fixed attribute standardization in the loader's scene coordinate frame."""
import json
import math

import torch


STATISTIC_KEYS = {
    "means": "means", "scales": "scales", "opacities": "opacities",
    "quats": "quats", "features_dc": "sh0", "features_rest": "shN",
}


QUATERNION_REPRESENTATIONS = ("raw_standardized", "unit_unstandardized")


def normalize_gaussian_quaternions(endpoint):
    """Return unit quaternion endpoints without modifying the input dictionary."""
    quats = endpoint["quats"].float()
    norms = torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    if not torch.isfinite(quats).all() or not torch.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Quaternion endpoints must be finite and non-zero")
    return {**endpoint, "quats": quats / norms}


class GaussianStandardizer:
    def __init__(self, path, variance_floor=1e-8, quaternion_representation="raw_standardized"):
        if quaternion_representation not in QUATERNION_REPRESENTATIONS:
            raise ValueError(f"Unsupported quaternion_representation={quaternion_representation!r}")
        self.quaternion_representation = quaternion_representation
        if not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("normalization_variance_floor must be finite and positive")
        self.path = str(path)
        self.variance_floor = variance_floor
        with open(path) as handle:
            document = json.load(handle)
        try:
            statistics = document["aggregate"]["normalized"]["output"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Statistics file {path} must contain aggregate.normalized.output with per-attribute mean and variance") from error
        self.means, self.variances, self.stds = {}, {}, {}
        for key, stored_key in STATISTIC_KEYS.items():
            try:
                mean = torch.tensor(statistics[stored_key]["mean"], dtype=torch.float32)
                variance = torch.tensor(statistics[stored_key]["variance"], dtype=torch.float32)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing or invalid mean/variance for {stored_key} in {path}") from error
            if key == "features_dc":
                if mean.shape == (1, 3):
                    mean = mean.squeeze(0)
                if variance.shape == (1, 3):
                    variance = variance.squeeze(0)
            if key == "opacities":
                mean, variance = mean.reshape(-1), variance.reshape(-1)
            expected = {"means": (3,), "scales": (3,), "quats": (4,), "opacities": (1,), "features_dc": (3,)}
            if mean.shape != variance.shape or (key in expected and tuple(mean.shape) != expected[key]):
                raise ValueError(f"Incompatible statistics shapes for {key}: {tuple(mean.shape)}, {tuple(variance.shape)}")
            if key == "features_rest" and (mean.ndim != 2 or mean.shape[-1] != 3):
                raise ValueError("features_rest statistics must have shape [SH coefficients, 3]")
            if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
                raise ValueError(f"Non-finite mean or invalid variance for {key}")
            self.means[key], self.variances[key] = mean, variance
            self.stds[key] = variance.clamp_min(variance_floor).sqrt()

    def prepare_endpoints(self, source, target=None, align_target_sign=True):
        """Normalize physical endpoints only; preserve source signs and caller tensors."""
        if self.quaternion_representation == "raw_standardized":
            return source, target
        prepared = []
        for endpoint in (source, target):
            if endpoint is None:
                prepared.append(None)
                continue
            prepared.append(normalize_gaussian_quaternions(endpoint))
        source, target = prepared
        if align_target_sign and source is not None and target is not None:
            if source["quats"].shape != target["quats"].shape:
                raise ValueError("Quaternion sign alignment requires paired endpoint shapes")
            flip = (source["quats"] * target["quats"]).sum(dim=-1, keepdim=True) < 0
            target["quats"] = torch.where(flip, -target["quats"], target["quats"])
        return source, target

    def encode(self, gaussians):
        result = {}
        for key in STATISTIC_KEYS:
            value = gaussians[key].float()
            if tuple(value.shape[1:]) != tuple(self.means[key].shape):
                raise ValueError(f"Statistics shape for {key} does not match Gaussian channels {tuple(value.shape[1:])}")
            if key == "quats" and self.quaternion_representation == "unit_unstandardized":
                result[key] = value
                continue
            result[key] = (value - self.means[key].to(value.device)) / self.stds[key].to(value.device)
        return result

    def decode(self, gaussians):
        return {key: (value.float() if key == "quats" and self.quaternion_representation == "unit_unstandardized"
                      else value.float() * self.stds[key].to(value.device) + self.means[key].to(value.device))
                for key, value in gaussians.items() if key in STATISTIC_KEYS}

    def report(self):
        attributes = {}
        for key in STATISTIC_KEYS:
            bypass = key == "quats" and self.quaternion_representation == "unit_unstandardized"
            effective_mean = torch.zeros_like(self.means[key]) if bypass else self.means[key]
            effective_scale = torch.ones_like(self.stds[key]) if bypass else self.stds[key]
            effective_variance = torch.ones_like(self.variances[key]) if bypass else self.variances[key].clamp_min(self.variance_floor)
            attributes[key] = {"mean": self.means[key].tolist(), "variance": self.variances[key].tolist(),
                               "effective_mean": effective_mean.tolist(), "effective_scale": effective_scale.tolist(),
                               "effective_variance": effective_variance.tolist()}
        return {"source_path": self.path, "statistics_group": "aggregate.normalized.output",
                "variance_floor": self.variance_floor, "quaternion_representation": self.quaternion_representation,
                "attributes": attributes}
