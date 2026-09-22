"""Fixed attribute standardization in the loader's scene coordinate frame."""
import json
import math

import torch


STATISTIC_KEYS = {
    "means": "means", "scales": "scales", "opacities": "opacities",
    "quats": "quats", "features_dc": "sh0", "features_rest": "shN",
}


class GaussianStandardizer:
    def __init__(self, path, variance_floor=1e-8):
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

    def encode(self, gaussians):
        result = {}
        for key in STATISTIC_KEYS:
            value = gaussians[key].float()
            if tuple(value.shape[1:]) != tuple(self.means[key].shape):
                raise ValueError(f"Statistics shape for {key} does not match Gaussian channels {tuple(value.shape[1:])}")
            result[key] = (value - self.means[key].to(value.device)) / self.stds[key].to(value.device)
        return result

    def decode(self, gaussians):
        return {key: value.float() * self.stds[key].to(value.device) + self.means[key].to(value.device)
                for key, value in gaussians.items() if key in STATISTIC_KEYS}

    def report(self):
        return {"source_path": self.path, "statistics_group": "aggregate.normalized.output",
                "variance_floor": self.variance_floor,
                "attributes": {key: {"mean": self.means[key].tolist(), "variance": self.variances[key].tolist(),
                                     "effective_variance": self.variances[key].clamp_min(self.variance_floor).tolist()}
                               for key in STATISTIC_KEYS}}
