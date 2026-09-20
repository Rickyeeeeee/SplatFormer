"""NeRF-style Fourier feature encoders for Gaussian attributes."""

from typing import Mapping, Sequence

import torch
import torch.nn as nn


class FourierFeatureEncoder(nn.Module):
    """Concatenate an input with sine and cosine features at fixed frequencies."""

    def __init__(self, input_dims, num_frequencies, include_input=True, log_sampling=True,
                 max_frequency_log2=None):
        super().__init__()
        if input_dims <= 0:
            raise ValueError(f"input_dims must be positive, got {input_dims}")
        if num_frequencies <= 0:
            raise ValueError(f"num_frequencies must be positive, got {num_frequencies}")
        if max_frequency_log2 is None:
            max_frequency_log2 = num_frequencies - 1
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(0.0, max_frequency_log2, num_frequencies)
        else:
            frequency_bands = torch.linspace(1.0, 2.0 ** max_frequency_log2, num_frequencies)
        self.input_dims = input_dims
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.register_buffer("frequency_bands", frequency_bands)
        self.output_dims = input_dims * (int(include_input) + 2 * num_frequencies)

    def forward(self, inputs):
        if inputs.shape[-1] != self.input_dims:
            raise ValueError(f"Expected input with {self.input_dims} channels, got {inputs.shape[-1]}")
        encoded = [inputs] if self.include_input else []
        for frequency in self.frequency_bands.to(dtype=inputs.dtype):
            encoded.extend([torch.sin(inputs * frequency), torch.cos(inputs * frequency)])
        return torch.cat(encoded, dim=-1)


def build_fourier_feature_encoders(
    feature_channels: Mapping[str, int],
    input_features: Sequence[str],
    fourier_input_features=(),
    fourier_num_frequencies=None,
    fourier_include_raw=True,
    fourier_log_sampling=True,
    fourier_max_frequency_log2=None,
):
    """Build validated encoders and return the resulting width for each feature."""
    enabled_features = list(fourier_input_features)
    if len(enabled_features) != len(set(enabled_features)):
        raise ValueError("fourier_input_features must not contain duplicates")
    supported_features = {"means", "scales", "quats"}
    unsupported = set(enabled_features) - supported_features
    if unsupported:
        raise ValueError(f"Fourier encoding is unsupported for: {sorted(unsupported)}")
    missing_inputs = set(enabled_features) - set(input_features)
    if missing_inputs:
        raise ValueError(f"Fourier-encoded features must be in input_features: {sorted(missing_inputs)}")
    num_frequencies = {} if fourier_num_frequencies is None else dict(fourier_num_frequencies)
    max_frequency_log2 = {} if fourier_max_frequency_log2 is None else dict(fourier_max_frequency_log2)
    unknown_counts = set(num_frequencies) - set(enabled_features)
    unknown_maximums = set(max_frequency_log2) - set(enabled_features)
    if unknown_counts or unknown_maximums:
        unknown = sorted(unknown_counts | unknown_maximums)
        raise ValueError(f"Fourier settings provided for disabled features: {unknown}")
    missing_counts = set(enabled_features) - set(num_frequencies)
    if missing_counts:
        raise ValueError(f"Missing fourier_num_frequencies for: {sorted(missing_counts)}")
    encoders = nn.ModuleDict()
    output_channels = dict(feature_channels)
    for feature in enabled_features:
        count = num_frequencies[feature]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"fourier_num_frequencies[{feature!r}] must be a positive integer")
        encoder = FourierFeatureEncoder(
            input_dims=feature_channels[feature], num_frequencies=count,
            include_input=fourier_include_raw, log_sampling=fourier_log_sampling,
            max_frequency_log2=max_frequency_log2.get(feature),
        )
        encoders[feature] = encoder
        output_channels[feature] = encoder.output_dims
    return encoders, output_channels
