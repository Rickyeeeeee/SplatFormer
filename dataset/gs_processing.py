import torch

from utils.transform_utils import MinMaxScaler, remove_outliers


class GaussianProcessor:
    def __init__(self, remove_outlier_ndevs, max_gs_num):
        self.remove_outlier_ndevs = remove_outlier_ndevs
        self.max_gs_num = max_gs_num

    def finite_mask(self, gs_params):
        count = gs_params["means"].shape[0]
        mask = torch.ones(count, dtype=torch.bool)
        for value in gs_params.values():
            finite = torch.isfinite(value)
            for dim in range(value.ndim - 1, 0, -1):
                finite = finite.all(dim=dim)
            mask &= finite
        return mask

    def selection_mask(self, gs_params):
        mask = self.finite_mask(gs_params)
        if self.remove_outlier_ndevs > 0:
            valid_indices = mask.nonzero(as_tuple=False).squeeze(1)
            _, inlier_mask = remove_outliers(
                gs_params["means"][mask], n_devs=self.remove_outlier_ndevs
            )
            outlier_mask = torch.zeros_like(mask)
            outlier_mask[valid_indices[inlier_mask]] = True
            mask &= outlier_mask
        selected_indices = mask.nonzero(as_tuple=False).squeeze(1)
        if selected_indices.numel() > self.max_gs_num:
            keep = torch.zeros_like(mask)
            keep[selected_indices[: self.max_gs_num]] = True
            mask &= keep
        if not mask.any():
            raise ValueError("Gaussian filtering removed every splat")
        return mask

    def normalize(self, gs_params, scaler=None):
        normalized = {key: value.clone() for key, value in gs_params.items()}
        if scaler is None:
            scaler = MinMaxScaler()
            normalized["means"] = scaler.fit_transform(normalized["means"])
        else:
            normalized["means"] = scaler.transform(normalized["means"])
        normalized["scales"] = normalized["scales"] + torch.log(scaler.scale_)
        return normalized, scaler

    def process_native(self, raw_gs):
        mask = self.selection_mask(raw_gs)
        selected = {key: value[mask] for key, value in raw_gs.items()}
        return self.normalize(selected)

    def process_fitted_pair(self, source_raw, fitted_raw, target_scaler):
        if set(source_raw) != set(fitted_raw):
            raise ValueError("Source and fitted GS attributes do not match")
        for key in source_raw:
            if source_raw[key].shape != fitted_raw[key].shape:
                raise ValueError(
                    f"Source/fitted shape mismatch for {key}: "
                    f"{tuple(source_raw[key].shape)} vs "
                    f"{tuple(fitted_raw[key].shape)}"
                )

        shared_mask = self.selection_mask(source_raw) & self.finite_mask(fitted_raw)
        if not shared_mask.any():
            raise ValueError("Source/fitted shared filtering removed every splat")
        source = {key: value[shared_mask] for key, value in source_raw.items()}
        fitted = {key: value[shared_mask] for key, value in fitted_raw.items()}
        source, source_scaler = self.normalize(source)
        fitted, _ = self.normalize(fitted, scaler=target_scaler)
        return source, source_scaler, fitted
