from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import (
    parse_bvecs,
    parse_bvecs_col,
    parse_bvals,
    parse_bvals_col,
    parse_response,
    parse_mrtrix,
    parse_c_bvecs_bvals_bdeltas,
)


def create_input_space(width: int, height: int, depth: int) -> torch.Tensor:
    p_width = 2 / width
    p_height = 2 / height
    p_depth = 2 / depth

    x = torch.linspace(-1 + p_width / 2, (1 - p_width / 2), width)
    y = torch.linspace(-1 + p_height / 2, (1 - p_height / 2), height)
    z = torch.linspace(-1 + p_depth / 2, (1 - p_depth / 2), depth)

    input_tensor = torch.cartesian_prod(x, y, z)

    return input_tensor.reshape(width, height, depth, 3)

def create_input_space_prop(width: int, height: int, depth: int) -> torch.Tensor:
    dims = np.array([width, height, depth])
    max_dim = np.max(dims)
    p_size = 2 / max_dim

    sizes = p_size*(dims - 1)

    x = torch.linspace(-sizes[0]/2, sizes[0]/2, width)
    y = torch.linspace(-sizes[1]/2, sizes[1]/2, height)
    z = torch.linspace(-sizes[2]/2, sizes[2]/2, depth)

    input_tensor = torch.cartesian_prod(x, y, z)

    return input_tensor.reshape(width, height, depth, 3)

def create_input_space_prop_upsampled(
    train_width: int, train_height: int, train_depth: int,
    pred_width: int, pred_height: int, pred_depth: int
) -> torch.Tensor:

    train_dims = np.array([train_width, train_height, train_depth])
    max_dim = np.max(train_dims)
    p_size = 2 / max_dim
    sizes = p_size * (train_dims - 1)

    x = torch.linspace(-sizes[0] / 2, sizes[0] / 2, pred_width)
    y = torch.linspace(-sizes[1] / 2, sizes[1] / 2, pred_height)
    z = torch.linspace(-sizes[2] / 2, sizes[2] / 2, pred_depth)

    input_tensor = torch.cartesian_prod(x, y, z)
    return input_tensor.reshape(pred_width, pred_height, pred_depth, 3)

def get_dwi_indices(bvals: np.ndarray, bval: float, delta: float):
    bval_low = bval - delta
    bval_high = bval + delta

    return np.nonzero((bval_low < bvals) & (bvals < bval_high))[0]

def get_mean_b0(img: np.ndarray, b0_idx: np.ndarray):
    if len(b0_idx) == 0:
        raise Exception("No b0 images found")
    b0_imgs = img[..., b0_idx]
    if b0_idx.sum() == 1:
        return b0_imgs
    return b0_imgs.mean(axis=-1, keepdims=True)


class DiffusionDataset(Dataset):
    def get_bvals(self):
        pass

    def get_bdelta(self):
        pass

    def get_scale(self) -> float:
        pass

    def get_directions(self) -> np.array:
        pass


class ZeppelinModelDataset(DiffusionDataset):
    def __init__(
        self,
        bvec_path: Path,
        bval_path: Path,
        bdelta_path: Path,
        bval_delta: float,
        nifti_path: Path,
        mask_path: Path = None,
        scale: str | int | float = 1,
        val_size: int = 0,
    ) -> None:
        nifti_file = nib.load(nifti_path)
        full_img = nifti_file.get_fdata()
        full_img[full_img<1e-6] = 1e-6
        full_img = np.nan_to_num(full_img)

        self.bvals = parse_bvals_col(bval_path)
        self.cart_bvecs = parse_bvecs_col(bvec_path)
        self.bdelta = parse_bvals_col(bdelta_path)

        voxel_wise_norm = False

        if isinstance(scale, int) or isinstance(scale, float):
            self.scale_values = scale
        elif scale == "voxel_norm":
            voxel_wise_norm = True
            b0_idx = (self.bvals < bval_delta) & (self.bdelta == 1)
            self.scale_values = np.percentile(
                full_img[..., b0_idx], 95, axis=-1, keepdims=True
            )

            self.scale_values[self.scale_values < 1] = 1
        elif scale == "norm":
            b0_idx = (self.bvals < bval_delta) & (self.bdelta == 1)
            self.scale_values = np.percentile(
                full_img[..., b0_idx], 95
            )

        width, height, depth, n_grad = full_img.shape
        self.input_tensor = create_input_space_prop(width, height, depth)

        output_array = full_img

        if mask_path:
            mask_data = nib.load(mask_path).get_fdata().astype(bool)
            self.input_tensor = self.input_tensor[mask_data]
            output_array = output_array[mask_data]

            if voxel_wise_norm:
                self.scale_values = self.scale_values[mask_data]
        else:
            self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
            output_array = output_array.reshape(width * height * depth, n_grad)

        output_array = output_array / self.scale_values
        if val_size > 0:
            val_idx = np.random.choice(output_array.shape[0], output_array.shape[0]//100*val_size, replace=False)
            bool_arr = np.zeros(output_array.shape[0], dtype=bool)
            bool_arr[val_idx] = 1

            self.val_input = self.input_tensor[bool_arr, :]
            self.val_output = torch.tensor(output_array[bool_arr, :])

            self.input_tensor = self.input_tensor[~bool_arr, :]
            output_array = output_array[~bool_arr, :]
        else:
            self.val_input, self.val_output = None, None

        self.output_tensor = torch.tensor(output_array, dtype=torch.float)

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.input_tensor[idx], self.output_tensor[idx], {}

    def get_val_set(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.val_input, self.val_output

    def get_bvals(self) -> np.ndarray:
        return self.bvals

    def get_bdelta(self) -> np.ndarray:
        return self.bdelta

    def get_scale(self) -> float:
        return self.scale_values

    def get_directions(self) -> np.ndarray:
        return self.cart_bvecs
    
    

def create_zeppelin(cfg: dict) -> ZeppelinModelDataset:
    mask_path = Path(cfg["paths"]["mask"]) if cfg["paths"]["mask"] else None

    dataset = ZeppelinModelDataset(
        bvec_path=Path(cfg["paths"].get("fsl_bvecs", None)),
        bval_path=Path(cfg["paths"].get("fsl_bvals", None)),
        bdelta_path=Path(cfg["paths"].get("fsl_bdelta", None)),
        bval_delta=cfg["bval_delta"],
        nifti_path=Path(cfg["paths"]["nifti"]),
        mask_path=mask_path,
        scale=cfg["scale_data"],
        val_size=cfg.get("val_size", 0)
    )

    return dataset




DATASETS = {
    "zeppelin": partial(create_zeppelin)
}


def get_dataset(cfg: dict) -> DiffusionDataset:
    constructor = DATASETS.get(cfg["dataset_name"], None)
    if constructor is None:
        raise Exception("Dataset name not recognized")
    return constructor(cfg)
