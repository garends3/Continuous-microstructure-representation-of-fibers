import argparse

import torch
import nibabel as nib
import numpy as np

from utils import parse_cfg

from loss_functions import get_loss_function
from output_calculators import get_output_calculator
from ml_utils import Trainer
from models import get_model
from datasets import get_dataset
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from plotting import plot_maps, plot_scatter

def initialize_run(cfg_path = None):
    if cfg_path is not None:
        cfg = parse_cfg(Path(cfg_path))
    else:
        cfg = parse_cfg(Path("configs/test.yaml"))

    train_cfg = cfg["train_cfg"]

    width = cfg["width"]
    height = cfg["height"]
    depth = cfg["depth"]

    l_max = train_cfg["lmax"]
    lr = train_cfg["lr"]
    lambda_ = train_cfg["lambda"]
    log_freq = cfg["log_freq"]

    model = get_model(cfg)
    print(model)

    dataset = get_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=3,
        drop_last=True,
    )

    loss_fn = get_loss_function(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        StepLR(optimizer, step_size=1, gamma=0.99) if cfg["lr_scheduler"] else None
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    epochs = train_cfg["epochs"]

    output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        l_max=l_max,
        data_shape=(width, height, depth),
        output_calculator=output_calculator,
        log_freq=log_freq,
        lambda_=lambda_,
        scheduler=scheduler,
        clip_grads=cfg["clip_gradients"],
    )

    best_model = trainer.train()

    file_inf = cfg.get("experiment_name", "nameless")
    output_folder = Path(cfg["paths"]["output"])/file_inf

    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    torch.save(best_model, output_folder / f"model_{file_inf}.pt")
    (
        nifti_img,
        grad_img,
        coeff_image,
        sm_coeff,
    ) = trainer.create_full_output_image(cfg, dataset.get_scale())

    sm_coeff_img = nib.Nifti1Image(
        sm_coeff, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(sm_coeff_img, output_folder / f"sm_coeffs_{file_inf}.nii.gz")

    full_nifti_img = nib.Nifti1Image(
        grad_img, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(full_nifti_img, output_folder / f"grads_{file_inf}.nii.gz")

    full_coeff_img = nib.Nifti1Image(
        coeff_image, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(full_coeff_img, output_folder / f"coeffs_{file_inf}.nii.gz")

    plot = cfg.get("plot_cfg", None)
    if plot is not None:
        print('Creating plots')
        mask_path = cfg['paths'].get('mask', None)

        if mask_path:
            mask = nib.load(mask_path)
            mask_data = mask.get_fdata().astype(bool)

        gt_path = Path(plot['gt_path'])
        if gt_path.suffix in ['.gz', '.nii']:
            gt_params = nib.load(gt_path).get_fdata()
        else:
            gt_params = np.load(plot['gt_path'])

        gt_params = np.nan_to_num(gt_params)

        map_list = plot['map_list']
        ranges = plot['ranges']
        names = plot['names']
        n_slice = plot['n_slice']
        output_path = Path(plot['path'])/file_inf

        if not output_path.exists():
            output_path.mkdir(parents=True)

        dwi = nib.load(plot['background_path']).get_fdata()

        plot_scatter(sm_coeff, gt_params, mask_data, map_list, names, output_path/'sm_par_scatter.png', min_v=0)
        plot_maps(sm_coeff, gt_params, dwi, mask_data, names, map_list, ranges, n_slice, output_path/'maps.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FOD Nesh",
        description="Fits FOD nesh",
        epilog="This fits FOD Nesh"
    )
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args = parser.parse_args()

    initialize_run(args.config)
