from typing import Any, Callable

import numpy as np
import torch

import nibabel as nib

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from output_calculators import OutputCalculator
from datasets import create_input_space, create_input_space_prop
from dataclasses import dataclass
from dipy.data import get_sphere


@dataclass
class Trainer:
    model: torch.nn.Module
    dataloader: DataLoader
    loss_fn: Callable
    optimizer: torch.optim.Optimizer
    device: str
    epochs: int
    nr_fiber_directions: int
    data_shape: tuple[int, int, int]
    output_calculator: OutputCalculator

    log_freq: int = 100

    lambda_: float = 0
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    slice_id: int = 0
    grad_id: int = 0
    clip_grads: float = 0

    def __post_init__(self):
        self.model.to(self.device)

    def train(self) -> dict[str, Any]:
        # torch.autograd.set_detect_anomaly(True)

        sphere = get_sphere('repulsion100')
        input_dirs = sphere.vertices[sphere.vertices[:,2]>=0]
        self.input_dirs = torch.tensor(input_dirs, device=self.device)

        avg_loss = []
        t = 0
        best_loss = float("inf")
        best_model = None
        val_in, val_label = self.dataloader.dataset.get_val_set()
        if val_in is not None:
            val_in = val_in.to(self.device)
            val_label = val_label.to(self.device)

        for epoch in range(self.epochs):
            print(epoch)
            losses = []
            for i, (_input, labels, kwargs) in enumerate(tqdm(self.dataloader)):
                self.model.train()
                _input = _input.to(self.device)
                inputs = torch.zeros((_input.shape[0], self.input_dirs.shape[0], 6), device=self.device)
                inputs[..., :3] = _input[:, None, :]
                inputs[..., 3:] = self.input_dirs[None, :, :]
                inputs = inputs.reshape(-1, 6)

                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                model_out = self.model(inputs)

                output, *res = self.output_calculator.output_from_model_out(model_out, self.input_dirs, **kwargs)

                loss = self.loss_fn(output, labels, *res)

                loss.backward()

                if self.clip_grads != 0:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.clip_grads
                    )

                self.optimizer.step()

                loss_item = loss.item()
                #print(loss_item)
                losses.append(loss_item)
            mean_loss = np.array(losses).mean()
            sum_loss = np.array(losses).sum()

            if sum_loss < best_loss:
                print(
                    f"Training loss improved ({best_loss:.6f} -> {sum_loss:.6f}). Saving model."
                )
                best_loss = sum_loss
                best_model = self.model.state_dict()  # Save the best model weights

            if val_in is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(val_in)
                    output, *res = self.output_calculator.output_from_model_out(val_out)
                    val_loss = self.loss_fn(output, val_label, *res)
                    print(f"Validation loss epoch {epoch + 1}: {val_loss}")

            if self.scheduler:
                self.scheduler.step()
            avg_loss.append(mean_loss)
            t += 1

        return best_model


    def create_full_output_image(
        self, cfg: dict, rescale_value: np.ndarray | int | float = 1
    ) -> Any:
        nifti_img = nib.load(Path(cfg["paths"]["nifti"]))
        np_img = nifti_img.get_fdata()
        width, height, depth = np_img.shape[:3]

        coeff_outputs = []
        image_outputs = []
        self.dataloader = DataLoader(
        self.dataloader.dataset,
        batch_size=cfg["train_cfg"]["batch_size"],
        shuffle=False,
        num_workers=3,
        drop_last=False,
        )

        self.model.eval()
        with torch.no_grad():
            for _input, labels, kwargs in self.dataloader:
                _input = _input.to(self.device)
                inputs = torch.zeros((_input.shape[0], self.input_dirs.shape[0], 6), device=self.device)
                inputs[..., :3] = _input[:, None, :]
                inputs[..., 3:] = self.input_dirs[None, :, :]
                inputs = inputs.reshape(-1, 6)
                model_out = self.model(inputs)
                coeff_outputs.append(model_out.cpu().detach())

                output, *res = self.output_calculator.output_from_model_out(model_out, self.input_dirs, **kwargs)
                image_outputs.append(output.cpu().detach())

        coeffs = torch.concat(coeff_outputs).numpy()
        coeffs = coeffs.reshape(-1, self.input_dirs.shape[0]*3)

        diff_output = torch.concat(image_outputs).numpy()

        mask_path = cfg["paths"].get("mask", None)
        if mask_path:
            mask_img = nib.load(mask_path).get_fdata().astype(bool)

            coeff_image = np.zeros((width, height, depth, coeffs.shape[1]))
            coeff_image[mask_img, :] = coeffs

            diff_image = np.zeros_like(np_img)
            diff_image[mask_img, :] = diff_output

        return (
            nifti_img,
            coeff_image,
            diff_image
        )
