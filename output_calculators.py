from functools import partial
from typing import Protocol, Any
from torch import Tensor

from diffusion_calculator import SignalFromCoefficients, SignalMultishell, SignalSM, GradCorSM
from datasets import DatasetCoord3D, MultiShellDataset

import numpy as np


class OutputCalculator(Protocol):
    def output_from_model_out(self, model_out: Any, **kwargs) -> Any:
        ...


class CoeffDiff:
    def __init__(self, diff_calculator: Any):
        self.diff_calculator = diff_calculator

    def output_from_model_out(self, model_out: Any, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        diff_signal = self.diff_calculator.compute_signal_from_coeff(model_out, **kwargs)
        neg_signal = self.diff_calculator.compute_negative_signal(model_out, **kwargs)
        return diff_signal, model_out, neg_signal


def create_sm_calculator(
    cfg: dict, dataset: MultiShellDataset, device: str, numerical: bool = False, **kwargs
) -> CoeffDiff:
    diff_calculator = SignalSM(
        cfg["train_cfg"]["lmax"],
        dataset.get_bvals(),
        dataset.get_directions(),
        dataset.get_bdelta(),
        device=device,
        numerical=numerical
    )
    return CoeffDiff(diff_calculator)

def create_grad_cor(
    cfg: dict, dataset: MultiShellDataset, device: str, numerical: bool = False, **kwargs
) -> CoeffDiff:
    diff_calculator = GradCorSM(
        cfg["train_cfg"]["lmax"],
        device=device,
        numerical=numerical,
    )
    return CoeffDiff(diff_calculator)

OUTPUT_CALCULATORS = {
    "standard_model": partial(create_sm_calculator, numerical=False),
    "standard_model_num": partial(create_sm_calculator, numerical=True),
    "grad_cor": partial(create_grad_cor, numerical=False),
    "grad_cor_num": partial(create_grad_cor, numerical=True),
}


def get_output_calculator(cfg: dict, **kwargs) -> OutputCalculator:
    constructor = None
    if cfg.get("output_calculator", None):
        constructor = OUTPUT_CALCULATORS.get(cfg["output_calculator"], None)

    if constructor is None:
        raise Exception("No output calculator for found")
    return constructor(cfg, **kwargs)
