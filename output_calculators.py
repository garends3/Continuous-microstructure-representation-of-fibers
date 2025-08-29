from functools import partial
from typing import Protocol, Any
from torch import Tensor

from diffusion_calculator import Signal_Zeppelin
from datasets import ZeppelinModelDataset
import numpy as np


class OutputCalculator(Protocol):
    def output_from_model_out(self, model_out: Any, **kwargs) -> Any:
        ...


class CoeffDiff:
    def __init__(self, diff_calculator: Any):
        self.diff_calculator = diff_calculator

    def output_from_model_out(self, model_out: Any, fiber_direction **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        diff_signal = self.diff_calculator.compute_signal_from_coeff(model_out, fiber_direction **kwargs)
        return diff_signal, model_out


def create_zeppelin_calculator(
    cfg: dict, dataset: ZeppelinModelDataset, device: str, **kwargs
) -> CoeffDiff:
    diff_calculator = Signal_Zeppelin(
        dataset.get_bvals(),
        dataset.get_directions(),
        dataset.get_bdelta(),
        device=device
    )
    return CoeffDiff(diff_calculator)


OUTPUT_CALCULATORS = {
    "zeppelin_model": partial(create_zeppelin_calculator, numerical=False),
}


def get_output_calculator(cfg: dict, **kwargs) -> OutputCalculator:
    constructor = None
    if cfg.get("output_calculator", None):
        constructor = OUTPUT_CALCULATORS.get(cfg["output_calculator"], None)

    if constructor is None:
        raise Exception("No output calculator for found")
    return constructor(cfg, **kwargs)
