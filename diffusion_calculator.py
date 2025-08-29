import math
from functools import lru_cache

import numpy as np
import torch
from dipy.data import get_sphere
from scipy.special import sph_harm
from utils import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)
import Signal_StandardWM_numerical
import Signal_StandardWM
import gradcor_num_StandardWM


def create_y_mat(thetas: np.ndarray, phis: np.ndarray, l_max: int) -> torch.Tensor:
    n_dir = thetas.shape[0]
    n = (l_max + 1) * (l_max + 2) // 2

    y_mat = torch.zeros((n_dir, n))
    for l in range(0, l_max + 1, 2):
        for m in range(-l, l + 1):
            coef_idx = (l**2 + l) // 2 + m
            Y = sph_harm(np.abs(m), l, thetas, phis)
            if m < 0:
                y_mat[:, coef_idx] = torch.tensor(Y.imag * np.sqrt(2))
            if m > 0:
                y_mat[:, coef_idx] = torch.tensor(Y.real * np.sqrt(2))
            if m == 0:
                y_mat[:, coef_idx] = torch.tensor(Y.real)

    return y_mat



def get_rescale_value(l_max: int, rescale: bool = True) -> torch.Tensor:
    rescale_value = (
        torch.sqrt(4 * np.pi / (2 * torch.arange(0, l_max + 1, 2) + 1))
        if rescale
        else torch.ones(l_max // 2 + 1)
    )
    return rescale_value


def create_conv_vec(
    l_max: int, resp_coeff: torch.Tensor, rescale: bool = True
) -> torch.Tensor:
    rescale_value = get_rescale_value(l_max, rescale)
    conv_vec = torch.zeros(
        np.shape(resp_coeff)[0], np.shape(resp_coeff)[1], (l_max + 1) * (l_max + 2) // 2
    )
    for i, l in enumerate(range(0, l_max + 1, 2)):
        for m in range(-l, l + 1):
            coef_idx = (l**2 + l) // 2 + m
            conv_vec[:, :, coef_idx] = resp_coeff[:, :, i] * rescale_value[i]
    return conv_vec


class SignalSM:
    def __init__(
        self,
        l_max: int,
        bvals: np.ndarray,
        cart_bvec: np.ndarray,
        bdelta: np.ndarray,
        sph_bvec: np.ndarray = None,
        device: str = "cpu",
        numerical: bool = False,
    ):
        self.l_max = l_max
        self.device = device
        self.numerical = numerical
        self.bvals = torch.tensor(bvals, device=device)
        self.bdelta = torch.tensor(bdelta, device=device)

        self.y_mat = None

        if cart_bvec is not None:
            self.cart_bvec = cart_bvec
            sph_bvec = cartesian_to_spherical(cart_bvec)

        if sph_bvec is not None:
            self.sph_bvec = sph_bvec
            if cart_bvec is None:
                self.cart_bvec = spherical_to_cartesian(sph_bvec)

        sample_phis = sph_bvec[..., 1]
        sample_thetas = sph_bvec[..., 2]
        self.y_mat = create_y_mat(sample_thetas, sample_phis, l_max).to(device)

        sphere = get_sphere("repulsion200")  # 100, 200, 724
        sph_sphere_vecs = cartesian_to_spherical(sphere.vertices)
        sphere_thetas = sph_sphere_vecs[:, 2]
        sphere_phis = sph_sphere_vecs[:, 1]
        self.sphere_y_mat = create_y_mat(sphere_thetas, sphere_phis, l_max).to(device)

    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        fod_coeffs = coeffs[:, :-5]

        if self.numerical:
            kernel_method = Signal_StandardWM_numerical.K2comp_fast
        else:
            kernel_method = Signal_StandardWM.K2comp_fast
        kernel = kernel_method(
            self.l_max,
            self.bvals,
            self.bdelta,
            S0=coeffs[:, [-1]],
            Di=coeffs[:, [-5]],
            De=coeffs[:, [-3]],
            Dp=coeffs[:, [-2]],
            f=coeffs[:, [-4]],
        )
        conv_vec = create_conv_vec(self.l_max, kernel).to(self.device)

        fod_signal = torch.einsum(
            "bk, bdk, dk -> bd", fod_coeffs, conv_vec, self.y_mat
        )

        return fod_signal

    def compute_negative_signal(self, coeffs: torch.Tensor, **kwargs):
        fod_coeffs = coeffs[:, :-5]
        return torch.clamp(
             torch.einsum("bk, dk -> bd", fod_coeffs, self.sphere_y_mat), max=0
        )


