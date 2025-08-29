import math
from functools import lru_cache

import numpy as np
import torch
from scipy.special import sph_harm
from utils import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)

class Signal_Zeppelin:
    def __init__(
        self,
        bvals: np.ndarray,
        cart_bvec: np.ndarray,
        bdelta: np.ndarray,
        fiber_direction: np.ndarray,
        sph_bvec: np.ndarray = None,
        device: str = "cpu",
    ):
        self.device = device
        self.bvals = torch.tensor(bvals, device=device)
        self.bdelta = torch.tensor(bdelta, device=device)
        self.fiber_direction = torch.tensor(fiber_direction, device = device)

        if cart_bvec is not None:
            self.cart_bvec = cart_bvec
            self.sph_bvec = cartesian_to_spherical(cart_bvec)


    def compute_signal_from_coeff(self, coeffs: torch.tensor):

        weight = coeffs[:,0]
        Dpar = coeffs[:,1]
        Dperp = coeffs[:,2]

        inproduct2 = torch.mm(self.sph_bvec, self.fiber_direction)

        signal = weight  * torch.exp(
                ((Dpar - Dperp) * self.bvals * self.bdelta) / 3
                - ((Dpar + 2 * Dperp) * self.bvals) / 3
                - ((Dpar - Dperp) * self.bvals * self.bdelta * inproduct2)
                    )


        return signal



