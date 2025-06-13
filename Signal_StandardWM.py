from math import lgamma
from typing import Any
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils


# from exceptiongroup import catch
from scipy.special import sph_harm, erf, gamma,eval_laguerre
from torchquad import Simpson


def K2comp_fast(order, bvalues, bdelta, S0, Di, De, Dp, f):
    """
    not really sure yet what we are calculating here. Probably something with cross terms?
    Maybe we are calculating k up to second order. Look at expression k_l0/A.10 from Tax et al.

    input:
    see WM standard for parameters


    :return [K, dK]
    """

    # create dictionary for exponential factors due to diffusion signal decay
    K = torch.zeros(len(Di), len(bvalues), order // 2 + 1)
    b = bvalues.squeeze()
    bd = bdelta.squeeze()

    for i, l in enumerate(np.arange(0, order + 1, 2)):
        y = b * bdelta * Di

        # Linearly interpolate clot and dclot at points y # why do we need to interpolate.
        # tmp = np.interp(y, np.squeeze(bD), np.squeeze(clot[l]))
        c_ias = analytical_sol(y, l)  # Extract the first column as c_ias

        # Calculate y = b * bd * (De - Dp)
        y_new = b * bdelta * (De - Dp)

        # Linearly interpolate clot and dclot at points y_new
        # tmp = np.interp(y_new, np.squeeze(bD), np.squeeze(clot[l]))
        # calculate analytically
        c_eas = analytical_sol(y_new, l)  # Extract the first column as c_eas

        # do not really see what K and dK are. K seems logical but what is dK doing. Doesnt it miss a factor 2? # check with chantal if this expression is correct vs tax et al *
        K[:, :, i] = (
            2
            * torch.sqrt(torch.pi * (2 * torch.tensor(l) + 1))
            * S0
            * (
                f * torch.exp((Di * b * bd) / 3 - (Di * b) / 3) * c_ias
                + (1 - f)
                * torch.exp(
                    (De * b * bd) / 3
                    - (De * b) / 3
                    - (2 * Dp * b) / 3
                    - (Dp * b * bd) / 3
                )
                * c_eas
            )
        )

    return K


def analytical_sol(a, n):
    # function is not continious in a = 0, so approx by a -> 0.00001
    a_limit = 0.5*gamma(n+0.5)/gamma(2*n + 3/2)*(-a)**n
    a_eps_idx = (a <= 1e-6)
    a[a_eps_idx] = 1e-6  # TODO: is this the way?
    # a = a.clone().detach().to(dtype=torch.complex64) # why do we detach and cast to complex here?

    if n == 0:
        analytical_sol = (
            torch.sqrt(torch.tensor(torch.pi)) * torch.erf(torch.sqrt(a))
        ) / (2 * torch.sqrt(a))

    if n == 2:
        analytical_sol = (
            -6 * torch.sqrt(a) * torch.exp(-a)
            + (3 - 2 * a)
            * torch.sqrt(torch.tensor(torch.pi))
            * torch.erf(torch.sqrt(a))
        ) / (8 * a * torch.sqrt(a))

    if n == 4:
        term1 = (
            3
            * torch.sqrt(torch.tensor(torch.pi))
            * (4 * a**2 - 20 * a + 35)
            * torch.erf(torch.sqrt(a))
        ) / (64 * a ** (5 / 2))
        term2 = (5 * (2 * a + 21) * torch.exp(-a)) / (32 * a**2)
        analytical_sol = term1 - term2

    if n == 6:
        term1 = (
            5
            * torch.sqrt(torch.tensor(torch.pi))
            * (-8 * a**3 + 84 * a**2 - 378 * a + 693)
            * torch.erf(torch.sqrt(a))
        )
        term2 = 42 * torch.sqrt(a) * (4 * a**2 + 20 * a + 165) * torch.exp(-a)
        analytical_sol = (term1 - term2) / (256 * a ** (7 / 2))

    if n == 8:
        term1 = (
            35
            * torch.sqrt(torch.tensor(torch.pi))
            * (8 * a * (a * (2 * (a - 18) * a + 297) - 1287) + 19305)
            * torch.erf(torch.sqrt(a))
        )
        term2 = (
            6
            * torch.sqrt(a)
            * (2 * a * (2 * a * (62 * a + 1925) + 15015) + 225225)
            * torch.exp(-a)
        )
        analytical_sol = (term1 - term2) / (4096 * a ** (9 / 2))

    analytical_sol[a_eps_idx] = a_limit[a_eps_idx]

    return torch.nan_to_num(analytical_sol)


