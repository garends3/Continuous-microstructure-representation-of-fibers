from math import lgamma
from typing import Any
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils


# from exceptiongroup import catch
from scipy.special import sph_harm, erf, gamma,eval_legendre
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
        c_ias = analytical_sol_int(y, l)  # Extract the first column as c_ias

        # Calculate y = b * bd * (De - Dp)
        y_new = b * bdelta * (De - Dp)

        # Linearly interpolate clot and dclot at points y_new
        # tmp = np.interp(y_new, np.squeeze(bD), np.squeeze(clot[l]))
        # calculate analytically
        c_eas = analytical_sol_int(y_new, l)  # Extract the first column as c_eas

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

def integrand(x, alpha, l):

    # Laguerre on CPU
    x_np = x.detach().cpu().numpy()  # shape (N,)
    legendre_vals = eval_legendre(l, x_np)  # (N,)
    legendre_tensor = torch.tensor(legendre_vals, device=alpha.device)

    # Compute result
    result = torch.exp(-alpha * x**2) * legendre_tensor  # (B, V, N)

    return result  # (B, V, N)


def analytical_sol_int(a, n):
    # Ensure 'a' is a tensor
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)

    # Handle small 'a' values
    a_eps = 1e-6
    a_abs = torch.abs(a)

    a_eps_idx = (a_abs <= 1e-6)
    a_safe = a
    a_safe[a_eps_idx] = 1e-6  # TODO: is this the way?

    small_a_mask = a_abs <= a_eps

    # Compute the limiting behavior for small 'a'
    a_limit = 0.5 * gamma(n + 0.5) / gamma(2 * n + 1.5) * (-a_safe)**n

    # Ensure 'a_limit_tensor' is on the same device as 'a_safe'
    device = a_safe.device
    a_limit_tensor = torch.tensor(a_limit, dtype=torch.float32, device=device)

    # Define the integration method
    integrator = Simpson()

    # Perform the integration
    a_safe_unvec = torch.flatten(a_safe)
    result = integrator.integrate(lambda x: integrand(x.to(device), a_safe_unvec, n), dim=1, N=200, integration_domain=[[0.0, 1.0]])
    unflattened_result = result.view(a_safe.shape)


    # Combine results
    analytical_result = torch.where(small_a_mask, a_limit_tensor, unflattened_result)

    return analytical_result
