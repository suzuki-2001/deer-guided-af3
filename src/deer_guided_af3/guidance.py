"""The objective and the Adam updates on the pair representation."""

from __future__ import annotations

import math

import torch

SIGMA_DATA = 16.0  # angstrom, the EDM weight's centre
P_MEAN, P_STD = -1.2, 1.5


def wasserstein(p, q, r):
    """1-D Wasserstein distance on a shared grid."""
    return torch.abs(torch.cumsum(p, 0) - torch.cumsum(q, 0)).sum() * (r[1] - r[0])


def edm_weight(sigma, sigma_data=SIGMA_DATA, p_mean=P_MEAN, p_std=P_STD):
    """EDM-inspired guidance weight in log-noise space."""
    if sigma <= 0:
        return 0.0
    u = (math.log(sigma / sigma_data) - p_mean) / p_std
    w = math.exp(-0.5 * u * u)
    return w if w > 1e-3 else 0.0


class Guidance:
    """One pair representation, shared by the ensemble and fitted to the measured curves.

    frame_index: site -> atom indices of N, CA, CB. measured: pair -> P(r) on the operator's grid.
    """

    def __init__(
        self,
        z,
        operator,
        measured,
        frame_index,
        *,
        lr=0.1,
        anchor=1.0,
        sigma_data=SIGMA_DATA,
        p_mean=P_MEAN,
        p_std=P_STD,
    ):
        self.z = z.detach().clone().requires_grad_(True)
        self.z_ref = z.detach()
        self.scale = float((self.z_ref**2).mean())
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("the pair representation must have positive finite scale")

        self.op = operator
        self.measured = measured
        self.index = frame_index
        self.lr, self.anchor = lr, anchor
        self.sigma_data, self.p_mean, self.p_std = sigma_data, p_mean, p_std

        self.opt = torch.optim.Adam([self.z], lr=lr)
        self.log = []

    def guided(self, sigma):
        return self.weight(sigma) > 0.0

    def weight(self, sigma):
        return edm_weight(float(sigma), self.sigma_data, self.p_mean, self.p_std)

    def mixture(self, x):
        """P(r) of the ensemble mixture, from x of shape [M, n_atom, 3]."""
        predictions = []
        for coordinates in x:
            frames = {
                site: {name: coordinates[index] for name, index in self.index[site].items()}
                for site in self.op.sites
            }
            predictions.append(self.op(frames))

        return {
            pair: sum(prediction[pair] for prediction in predictions) / len(predictions)
            for pair in self.measured
        }

    def objective(self, x):
        r = self.op.r
        mixture = self.mixture(x)
        fit = sum(
            wasserstein(density / (torch.trapezoid(density, r) + 1e-9), self.measured[pair], r)
            for pair, density in mixture.items()
        ) / len(mixture)

        # anchor = 1 penalises a one per cent move by one, on the scale of the fit.
        pull = self.anchor * 1e4 * ((self.z - self.z_ref) ** 2).mean() / self.scale
        return fit, pull

    def step(self, x, sigma):
        """One Adam update at the weighted step size. False if this noise level is not guided."""
        w = self.weight(sigma)
        if w <= 0.0:
            return False

        for group in self.opt.param_groups:
            group["lr"] = self.lr * w

        fit, pull = self.objective(x)
        loss = fit + pull
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite objective at sigma {float(sigma):.3f}")

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.log.append(
            dict(sigma=float(sigma), fit=float(fit.detach()), anchor=float(pull.detach()))
        )
        return True
