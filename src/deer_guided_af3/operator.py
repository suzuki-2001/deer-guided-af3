"""Differentiable forward operator: backbone coordinates -> spin-spin P(r)."""

from __future__ import annotations

import numpy as np
import torch


def backbone_frame(N, CA, CB, eps=1e-8):
    """Frame axes as columns, origin at CA: global = R @ local + t."""
    e1 = N - CA
    e1 = e1 / (e1.norm(dim=-1, keepdim=True) + eps)

    # Remove the component along N–CA to obtain an orthogonal second axis.
    v2 = CB - CA
    e2 = v2 - e1 * (v2 * e1).sum(-1, keepdim=True)
    e2 = e2 / (e2.norm(dim=-1, keepdim=True) + eps)

    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1), CA


def backbone_frame_np(N, CA, CB, eps=1e-8):
    """NumPy version of backbone_frame, used when building the spin-label clouds."""
    e1 = N - CA
    e1 = e1 / (np.linalg.norm(e1) + eps)

    v2 = CB - CA
    e2 = v2 - e1 * (v2 @ e1)
    e2 = e2 / (np.linalg.norm(e2) + eps)

    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1), CA


class LabelSite:
    """A site's spin centres in its local backbone frame, with the rotamer weights."""

    def __init__(self, s_local, weights, site):
        self.site = site
        self.s_local = np.asarray(s_local, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        self.weights = weights / weights.sum()


class RotamerPrOperator(torch.nn.Module):
    """P(r) for a set of labelled pairs, as a sum of Gaussians of width kappa_sigma_A."""

    def __init__(self, sites, pairs, r_grid, kappa_sigma_A=0.5, device="cpu", dtype=torch.float32):
        super().__init__()
        self.pairs = list(pairs)
        self.sites = list(sites)
        self.r = torch.tensor(np.asarray(r_grid), dtype=dtype, device=device)
        self.sigma = float(kappa_sigma_A)

        for key, site in sites.items():
            self.register_buffer(
                f"sloc_{key}", torch.tensor(site.s_local, dtype=dtype, device=device)
            )
            self.register_buffer(f"w_{key}", torch.tensor(site.weights, dtype=dtype, device=device))

    def centres(self, frame, site):
        """Place a site's local spin centres in the current backbone frame."""
        rotation, origin = backbone_frame(frame["N"], frame["CA"], frame["CB"])
        return torch.einsum("ij,rj->ri", rotation, getattr(self, f"sloc_{site}")) + origin

    def pair_pr(self, S_i, S_j, w_i, w_j):
        distances = torch.cdist(S_i, S_j)
        pair_weights = w_i[:, None] * w_j[None, :]

        # Broaden each rotamer-pair distance, then sum with its joint weight.
        kernels = torch.exp(-0.5 * ((self.r[:, None, None] - distances[None]) / self.sigma) ** 2)
        density = (kernels * pair_weights[None]).sum(dim=(1, 2))
        return density / (torch.trapezoid(density, self.r) + 1e-12)

    def forward(self, frames):
        """frames: site -> {'N', 'CA', 'CB'}. Returns pair -> P(r)."""
        centres = {site: self.centres(frames[site], site) for site in self.sites}
        return {
            (i, j): self.pair_pr(
                centres[i], centres[j], getattr(self, f"w_{i}"), getattr(self, f"w_{j}")
            )
            for (i, j) in self.pairs
        }
