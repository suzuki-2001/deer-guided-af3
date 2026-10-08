"""Boltz-2 sampling with diffusion conditioning recomputed from the current z."""

from __future__ import annotations

import json
from dataclasses import asdict
from math import sqrt
from pathlib import Path

import numpy as np
import torch

from . import clouds

CACHE = Path.home() / ".boltz"


def load(checkpoint=None, cache=CACHE, recycling=3, n_step=200, device="cuda"):
    """Load the pretrained model with its weights frozen."""
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2

    torch.set_grad_enabled(True)
    diffusion = Boltz2DiffusionParams()
    diffusion.step_scale = 1.5

    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False

    model = Boltz2.load_from_checkpoint(
        str(checkpoint or Path(cache) / "boltz2_conf.ckpt"),
        strict=True,
        map_location="cpu",
        predict_args=dict(
            recycling_steps=recycling,
            sampling_steps=n_step,
            diffusion_samples=1,
            max_parallel_samples=1,
            write_confidence_summary=False,
            write_full_pae=False,
            write_full_pde=False,
        ),
        diffusion_process_args=asdict(diffusion),
        ema=False,
        use_kernels=True,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(
            MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
        ),
        steering_args=asdict(steering),
    )
    return dict(
        model=model.to(device).eval().requires_grad_(False),
        device=device,
        cache=Path(cache),
        recycling=recycling,
    )


def target(context, processed, pairs=(), label_sites=None):
    """Features, trunk output and forward operator, from inputs Boltz has already processed."""
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest

    model, device = context["model"], context["device"]
    processed = Path(processed)
    manifest = Manifest.load(processed / "manifest.json")
    if len(manifest.records) != 1:
        raise ValueError("Boltz inputs must contain exactly one target")
    structure_file = processed / "structures" / f"{manifest.records[0].id}.npz"

    data = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=context["cache"] / "mols",
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
        override_method="other",
        num_workers=0,
    )
    data.setup("predict")
    batch = next(iter(data.predict_dataloader()))
    feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # Evaluate the trunk once; sampling updates a copy of its pair representation.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        s_inputs = model.input_embedder(feats)
        s_init = model.s_init(s_inputs)
        z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
        relp = model.rel_pos(feats)

        z_init = z_init + relp + model.token_bonds(feats["token_bonds"].float())
        if model.bond_type_feature:
            z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + model.contact_conditioning(feats)

        s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        for _ in range(context["recycling"] + 1):
            s = s_init + model.s_recycle(model.s_norm(s))
            z = z_init + model.z_recycle(model.z_norm(z))
            if model.use_templates:
                z = z + model.template_module(z, feats, pair_mask, use_kernels=model.use_kernels)
            z = z + model.msa_module(z, s_inputs, feats, use_kernels=model.use_kernels)
            s, z = model.pairformer_module(
                s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels
            )

    s, s_inputs, z, relp = (
        s.float().detach(),
        s_inputs.float().detach(),
        z.float().detach(),
        relp.float().detach(),
    )

    order = atom_order(structure_file)
    pad = feats["atom_pad_mask"][0].bool()
    if len(order) != int(pad.sum()):
        raise ValueError(f"{len(order)} atoms in the processed inputs, {int(pad.sum())} unpadded")

    index = frame_index(order, {site for pair in pairs for site in pair})
    operator, kept = (None, [])
    if label_sites:
        operator, kept = clouds.operator(label_sites, pairs, index, device)

    def conditioning(z_current):
        q, c, to_keys, enc, dec, trans = model.diffusion_conditioning(
            s_trunk=s, z_trunk=z_current, relative_position_encoding=relp, feats=feats
        )
        return dict(
            q=q,
            c=c,
            to_keys=to_keys,
            atom_enc_bias=enc,
            atom_dec_bias=dec,
            token_trans_bias=trans,
        )

    return dict(
        feats=feats,
        pad=pad,
        order=order,
        residue=residue_names(structure_file),
        index=index,
        operator=operator,
        pairs=kept,
        s=s,
        s_inputs=s_inputs,
        z=z,
        conditioning=conditioning,
    )


def _structure_file(path):
    path = Path(path)
    if path.is_dir():
        records = json.loads((path / "manifest.json").read_text())["records"]
        if len(records) != 1:
            raise ValueError("Boltz inputs must contain exactly one target")
        return path / "structures" / f"{records[0]['id']}.npz"
    return path


def atom_order(processed):
    """[(chain, residue number, atom name)] in the order Boltz emits atoms."""
    with np.load(_structure_file(processed), allow_pickle=False) as stored:
        atoms, residues, chains = stored["atoms"], stored["residues"], stored["chains"]

    order = []
    for chain in chains:
        residue_start = int(chain["res_idx"])
        chain_residues = residues[residue_start : residue_start + int(chain["res_num"])]
        for residue in chain_residues:
            start = int(residue["atom_idx"])
            for atom in atoms[start : start + int(residue["atom_num"])]:
                order.append((str(chain["name"]), int(residue["res_idx"]) + 1, str(atom["name"])))

    return order


def residue_names(processed):
    """Map each chain and residue number to its three-letter residue name."""
    with np.load(_structure_file(processed), allow_pickle=False) as stored:
        residues, chains = stored["residues"], stored["chains"]

    names = {}
    for chain in chains:
        residue_start = int(chain["res_idx"])
        chain_residues = residues[residue_start : residue_start + int(chain["res_num"])]
        for residue in chain_residues:
            key = (str(chain["name"]), int(residue["res_idx"]) + 1)
            names[key] = str(residue["name"])
    return names


def frame_index(order, sites):
    """site -> atom indices of N, CA, CB in the unpadded coordinates."""
    from .restraints import parse_site

    by_chain, by_residue = {}, {}
    for i, (chain, resid, name) in enumerate(order):
        by_chain[(chain, resid, name)] = i
        by_residue.setdefault((resid, name), []).append(i)

    index = {}
    for site in sites:
        chain, resid = parse_site(site)
        found = {}
        for name in ("N", "CA", "CB"):
            matches = by_residue.get((resid, name), [])
            if chain is None and len(matches) > 1:
                raise ValueError(f"site {site} is present on multiple chains; specify its chain")
            if chain:
                found[name] = by_chain.get((chain, resid, name))
            else:
                found[name] = matches[0] if matches else None

        if all(v is not None for v in found.values()):
            index[site] = found
    return index


def sample(context, target, guidance=None, *, members=25, n_step=200, seed=101):
    """Sample an ensemble using a shared pair representation."""
    from boltz.model.loss.diffusionv2 import weighted_rigid_align
    from boltz.model.modules.utils import compute_random_augmentation

    diffusion, device = context["model"].structure_module, context["device"]
    feats, pad = target["feats"], target["pad"]
    mask = feats["atom_pad_mask"].float()
    shape = (*mask.shape, 3)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    sigmas = diffusion.sample_schedule(n_step)
    gammas = torch.where(sigmas > diffusion.gamma_min, diffusion.gamma_0, 0.0)

    @torch.no_grad()
    def augment(x):
        rotation, translation = compute_random_augmentation(1, device=device, dtype=x.dtype)
        x = x - x.mean(dim=-2, keepdims=True)
        return torch.einsum("bmd,bds->bms", x, rotation) + translation

    def denoise(x_noisy, t, conditioning):
        return diffusion.preconditioned_network_forward(
            x_noisy,
            t,
            network_condition_kwargs=dict(
                s_inputs=target["s_inputs"],
                s_trunk=target["s"],
                feats=feats,
                multiplicity=1,
                diffusion_conditioning=conditioning,
            ),
        )

    def guided_denoise(z, x_noisy, t):
        return denoise(x_noisy, t, target["conditioning"](z))

    x = [sigmas[0] * torch.randn(shape, device=device) for _ in range(members)]
    with torch.no_grad():
        # Unguided sampling can reuse conditioning; guidance must recompute it from z.
        fixed_conditioning = None if guidance is not None else target["conditioning"](target["z"])

    with torch.set_grad_enabled(guidance is not None):
        for last, sigma, gamma in zip(sigmas[:-1], sigmas[1:], gammas[1:], strict=True):
            last, sigma = last.item(), sigma.item()
            t_hat = last * (1 + gamma.item())
            noise_level = sqrt(max(diffusion.noise_scale**2 * (t_hat**2 - last**2), 0.0))
            guiding = guidance is not None and guidance.guided(sigma)
            z = guidance.z if guidance is not None else target["z"]
            conditioning = fixed_conditioning
            if not guiding and conditioning is None:
                with torch.no_grad():
                    conditioning = target["conditioning"](z.detach())

            noisy, clean = [], []
            for member in range(members):
                x_noisy = augment(x[member]) + noise_level * torch.randn(shape, device=device)

                if guiding:
                    denoised = torch.utils.checkpoint.checkpoint(
                        guided_denoise, z, x_noisy, t_hat, use_reentrant=False
                    )
                else:
                    with torch.no_grad():
                        denoised = denoise(x_noisy, t_hat, conditioning)

                noisy.append(x_noisy)
                clean.append(denoised)

            if guiding:
                # Remove padded atoms before evaluating the spin-label distributions.
                guidance.step(torch.cat([c[0][pad][None] for c in clean], dim=0), sigma)
                clean = [c.detach() for c in clean]

            for member in range(members):
                x_noisy, denoised = noisy[member], clean[member]
                if diffusion.alignment_reverse_diff:
                    with torch.autocast("cuda", enabled=False):
                        x_noisy = weighted_rigid_align(
                            x_noisy.float(), denoised.float(), mask.float(), mask.float()
                        )
                    x_noisy = x_noisy.to(denoised)

                x[member] = x_noisy + diffusion.step_scale * (sigma - t_hat) * (
                    (x_noisy - denoised) / t_hat
                )

    return torch.cat(x, dim=0).detach()[:, pad]


def write(target, coords, paths):
    """One PDB per structure, in the order Boltz emits atoms."""
    written = []
    for path, xyz in zip(paths, coords.float().cpu().numpy(), strict=True):
        if not np.all(np.isfinite(xyz)):
            raise FloatingPointError(f"non-finite coordinates for {path}")

        lines = []
        for i, (chain, resid, name) in enumerate(target["order"]):
            x, y, z = xyz[i]
            atom = f" {name:<3}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {i + 1:5d} {atom} {target['residue'][(chain, resid)]:>3} {chain}"
                f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {name[0]:>2}"
            )

        Path(path).write_text("\n".join(lines + ["END"]) + "\n")
        written.append(path)
    return written
