"""Protenix-v1 sampling with updates to the shared pair representation."""

from __future__ import annotations

import numpy as np
import torch

from . import clouds

GAMMA0, GAMMA_MIN = 0.8, 1.0
NOISE_SCALE, STEP_SCALE = 1.003, 1.5


def load(model_name="protenix_base_default_v1.0.0", n_cycle=4, n_step=200, device="cuda"):
    """Load the pretrained model with its weights frozen."""
    from runner.batch_inference import get_default_runner

    runner = get_default_runner(
        seeds=[101],
        n_cycle=n_cycle,
        n_step=n_step,
        n_sample=1,
        dtype="fp32",
        model_name=model_name,
        use_msa=True,
        enable_cache=False,
        enable_fusion=True,
        enable_tf32=True,
    )
    model = runner.model.to(device).eval().requires_grad_(False)
    return dict(runner=runner, model=model, device=device, n_cycle=n_cycle)


def target(context, input_json, pairs=(), label_sites=None):
    """Features, trunk output and forward operator for one system."""
    from protenix.data.inference.infer_dataloader import get_inference_dataloader
    from protenix.model.protenix import update_input_feature_dict
    from runner.inference import update_inference_configs

    model, runner, device = context["model"], context["runner"], context["device"]
    runner.configs.input_json_path = str(input_json)
    data, atom_array, error = next(iter(get_inference_dataloader(configs=runner.configs)))[0]
    if error:
        raise ValueError(f"could not prepare {input_json}: {error}")

    runner.update_model_configs(update_inference_configs(runner.configs, data["N_token"].item()))
    features = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in data["input_feature_dict"].items()
    }
    features = update_input_feature_dict(model.relative_position_encoding.generate_relp(features))

    index = clouds.frame_index(atom_array, {s for pair in pairs for s in pair})
    operator, kept = (None, [])
    if label_sites:
        operator, kept = clouds.operator(label_sites, pairs, index, device)

    # Evaluate the trunk once; sampling updates a copy of its pair representation.
    with torch.no_grad():
        s_inputs, s_trunk, z = model.get_pairformer_output(
            input_feature_dict=features,
            N_cycle=context["n_cycle"],
            inplace_safe=False,
            chunk_size=None,
        )

    return dict(
        features=features,
        atom_array=atom_array,
        index=index,
        operator=operator,
        pairs=kept,
        s_inputs=s_inputs.detach(),
        s_trunk=s_trunk.detach(),
        z=z.detach(),
    )


def sample(context, target, guidance=None, *, members=25, n_step=200, seed=101):
    """Sample an ensemble using a shared pair representation."""
    from protenix.model.utils import centre_random_augmentation

    model, device = context["model"], context["device"]
    features, s_inputs, s_trunk = target["features"], target["s_inputs"], target["s_trunk"]
    n_atom = features["atom_to_token_idx"].size(-1)
    dtype = s_inputs.dtype

    schedule = model.inference_noise_scheduler(N_step=n_step, device=device, dtype=torch.float32)
    generator = _seed(seed, device)

    def noise(shape):
        return torch.randn(size=shape, device=device, dtype=dtype, generator=generator)

    def denoise(z, x_noisy, t):
        return model.diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=t,
            input_feature_dict=features,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z,
            pair_z=None,
            p_lm=None,
            c_l=None,
            chunk_size=None,
            inplace_safe=False,
            enable_efficient_fusion=False,
        )

    x = [schedule[0] * noise((1, n_atom, 3)) for _ in range(members)]

    with torch.set_grad_enabled(guidance is not None):
        for last, sigma in zip(schedule[:-1], schedule[1:], strict=True):
            t_hat = last * ((GAMMA0 if sigma > GAMMA_MIN else 0.0) + 1)
            noise_level = torch.sqrt(t_hat**2 - last**2)
            t = t_hat.reshape(1).expand(1).to(dtype)
            guiding = guidance is not None and guidance.guided(float(sigma))
            z = guidance.z if guidance is not None else target["z"]

            noisy, clean = [], []
            for member in range(members):
                augmented = (
                    centre_random_augmentation(x_input_coords=x[member], N_sample=1)
                    .squeeze(-3)
                    .to(dtype)
                )
                x_noisy = augmented + NOISE_SCALE * noise_level * noise(augmented.shape)

                if guiding:
                    denoised = torch.utils.checkpoint.checkpoint(
                        denoise, z, x_noisy, t, use_reentrant=False
                    )
                else:
                    with torch.no_grad():
                        denoised = denoise(z.detach(), x_noisy, t)

                noisy.append(x_noisy)
                clean.append(denoised)

            if guiding:
                # Fit one shared z to the ensemble's predicted distributions.
                guidance.step(torch.cat(clean, dim=0), float(sigma))
                clean = [c.detach() for c in clean]

            for member in range(members):
                delta = (noisy[member] - clean[member]) / t[..., None, None]
                x[member] = noisy[member] + STEP_SCALE * (sigma - t_hat)[..., None, None] * delta

    return torch.cat(x, dim=0).detach()


def write(target, coords, paths):
    """One PDB per structure, in the predictor's own atom order."""
    from biotite.structure.io.pdb import PDBFile

    written = []
    for path, xyz in zip(paths, coords.float().cpu().numpy(), strict=True):
        if not np.all(np.isfinite(xyz)):
            raise FloatingPointError(f"non-finite coordinates for {path}")

        atoms = target["atom_array"].copy()
        atoms.coord = xyz.astype(np.float32)
        out = PDBFile()
        out.set_structure(atoms)
        out.write(str(path))
        written.append(path)
    return written


def _seed(seed, device):
    """Seed diffusion noise and coordinate augmentation."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return torch.Generator(device=device).manual_seed(int(seed))
