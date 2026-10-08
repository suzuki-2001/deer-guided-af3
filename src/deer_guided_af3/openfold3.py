"""OpenFold3-preview2 sampling with updates to the shared pair representation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import clouds


def load(
    checkpoint="openfold3-p2-155k",
    runner_yaml=None,
    out_dir="output/of3_work",
    device="cuda",
    n_step=200,
):
    """Load frozen model weights; the diffusion step count is applied in sample()."""
    from openfold3.core.config import config_utils
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig

    if runner_yaml:
        args = config_utils.load_yaml(Path(runner_yaml))
    else:
        args = {
            "model_update": {
                "presets": ["predict"],
                "custom": {
                    "settings": {"memory": {"eval": {"use_deepspeed_evo_attention": False}}}
                },
            }
        }

    args.setdefault("experiment_settings", {})["seeds"] = [101]
    args.setdefault("data_module_args", {})["data_seed"] = 101
    args["data_module_args"].setdefault("num_workers", 0)

    runner = InferenceExperimentRunner(
        InferenceExperimentConfig(inference_ckpt_name=checkpoint, **args),
        num_diffusion_samples=1,
        num_model_seeds=None,
        use_msa_server=False,
        use_templates=False,
        output_dir=Path(out_dir),
    )
    runner.setup()

    memory = runner.model_config.settings.memory.eval
    return dict(
        runner=runner,
        model=runner.lightning_module.model.to(device).eval().requires_grad_(False),
        config=runner.model_config,
        device=device,
        kernels=dict(
            use_deepspeed_evo_attention=memory.use_deepspeed_evo_attention,
            use_triton_triangle_kernels=memory.use_triton_triangle_kernels,
            use_cueq_triangle_kernels=memory.use_cueq_triangle_kernels,
            use_lma=memory.use_lma,
            chunk_size=memory.chunk_size,
        ),
    )


def target(context, query, pairs=(), label_sites=None):
    """Features, trunk output and forward operator for one system."""
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet

    model, runner, device = context["model"], context["runner"], context["device"]
    runner.inference_query_set = InferenceQuerySet.from_json(Path(query))
    if len(runner.inference_query_set.queries) != 1:
        raise ValueError("OpenFold3 inputs must contain exactly one target")
    _configure_msa(runner, json.loads(Path(query).read_text()))

    # The data module is a cached property keyed to the first query, so drop it before each system.
    for attribute in ("data_module_config", "lightning_data_module"):
        runner.__dict__.pop(attribute, None)

    data = runner.lightning_data_module
    data.prepare_data()
    data.setup("predict")
    batch = _to_device(next(iter(data.predict_dataloader())), device)
    atom_array = batch["atom_array"][0]

    index = clouds.frame_index(atom_array, {site for pair in pairs for site in pair})
    operator, kept = (None, [])
    if label_sites:
        operator, kept = clouds.operator(label_sites, pairs, index, device)

    # Evaluate the trunk once; sampling updates a copy of its pair representation.
    with torch.no_grad():
        s_input, s_trunk, z = model.run_trunk(
            batch,
            num_cycles=context["config"].architecture.shared.num_recycles + 1,
            inplace_safe=True,
        )

    batch.pop("ref_space_uid_to_perm", None)
    return dict(
        batch=tensor_tree_map(lambda t: t.unsqueeze(1), batch),
        atom_array=atom_array,
        index=index,
        operator=operator,
        pairs=kept,
        s_input=s_input.unsqueeze(1).detach(),
        s_trunk=s_trunk.unsqueeze(1).detach(),
        z=z.unsqueeze(1).float().detach(),
    )


def _configure_msa(runner, query):
    """Include explicitly supplied A3M filenames in OpenFold3's alignment filters."""
    msa = runner.dataset_config_kwargs.msa
    limits = dict(msa.max_seq_counts)
    main, paired = set(), set()

    for entry in query["queries"].values():
        for chain in entry["chains"]:
            for key, names in (("main_msa_file_paths", main), ("paired_msa_file_paths", paired)):
                for filename in chain.get(key) or []:
                    path = Path(filename)
                    names.add(path.stem)
                    with path.open() as handle:
                        count = sum(line.startswith(">") for line in handle)
                    limits[path.stem] = max(limits.get(path.stem, 0), count)

    msa.max_seq_counts = limits
    msa.aln_order = list(msa.aln_order) + sorted(main - set(msa.aln_order))
    msa.paired_msa_order = list(msa.paired_msa_order) + sorted(paired - set(msa.paired_msa_order))


def sample(context, target, guidance=None, *, members=25, n_step=200, seed=101):
    """Sample an ensemble using a shared pair representation."""
    from openfold3.core.model.structure.diffusion_module import (
        centre_random_augmentation,
        create_noise_schedule,
    )

    model, device, config = context["model"], context["device"], context["config"]
    batch = target["batch"]
    atom_mask = batch["atom_mask"]
    diffusion = model.sample_diffusion

    schedule = create_noise_schedule(
        no_rollout_steps=n_step,
        **config.architecture.noise_schedule,
        dtype=torch.float32,
        device=device,
    )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    def denoise(z, x_noisy, t):
        return model.diffusion_module(
            batch=batch,
            xl_noisy=x_noisy,
            token_mask=batch["token_mask"],
            atom_mask=atom_mask,
            t=t.to(device),
            si_input=target["s_input"],
            si_trunk=target["s_trunk"],
            zij_trunk=z,
            use_conditioning=True,
            **context["kernels"],
        )

    shape = (atom_mask.shape[0], 1, atom_mask.shape[-1], 3)
    x = [
        schedule[0] * torch.randn(shape, device=device, dtype=torch.float32) for _ in range(members)
    ]

    with torch.set_grad_enabled(guidance is not None):
        for tau, sigma in enumerate(schedule[1:]):
            t = schedule[tau] * ((diffusion.gamma_0 if sigma > diffusion.gamma_min else 0) + 1)
            guiding = guidance is not None and guidance.guided(float(t))
            z = guidance.z if guidance is not None else target["z"]

            noisy, clean = [], []
            for member in range(members):
                augmented = centre_random_augmentation(xl=x[member], atom_mask=atom_mask)
                noise_level = diffusion.noise_scale * torch.sqrt(t**2 - schedule[tau] ** 2)
                x_noisy = augmented + noise_level * torch.randn_like(augmented)

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
                guidance.step(torch.cat([c[0] for c in clean], dim=0), float(t))
                clean = [c.detach() for c in clean]

            for member in range(members):
                delta = (noisy[member] - clean[member]) / t
                x[member] = noisy[member] + diffusion.step_scale * (sigma - t) * delta

    return torch.cat([coordinates[0] for coordinates in x], dim=0).detach()


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


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    return value
