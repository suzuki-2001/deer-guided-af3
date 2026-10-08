"""deer-guide: unguided prediction, rotamer clouds, and DEER-guided sampling."""

from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from . import clouds, restraints
from .guidance import Guidance

BACKENDS = ("protenix", "boltz", "openfold3")
TRUNK_SEED = 101


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def backend(name):
    """Import only the predictor selected by the command."""
    return importlib.import_module(f".{name}", __package__)


def sites_of(pairs):
    return sorted({site for pair in pairs for site in pair}, key=str)


def store(module, target, coords, out):
    if not torch.isfinite(coords).all():
        raise FloatingPointError("the prediction contains non-finite coordinates")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"model_{m}.pdb" for m in range(len(coords))]
    written = module.write(target, coords, paths)
    print(f"{len(written)} of {len(paths)} structures written to {out}")
    return written


def predict(args):
    seed_all(TRUNK_SEED)
    module = backend(args.model)
    context = module.load(n_step=args.steps, device=args.device)

    # Model initialisation consumes RNG state; reset it before input preparation.
    seed_all(TRUNK_SEED)
    target = module.target(context, args.input)

    coords = module.sample(
        context, target, None, members=args.members, n_step=args.steps, seed=args.seed
    )
    store(module, target, coords, args.out)


def build(args):
    pairs = restraints.read(args.pairs)
    sites = sites_of(pairs)
    clouds.save(args.out, clouds.build(args.structure, sites))
    print(f"{len(sites)} sites from {args.structure} -> {args.out}")


def guide(args):
    measured = restraints.read(args.pairs)
    label_sites, missing = clouds.load(args.clouds, sites_of(measured))
    if missing:
        print(f"no cloud for {len(missing)} site(s): {missing}")
    if not label_sites:
        raise SystemExit(f"{args.clouds} holds no cloud for any site of {args.pairs}")

    seed_all(TRUNK_SEED)
    module = backend(args.model)
    context = module.load(n_step=args.steps, device=args.device)

    # Model initialisation consumes RNG state; reset it before input preparation.
    seed_all(TRUNK_SEED)
    target = module.target(context, args.input, list(measured), label_sites)

    grid = target["operator"].r
    curves = {
        pair: torch.tensor(measured[pair], dtype=grid.dtype, device=grid.device)
        for pair in target["pairs"]
    }
    guidance = Guidance(
        target["z"], target["operator"], curves, target["index"], lr=args.lr, anchor=args.anchor
    )
    print(f"guiding on {len(curves)} pair(s), {args.members} structures, {args.steps} steps")

    coords = module.sample(
        context, target, guidance, members=args.members, n_step=args.steps, seed=args.seed
    )
    store(module, target, coords, args.out)

    if guidance.log:
        first, last = guidance.log[0], guidance.log[-1]
        print(f"fit {first['fit']:.2f} -> {last['fit']:.2f} A over {len(guidance.log)} updates")
    else:
        print("no guidance updates at the requested noise levels")

    metadata = dict(
        model=args.model,
        members=args.members,
        steps=args.steps,
        seed=args.seed,
        trunk_seed=TRUNK_SEED,
        lr=args.lr,
        anchor=args.anchor,
        pairs=[f"{i}-{j}" for i, j in target["pairs"]],
        updates=guidance.log,
    )
    (Path(args.out) / "guidance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="deer-guide", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_sampling_options(command):
        command.add_argument("--model", required=True, choices=BACKENDS)
        command.add_argument(
            "--input",
            required=True,
            help="Protenix input JSON, Boltz processed directory, or OpenFold3 query",
        )
        command.add_argument("--out", required=True)
        command.add_argument("--members", type=int, default=25)
        command.add_argument("--steps", type=int, default=200)
        command.add_argument("--seed", type=int, default=101)
        command.add_argument("--device", default="cuda")

    predict_parser = commands.add_parser("predict", help="sample without restraints")
    add_sampling_options(predict_parser)
    predict_parser.set_defaults(run=predict)

    clouds_parser = commands.add_parser(
        "clouds", help="build the rotamer clouds on an unguided prediction"
    )
    clouds_parser.add_argument("--structure", required=True)
    clouds_parser.add_argument("--pairs", required=True)
    clouds_parser.add_argument("--out", required=True)
    clouds_parser.set_defaults(run=build)

    guide_parser = commands.add_parser("guide", help="sample under the measured distributions")
    add_sampling_options(guide_parser)
    guide_parser.add_argument("--pairs", required=True)
    guide_parser.add_argument("--clouds", required=True)
    guide_parser.add_argument("--lr", type=float, default=0.1)
    guide_parser.add_argument("--anchor", type=float, default=1.0)
    guide_parser.set_defaults(run=guide)

    args = parser.parse_args(argv)
    if args.command != "clouds":
        if args.members <= 0 or args.steps <= 0:
            parser.error("--members and --steps must be positive")
        if not 0 <= args.seed < 2**32:
            parser.error("--seed must be between 0 and 2**32 - 1")
    args.run(args)


if __name__ == "__main__":
    main()
