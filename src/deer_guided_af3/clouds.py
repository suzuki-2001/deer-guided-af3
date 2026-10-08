"""Spin-label clouds built with chiLife on an unguided prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .operator import LabelSite, RotamerPrOperator, backbone_frame_np
from .restraints import R_GRID, parse_site

LABEL = "R1M"


def build(structure, sites, label=LABEL):
    """{site: (spin centres in the local frame, weights)}."""
    import chilife

    protein = chilife.load_protein(str(structure))
    out = {}

    for site in sites:
        chain, resid = parse_site(site)
        select = f"resid {resid}" + (f" and segid {chain}" if chain else "")
        residue = protein.select_atoms(select)
        if not len(residue):
            raise ValueError(f"site {site} is absent from {structure}")

        third = "C" if residue.resnames[0] == "GLY" else "CB"
        xyz = []
        for name in ("N", "CA", third):
            atom = protein.select_atoms(f"{select} and name {name}")
            if len(atom) != 1:
                raise ValueError(f"site {site}: {len(atom)} atoms named {name}")
            xyz.append(atom.positions[0].astype(float))

        rotamers = chilife.SpinLabel(
            label, site=resid, chain=chain, protein=protein, eval_clash=True
        )
        centres = np.asarray(rotamers.spin_centers, float)
        weights = np.asarray(rotamers.weights, float)
        if (
            not len(weights)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not weights.sum() > 0
            or not np.isfinite(centres).all()
        ):
            raise ValueError(f"site {site}: chiLife returned no usable rotamer")

        # Local coordinates let the cloud move with its residue during sampling.
        rotation, origin = backbone_frame_np(*xyz)
        out[site] = ((centres - origin) @ rotation, weights / weights.sum())

    return out


def save(path, clouds):
    """Store local spin centres and rotamer weights in a compressed NPZ file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"sloc_{s}": c for s, (c, _) in clouds.items()},
        **{f"w_{s}": w for s, (_, w) in clouds.items()},
    )


def load(path, sites):
    """Load available label sites and report those missing centres or weights."""
    label_sites, missing = {}, []

    with np.load(path, allow_pickle=False) as stored:
        for site in sites:
            centres_key, weights_key = f"sloc_{site}", f"w_{site}"
            if centres_key not in stored or weights_key not in stored:
                missing.append(site)
                continue
            label_sites[site] = LabelSite(stored[centres_key], stored[weights_key], site)

    return label_sites, missing


def frame_index(atom_array, sites):
    """site -> atom indices of N, CA, CB; a site without all three is left out."""
    res_id = np.asarray(atom_array.res_id)
    atom_name = np.asarray(atom_array.atom_name)
    chain_id = np.asarray(atom_array.chain_id)
    index = {}

    for site in sites:
        chain, resid = parse_site(site)
        found = {}
        for name in ("N", "CA", "CB"):
            mask = (res_id == resid) & (atom_name == name)
            if chain is not None:
                mask = mask & (chain_id == chain)
            where = np.where(mask)[0]
            if len(where) > 1:
                raise ValueError(f"site {site} has multiple atoms named {name}; specify its chain")
            if len(where):
                found[name] = int(where[0])

        if len(found) == 3:
            index[site] = found
    return index


def operator(label_sites, pairs, index, device="cuda"):
    """The forward operator over the pairs both the structure and the clouds support."""
    keep = [p for p in pairs if all(s in index and s in label_sites for s in p)]
    if not keep:
        raise ValueError("no pair has clouds and a full backbone at both sites")

    used = sorted({s for p in keep for s in p}, key=str)
    return RotamerPrOperator({s: label_sites[s] for s in used}, keep, R_GRID, device=device), keep
