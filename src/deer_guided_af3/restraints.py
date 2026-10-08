"""Measured distance distributions: one CSV row per pair, two site tokens then P(r) per bin."""

from __future__ import annotations

import csv
import warnings

import numpy as np

R_GRID = np.arange(1.0, 100.0, 1.0)  # angstrom


def parse_site(token):
    """Parse 134 or A_134 into (chain or None, residue number)."""
    if isinstance(token, (int, np.integer)):
        return None, int(token)

    text = str(token)
    if "_" not in text:
        return None, int(text)

    chain, resid = text.rsplit("_", 1)
    if not chain or any(c.isspace() for c in chain):
        raise ValueError(f"invalid chain in site {token!r}")
    return chain, int(resid)


def _token(text):
    text = text.strip()
    parse_site(text)
    return text if "_" in text else int(text)


def read(path):
    """Read and normalise each pair's distribution on R_GRID."""
    measured = {}
    seen = set()

    with open(path, newline="") as handle:
        for row_number, fields in enumerate(csv.reader(handle), 1):
            if not fields or all(not value.strip() for value in fields):
                continue
            if fields[0].lstrip().startswith("#"):
                continue

            prefix = f"{path}:{row_number}"
            if len(fields) not in (len(R_GRID) + 2, len(R_GRID) + 3):
                raise ValueError(f"{prefix}: expected two sites and {len(R_GRID)} distance bins")

            try:
                i, j = _token(fields[0]), _token(fields[1])
                probabilities = np.asarray(fields[2:], dtype=float)
            except ValueError as error:
                raise ValueError(f"{prefix}: invalid site or distribution value") from error

            # Reversing the two sites still describes the same measured pair.
            pair = frozenset((i, j))
            if i == j or pair in seen:
                raise ValueError(f"{prefix}: self-pair or duplicate pair {i}-{j}")
            if not np.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError(f"{prefix}: distribution values must be finite and non-negative")

            # DEERFold's extra bin holds overflow outside the finite 1–99 Å grid.
            if len(probabilities) == len(R_GRID) + 1:
                if probabilities[-1] != 0:
                    warnings.warn(
                        f"{path}: using the 1–99 Å grid; the terminal overflow bin is excluded",
                        stacklevel=2,
                    )
                probabilities = probabilities[:-1]

            total = probabilities.sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f"{prefix}: distribution must have positive finite mass")
            seen.add(pair)
            measured[(i, j)] = probabilities / total

    if not measured:
        raise ValueError(f"{path}: no distance distributions")
    return measured
