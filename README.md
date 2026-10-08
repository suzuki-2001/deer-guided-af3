## DEER-guided AF3

DEER-guided conformational sampling for
[Protenix](https://github.com/bytedance/Protenix),
[Boltz-2](https://github.com/jwohlwend/boltz) and
[OpenFold3](https://github.com/aqlaboratory/openfold3).

A differentiable spin-label operator predicts distance distributions from each conformation.
During sampling, guidance adjusts the ensemble's shared pair representation to fit the measured
DEER distributions. The pretrained model weights remain fixed.

### Installation

Install the predictor you want to use, including its checkpoints, following its installation
instructions above. The predictors require separate environments; the versions used in our runs
are listed in [envs/](envs/README.md).

Within the predictor's environment:

```bash
git clone https://github.com/suzuki-2001/deer-guided-af3.git
cd deer-guided-af3
pip install -e .
```

Building spin-label clouds also requires chiLife. Install it in a compatible environment with
`pip install -e ".[clouds]"`; the recorded Protenix environment supports this step.

### Usage

Start with an unguided prediction, build the spin-label clouds on that structure, then sample
with DEER guidance. For the included PfMATE example:

```bash
cd example/pfmate

deer-guide predict --model protenix --input pfmate.json --out output/unguided

deer-guide clouds --structure output/unguided/model_0.pdb \
    --pairs deer.csv --out output/clouds.npz

deer-guide guide --model protenix --input pfmate.json \
    --pairs deer.csv --clouds output/clouds.npz --out output/guided
```

The example includes the sequence, MSA and measured distributions. See
[example/pfmate/](example/pfmate/README.md) for the stored reference results.

For other inputs, choose the corresponding backend:

| `--model` | `--input` |
|---|---|
| `protenix` | Protenix JSON for one system |
| `boltz` | Boltz processed directory containing one target and `manifest.json` |
| `openfold3` | OpenFold3 query JSON for one system |

Boltz expects its checkpoint and molecule files in `~/.boltz/`. OpenFold3 uses
`$OPENFOLD_CACHE`, or its default `~/.openfold3/` cache.

Sampling defaults to 25 structures, 200 diffusion steps and seed 101. Override these with
`--members`, `--steps` and `--seed`. Guidance uses `--lr 0.1` and `--anchor 1.0` by default.
Input preparation and trunk evaluation use a fixed seed of 101. `--seed` controls the diffusion
noise, so comparisons between diffusion seeds retain the same trunk output.

Both prediction commands write `model_0.pdb`, `model_1.pdb`, and so on to `--out`.
The guided run also writes `guidance.json` with the settings, retained pairs and fit history.

#### Distance distributions

The restraint CSV has no header. Each row contains two residue sites followed by 99 values of
P(r), at distances 1 through 99 Å. Sites are residue numbers such as `134`, or chain-qualified
numbers such as `A_134`. Use chain-qualified sites when residue numbers occur in multiple chains.
For DEERFold's 100-bin rows, this implementation uses the first 99 bins and normalises them.
A non-zero terminal overflow bin produces a warning because its mass is excluded.

Values must be finite and non-negative, with positive total mass; they are normalised when read.
Self-pairs, duplicate pairs and rows with the wrong number of bins are rejected. Blank lines and
lines beginning with `#` are ignored. [deer.csv](example/pfmate/deer.csv) is a complete example.

### Acknowledgements

- **chiLife** — Tessmer & Stoll 2023, *PLOS Computational Biology*. [doi.org/10.1371/journal.pcbi.1010834](https://doi.org/10.1371/journal.pcbi.1010834). The R1M rotamer library and the spin-label model this operator differentiates. GPL-3.0, which this repository follows.
- **Protenix** — ByteDance AML AI4Science Team 2025, *bioRxiv*. [doi.org/10.1101/2025.01.08.631967](https://doi.org/10.1101/2025.01.08.631967).
- **Boltz-2** — Passaro *et al.* 2025, *bioRxiv*. [doi.org/10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707).
- **OpenFold3** — The OpenFold3 Team 2025, preview 2. [doi.org/10.5281/zenodo.19001000](https://doi.org/10.5281/zenodo.19001000).
- **DEERFold** — Wu *et al.* 2025, *Nature Communications*. [doi.org/10.1038/s41467-025-62582-4](https://doi.org/10.1038/s41467-025-62582-4). The measured distributions of the example, redistributed under CC BY 4.0.
