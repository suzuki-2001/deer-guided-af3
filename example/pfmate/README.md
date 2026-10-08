## PfMATE

A multidrug transporter with an outward-facing ([6GWH](https://www.rcsb.org/structure/6GWH)) and an
inward-facing ([6FHZ](https://www.rcsb.org/structure/6FHZ)) crystal structure. `deer.csv` holds 25
measured distributions over 19 labelled sites.

Run from `example/pfmate` in the Protenix environment, with chiLife installed:

```bash
deer-guide predict --model protenix --input pfmate.json --out output/unguided
deer-guide clouds --structure output/unguided/model_0.pdb \
    --pairs deer.csv --out output/clouds.npz
deer-guide guide --model protenix --input pfmate.json --pairs deer.csv \
    --clouds output/clouds.npz --out output/guided
```

### Reference results

`results/` contains the stored Protenix reference run: 25 guided structures, 200 diffusion steps,
seed 101, learning rate 0.1 and anchor weight 1.0.

| File | Contents |
|---|---|
| [results/unguided/model_0.pdb](results/unguided/model_0.pdb) | Unguided structure used to build the clouds |
| [results/clouds.npz](results/clouds.npz) | Local spin-centre coordinates and rotamer weights for 19 sites |
| [results/guided/](results/guided/) | 25 guided structures, `model_0.pdb` through `model_24.pdb` |
| [results/guided/guidance.json](results/guided/guidance.json) | Sampling settings, 25 retained pairs and fit history |

The commands above write to `output/`, leaving these reference files intact.

In that reference run, mean Cα RMSD over the 25-member ensembles changes from 5.26 to 1.45 Å against
the inward-facing 6FHZ, and from 0.94 to 4.09 Å against the outward-facing 6GWH. The logged
distribution error decreases from 8.14 to 2.22 Å over 171 updates.
Only the unguided structure used for cloud construction is included here; all 25 guided structures
are included.

`pfmate.json` points at `pfmate_msa.a3m`, the MSA shipped here (17,417 sequences, ColabFold).
The commands use this precomputed MSA; running from this directory resolves its relative path.

### Data

`deer.csv` is the Experiment 1 set of the DEERFold benchmark
([10.5281/zenodo.15147304](https://doi.org/10.5281/zenodo.15147304), CC BY 4.0), published with
Wu *et al.* 2025, *Nature Communications*
[10.1038/s41467-025-62582-4](https://doi.org/10.1038/s41467-025-62582-4). The measurements are from
Del Alamo *et al.* 2021, *PLOS Computational Biology*
[10.1371/journal.pcbi.1009107](https://doi.org/10.1371/journal.pcbi.1009107).
