## Environments

Use a separate environment for each predictor, following its upstream installation instructions.
Install this package in each environment with `pip install -e .` from the repository root.
Only the selected backend is imported at runtime.

| file | predictor | python | torch | numpy |
|---|---|---|---|---|
| `protenix.txt` | protenix 2.0.0 | 3.11.15 | 2.7.1 | 2.4.1 |
| `boltz.txt` | boltz 2.2.1 (git `b1ebfc4`) | 3.12.13 | 2.11.0 | 1.26.4 |
| `openfold3.txt` | openfold3 0.4.2.dev83 (git `c9bfe23`) | 3.12.13 | 2.5.1 | 2.3.5 |

These files are version snapshots from the environments used for the reported runs. They include
packages installed through conda-forge and are not pip installation requirements.

The `clouds` command requires chiLife. It is installed in the recorded Protenix environment;
`pip install -e ".[clouds]"` installs the corresponding optional dependency.
