# HPC conda environments

Two independent environments run this project on the QUT HPC (`aqua`). They are
deliberately separate because their dependency stacks conflict (ActivitySim pins
numpy/numba/pandas that fight the jax+larch estimation stack).

| Env | Spec | Purpose |
|-----|------|---------|
| `zigenv` | [`environment_qld.yml`](environment_qld.yml) | **QLD pipeline (normal runs).** larch + jax + metacountregressor/SearchLibrium estimation, stages 0–6, GA search, prediction. This is the `CONDA_ENV` default in the PBS jobs. |
| `asimbench` | [`environment_activitysim.yml`](environment_activitysim.yml) | **ActivitySim benchmark.** Standalone `activitysim` to run the example models and compare results. Isolated on ActivitySim's own tested stack. |

(`geotfenv` is a separate, pre-existing env for geo/transit *preprocessing* —
GTFS skims, Overture landuse, WorldPop — and is unaffected by this rebuild.)

## Why `zigenv` broke (Sep 2026)

Repeated `pip install`-on-conda upgrades drifted the env out of its
compatibility window and left duplicate/partial installs:

- **`sharrow 2.15.0` was a corrupted partial install** (missing `sharrow.dataset`,
  `sharrow.flows`, …) → `import larch` failed → the whole pipeline was dead.
- **numpy 2.5.1** (conda) sat above numba's `<2.4` ceiling; a duplicate pip
  numpy 2.3.4 was also present.
- **scipy 1.18.0** sat above larch's `<1.16` ceiling.
- Duplicate `metacountregressor` (1.0.150 **and** 1.0.153).

The compatibility window is the intersection of every hard pin in the stack:

```
numba 0.62          -> numpy <2.4
jax  (>=0.4.30)     -> numpy >=2.1, scipy >=1.15   (jax 0.11 also needs py>=3.12)
larch 6.0.46        -> scipy <1.16, numba >=0.60, sharrow >=2.15
metacountregressor  -> numpy >=1.26.4, scipy >=1.11, pandas >=2.1, jax >=0.4.30
searchlibrium       -> numpy >=2.0, pandas >=2.0, jax >=0.4.30
=> numpy in [2.1, 2.4), scipy in [1.15, 1.16), python 3.12
```

## Preventing re-drift

`setup_env.sh` runs `pip install --upgrade -r requirements.txt` on every job.
[`requirements.txt`](requirements.txt) now carries the numpy/scipy/numba
CEILINGS so that upgrade can never re-pull numpy 2.5 / scipy 1.18. Any ad-hoc
`pip install --upgrade` of the estimators should pass the constraints file:

```bash
pip install --upgrade -c constraints.txt metacountregressor SearchLibrium
```

## (Re)create the environments

Both specs live in the repo; `Z:` is the HPC home, so they are already on the
cluster. On a login node:

```bash
module load Anaconda3/2024.02-1
conda env create -f environment_qld.yml          # -> zigenv
conda env create -f environment_activitysim.yml  # -> asimbench
```

Verify an estimation env with the bundled checker:

```bash
conda activate zigenv && python _verify_env.py   # expect RESULT: PASS
```

## Verified builds (Sep 2026)

- **zigenv**: python 3.12, numpy 2.3.5, scipy 1.15.2, numba 0.62.1,
  jax/jaxlib 0.11.2, larch 6.0.46, sharrow 2.16.2, pandas 2.3.3,
  metacountregressor 1.0.153, SearchLibrium 0.0.239 — `_verify_env.py` PASS.
- **asimbench**: python 3.11, activitysim 1.5.1, numpy 2.3.4, numba 0.67.0,
  pandas 2.3.3.
