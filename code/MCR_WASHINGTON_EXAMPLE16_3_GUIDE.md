# MetaCount Example 16-3 Guide

This guide shows where survival models belong in MetaCount families and how to run a random-parameter search using Washington Example 16-3 data.

## Family Mapping

- count: crash frequency outcomes (Poisson/NB style)
- cmf: crash modification function structures
- linear: continuous Gaussian outcomes
- duration: positive-duration outcomes, including survival/AFT-style models

In short, survival models map to `model_family="duration"`.

## Random Parameters in Search

Enable random parameters by allowing roles:

- 2: random independent
- 3: random correlated
- 4: grouped random

Example evaluator setup:

```python
from metacountregressor import ExperimentBuilder, load_example_duration_data

df = load_example_duration_data().copy()
df["B"] = df["LENGTH"].clip(lower=1e-6)

builder = ExperimentBuilder(df=df, id_col="ID", y_col="DURATION")
evaluator = builder.build_evaluator(
    variables=["AADT", "LENGTH", "SPEED", "CURVES", "WIDTH", "SLOPE", "SHOULDERW"],
    model_family="duration",
    budget_col="B",
    default_roles=[0, 1, 2, 3, 4, 5, 6],
    max_latent_classes=1,
    R=200,
)
result = builder.run(evaluator, algo="sa", max_iter=500, n_jobs=1, seed=42)
```

## Washington Data Analysis Example

Run:

```bash
python mcr_duration_survival_example.py
```

Outputs:

- `runs/example16_3_duration_demo/result.json`
- `runs/example16_3_duration_demo/summary.html`

The HTML report includes:

- descriptive statistics for numeric columns in Example 16-3 data
- variable descriptions used in the model
- the configured family/algo and best score

## Variable Notes (Example 16-3)

- `FREQ`: crash count response
- `AADT`: traffic exposure proxy
- `LENGTH`: segment length
- `SPEED`: speed measure
- `CURVES`: curvature measure
- `WIDTH`: roadway width
- `SLOPE`: grade/slope
- `SHOULDERW`: shoulder width
- `OFFSET`: log exposure term for count-family models
- `DURATION`: positive target used in duration/survival demo
- `B`: positive budget/reference column required by duration-family search
