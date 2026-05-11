import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    from metacountregressor import ExperimentBuilder, load_example_duration_data

    out_dir = Path("runs") / "example16_3_duration_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_example_duration_data().copy()
    if "ID" not in df.columns:
        df["ID"] = np.arange(len(df))
    if "DURATION" not in df.columns:
        raise ValueError("Expected DURATION in load_example_duration_data output")

    # For MetaCount duration-family search, survival/AFT-style models are searched
    # under model_family='duration' and require a positive budget/reference column.
    df["B"] = np.clip(df.get("LENGTH", 1.0), 1e-6, None)

    builder = ExperimentBuilder(df=df, id_col="ID", y_col="DURATION")

    # Roles include random parameters:
    # 2 = random independent, 3 = random correlated, 4 = grouped random.
    evaluator = builder.build_evaluator(
        variables=[c for c in ["AADT", "LENGTH", "SPEED", "CURVES", "WIDTH", "SLOPE", "SHOULDERW"] if c in df.columns],
        model_family="duration",
        budget_col="B",
        default_roles=[0, 1, 2, 3, 4, 5, 6],
        max_latent_classes=1,
        R=200,
    )

    result = builder.run(
        evaluator,
        algo="sa",
        max_iter=500,
        n_jobs=1,
        seed=42,
    )

    payload = {
        "model_family": "duration",
        "survival_mapping": "Survival/AFT search is run through model_family='duration'",
        "random_parameters_enabled": True,
        "roles_exposed": [0, 1, 2, 3, 4, 5, 6],
        "best_score": float(getattr(result, "best_score", np.nan)),
        "best_spec": str(getattr(result, "best_spec", None)),
        "n_rows": int(len(df)),
    }

    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    desc = {
        "ID": "Segment identifier",
        "DURATION": "Positive duration target used for duration/survival modeling",
        "AADT": "Annual Average Daily Traffic",
        "LENGTH": "Segment length",
        "SPEED": "Speed variable from Example 16-3",
        "CURVES": "Curvature metric",
        "WIDTH": "Roadway width",
        "SLOPE": "Grade/slope",
        "SHOULDERW": "Shoulder width",
        "B": "Positive budget/reference column required by duration-family search",
    }

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    describe_html = df[numeric_cols].describe().to_html(border=0)
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in desc.items() if k in df.columns)

    html = (
        "<html><head><meta charset='utf-8'><title>Example16-3 Duration Search</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:6px;}th{background:#f4f4f4;}</style>"
        "</head><body>"
        "<h1>Washington Example 16-3: Duration/Survival Search Demo</h1>"
        "<p>MetaCount survival models are searched under <b>model_family='duration'</b>.</p>"
        "<p>Random parameters are enabled by allowing roles 2/3/4 in default_roles.</p>"
        f"<p><b>Best score:</b> {payload['best_score']}</p>"
        "<h2>Data properties</h2>"
        f"{describe_html}"
        "<h2>Variable descriptions</h2>"
        "<table><tr><th>Variable</th><th>Description</th></tr>"
        f"{rows}</table>"
        "</body></html>"
    )

    with (out_dir / "summary.html").open("w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved: {out_dir / 'result.json'}")
    print(f"Saved: {out_dir / 'summary.html'}")


if __name__ == "__main__":
    # Keep JAX deterministic and CPU-friendly by default for reproducible demos.
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
