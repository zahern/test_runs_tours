import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_or_mock(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(
        {
            "segment": ["Workers", "Students", "Retired", "Low-income", "High-income"],
            "delta_safe_zone_share": [0.021, 0.034, 0.012, 0.028, 0.015],
        }
    )


def main() -> None:
    in_path = os.path.join("runs", "safety_analysis", "safety_profile_segments.csv")
    out_dir = os.path.join("runs", "safety_analysis", "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = load_or_mock(in_path)
    df = df.sort_values("delta_safe_zone_share", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["segment"], df["delta_safe_zone_share"], color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta share choosing safer zones")
    ax.set_xlabel("Population segment")
    ax.set_title("Safety Attraction by Segment")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    out_png = os.path.join(out_dir, "safety_attraction_segments.png")
    fig.savefig(out_png, dpi=160)
    print(out_png)


if __name__ == "__main__":
    main()
