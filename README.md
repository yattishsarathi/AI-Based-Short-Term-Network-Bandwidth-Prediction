
# AI-Based-Short-Term-Network-Bandwidth-Prediction
An AI/ML-based project that analyzes historical network measurements such as bandwidth utilization, throughput, latency, and packet loss to predict short-term network bandwidth using Machine Learning and visualize results through an interactive Streamlit dashboard.
from pathlib import Path
import numpy as np
import pandas as pd


def generate_dataset(
    output_path="data/network_data.csv",
    rows=3000,
    seed=42
):
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="5min"
    )

    t = np.arange(rows)

    hour = (
        timestamps.hour.to_numpy()
        + timestamps.minute.to_numpy() / 60
    )

    daily_pattern = (
        0.5
        + 0.35 * np.sin(
            2 * np.pi * (hour - 8) / 24
        )
    )

    weekly_pattern = np.where(
        timestamps.dayofweek.to_numpy() < 5,
        1.0,
        0.78
    )

    utilization = np.clip(
        45
        + 28 * daily_pattern * weekly_pattern
        + 8 * np.sin(2 * np.pi * t / 90)
        + rng.normal(0, 6, rows),
        5,
        98
    )

    latency = np.clip(
        12
        + utilization * 0.28
        + rng.normal(0, 3, rows),
        4,
        120
    )

    packet_loss = np.clip(
        0.05
        + np.maximum(utilization - 75, 0) * 0.025
        + rng.normal(0, 0.08, rows),
        0,
        5
    )

    throughput = np.clip(
        utilization
        - 0.18 * latency
        - 2.5 * packet_loss
        + 4 * np.sin(2 * np.pi * t / 48)
        + rng.normal(0, 2.5, rows),
        1,
        100
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "bandwidth_utilization_pct": utilization,
        "throughput_mbps": throughput,
        "latency_ms": latency,
        "packet_loss_pct": packet_loss
    })

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_path, index=False)

    print("Dataset generated successfully!")
    print(df.head())


if __name__ == "__main__":
    generate_dataset()
