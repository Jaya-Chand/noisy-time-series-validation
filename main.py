"""
Noisy Time-Series Validation

A small Python project for validating noisy, stochastic time-series measurements
from two independent instruments.

Workflow:
1. Load Polaris instrument data.
2. Fetch R2D2 reference data.
3. Resample both streams into 60-second bins.
4. Match nearest timestamps.
5. Filter low-signal points.
6. Calculate correlation, bias, and RMSE.
7. Save results and create a validation plot.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy.stats import pearsonr


# ============================================================
# CONFIGURATION
# ============================================================

# This is the Google Colab / Google Drive path where your Polaris files live.
POLARIS_DIR = "/content/drive/MyDrive/Polaris_r0_data/"

# Start with one date to keep the project simple.
DATE = "2025_11_24"

# Only use Polaris points with decent background-subtracted signal.
MIN_COUNTS = 800

# Resample both time series into 60-second bins.
BIN_WIDTH = "60s"

# Maximum allowed difference between matched Polaris and R2D2 timestamps.
MATCH_TOLERANCE = "5min"


# ============================================================
# DATA CONVERSION
# ============================================================

def seeing_to_r0(seeing_arcsec):
    """
    Convert seeing in arcseconds to r0 in metres.

    The R2D2 monitor reports seeing. This function converts it into the same
    type of turbulence metric used in the Polaris logs.
    """
    wavelength = 500e-9
    seeing_rad = np.deg2rad(float(seeing_arcsec) / 3600.0)

    if seeing_rad == 0 or not np.isfinite(seeing_rad):
        return np.nan

    return wavelength / seeing_rad


# ============================================================
# POLARIS LOADING
# ============================================================

def load_polaris_log(date_str):
    """
    Load one Polaris nightly log and compute background-subtracted net counts.

    Expected file:
        /content/drive/MyDrive/Polaris_r0_data/r0_values_log_YYYY_MM_DD.txt

    Expected columns:
        timestamp_UT
        r0_L
        r0_T
        mean_counts_L
        mean_counts_R
        bg_flux_mean
    """
    path = os.path.join(POLARIS_DIR, f"r0_values_log_{date_str}.txt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find Polaris file: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_cols = [
        "timestamp_UT",
        "r0_L",
        "r0_T",
        "mean_counts_L",
        "mean_counts_R",
        "bg_flux_mean",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Polaris file: {missing}")

    df["datetime"] = pd.to_datetime(df["timestamp_UT"], errors="coerce")

    numeric_cols = [
        "r0_L",
        "r0_T",
        "mean_counts_L",
        "mean_counts_R",
        "bg_flux_mean",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Background-subtracted signal quality metric.
    df["net_counts"] = (
        (df["mean_counts_L"] - df["bg_flux_mean"])
        + (df["mean_counts_R"] - df["bg_flux_mean"])
    ) / 2

    return df[["datetime", "r0_L", "r0_T", "net_counts"]].dropna()


# ============================================================
# R2D2 FETCHING
# ============================================================

def fetch_r2d2(date_str):
    """
    Fetch R2D2 reference data for one date from the ING public seeing page.

    This is kept intentionally simple for the starter GitHub project.
    """
    d0 = datetime.strptime(date_str, "%Y_%m_%d")
    d1 = d0 + timedelta(days=1)

    start = d0.strftime("%Y-%m-%d")
    end = d1.strftime("%Y-%m-%d")

    response = requests.get(
        "https://astro.ing.iac.es/seeing/r2d2dimm.php",
        params={"date1": start, "date2": end, "submit": "Submit"},
        timeout=15,
    )

    response.raise_for_status()

    if "<pre>" not in response.text or "</pre>" not in response.text:
        return pd.DataFrame()

    block = (
        response.text
        .split("<pre>")[1]
        .split("</pre>")[0]
        .strip()
        .splitlines()
    )

    if len(block) < 3:
        return pd.DataFrame()

    columns = [c.strip() for c in block[0].split("|")]
    rows = []

    for line in block[2:]:
        parts = line.split()

        # The R2D2 page has a fixed-width structure. Your previous parser
        # expected 21 values per row, so malformed rows are skipped.
        if len(parts) != 21:
            continue

        date, time, star, *numbers = parts
        rows.append([date + " " + time, star] + numbers)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=columns)

    df["datetime"] = (
        pd.to_datetime(df["measure_date"], utc=True, errors="coerce")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    for col in columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["r0_L"] = df["seeing_long"].map(seeing_to_r0)
    df["r0_T"] = df["seeing_trans"].map(seeing_to_r0)

    return df[["datetime", "r0_L", "r0_T"]].dropna()


# ============================================================
# PREPROCESSING
# ============================================================

def resample_to_60s(df):
    """
    Resample a time series into 60-second mean bins.
    """
    return (
        df.set_index("datetime")
        .resample(BIN_WIDTH)
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )


def match_streams(polaris, r2d2):
    """
    Match Polaris and R2D2 measurements by nearest timestamp.
    """
    polaris = polaris.sort_values("datetime")
    r2d2 = r2d2.sort_values("datetime")

    matched = pd.merge_asof(
        polaris.rename(
            columns={
                "r0_L": "r0_L_polaris",
                "r0_T": "r0_T_polaris",
            }
        ),
        r2d2.rename(
            columns={
                "r0_L": "r0_L_r2d2",
                "r0_T": "r0_T_r2d2",
            }
        ),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(MATCH_TOLERANCE),
    )

    matched = matched.dropna(
        subset=[
            "r0_L_polaris",
            "r0_L_r2d2",
            "r0_T_polaris",
            "r0_T_r2d2",
            "net_counts",
        ]
    )

    # Filter low-signal Polaris points.
    matched = matched[matched["net_counts"] >= MIN_COUNTS]

    return matched


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(matched, axis):
    """
    Calculate correlation, bias, and RMSE for one axis.

    axis should be:
        "L" for longitudinal
        "T" for transverse
    """
    pol_col = f"r0_{axis}_polaris"
    r2_col = f"r0_{axis}_r2d2"

    residual = matched[pol_col] - matched[r2_col]

    correlation = pearsonr(matched[pol_col], matched[r2_col])[0]
    bias = residual.mean()
    rmse = np.sqrt(np.mean(residual ** 2))

    return {
        "axis": axis,
        "N": len(matched),
        "correlation": correlation,
        "bias": bias,
        "rmse": rmse,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_correlation(matched):
    """
    Plot Polaris vs R2D2 for longitudinal and transverse measurements.
    """
    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_info = [
        ("L", "Longitudinal"),
        ("T", "Transverse"),
    ]

    for ax, (axis, title) in zip(axes, plot_info):
        pol_col = f"r0_{axis}_polaris"
        r2_col = f"r0_{axis}_r2d2"

        ax.scatter(
            matched[r2_col],
            matched[pol_col],
            c=matched["net_counts"],
            s=20,
            alpha=0.8,
        )

        min_val = min(matched[r2_col].min(), matched[pol_col].min())
        max_val = max(matched[r2_col].max(), matched[pol_col].max())

        ax.plot([min_val, max_val], [min_val, max_val], "k--", label="1:1 line")

        ax.set_xlabel("R2D2")
        ax.set_ylabel("Polaris")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.suptitle("Polaris vs R2D2 time-matched measurements")
    plt.tight_layout()
    plt.savefig("figures/example_correlation.png", dpi=200)
    plt.show()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Run the full starter validation pipeline.
    """
    print("Loading Polaris data...")
    polaris = load_polaris_log(DATE)

    # Your previous comparison code applied a +1 hour offset to Polaris.
    # Keep this here because it matches your existing analysis setup.
    polaris["datetime"] = polaris["datetime"] + pd.Timedelta(hours=1)

    print("Fetching R2D2 data...")
    r2d2 = fetch_r2d2(DATE)

    if r2d2.empty:
        print("No R2D2 data found.")
        return

    print("Resampling data into 60-second bins...")
    polaris_60s = resample_to_60s(polaris)
    r2d2_60s = resample_to_60s(r2d2)

    print("Matching data streams...")
    matched = match_streams(polaris_60s, r2d2_60s)

    print(f"Matched points after filtering: {len(matched)}")

    if len(matched) < 2:
        print("Not enough matched data points to calculate metrics.")
        return

    metrics = pd.DataFrame(
        [
            calculate_metrics(matched, "L"),
            calculate_metrics(matched, "T"),
        ]
    )

    print("\nValidation metrics:")
    print(metrics)

    os.makedirs("results", exist_ok=True)

    matched.to_csv("results/matched_data.csv", index=False)
    metrics.to_csv("results/metrics.csv", index=False)

    print("\nSaved:")
    print("results/matched_data.csv")
    print("results/metrics.csv")

    plot_correlation(matched)

    print("\nSaved:")
    print("figures/example_correlation.png")


if __name__ == "__main__":
    main()
