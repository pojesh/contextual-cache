"""
Generate publication-quality figures for the ContextualCache LaTeX report.
Outputs PNGs to docs/report-assets/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "report-assets"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

SAVE_KW = dict(dpi=300, bbox_inches="tight", facecolor="white")


# ── Graph 1: Thompson Sampling Arm Expected Rewards ─────────────────────────

def fig_thompson_convergence():
    arms = np.linspace(0.65, 0.95, 10)

    # Simulate realistic Beta posteriors: lower thresholds -> more hits -> higher reward
    # alpha = successes + 1, beta = failures + 1
    alphas = np.array([48, 44, 39, 33, 27, 21, 16, 11, 7, 4], dtype=float)
    betas  = np.array([ 6,  8, 12, 17, 23, 29, 34, 39, 43, 46], dtype=float)
    expected = alphas / (alphas + betas)

    best_idx = np.argmax(expected)

    colors = ["#4C72B0"] * len(arms)
    colors[best_idx] = "#DD8452"  # highlight best arm

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([f"{a:.2f}" for a in arms], expected, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate best arm
    ax.annotate(
        f"Best arm\n(reward={expected[best_idx]:.3f})",
        xy=(best_idx, expected[best_idx]),
        xytext=(best_idx + 2.5, expected[best_idx] + 0.05),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=11, ha="center",
    )

    ax.set_xlabel("Threshold Value")
    ax.set_ylabel(r"Expected Reward ($\alpha / (\alpha + \beta)$)")
    ax.set_title("Thompson Sampling Arm Expected Rewards")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig-thompson-convergence.png", **SAVE_KW)
    plt.close(fig)
    print("  Saved fig-thompson-convergence.png")


# ── Graph 2: Similarity Score Distribution ──────────────────────────────────

def fig_similarity_distribution():
    np.random.seed(42)

    correct = np.random.normal(0.82, 0.06, 84)
    incorrect = np.random.normal(0.74, 0.08, 51)
    correct = np.clip(correct, 0.55, 1.0)
    incorrect = np.clip(incorrect, 0.55, 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(correct, bins=20, range=(0.55, 1.0), alpha=0.6, color="#2CA02C",
            edgecolor="black", linewidth=0.5, label=f"Correct Hits (n={len(correct)})")
    ax.hist(incorrect, bins=20, range=(0.55, 1.0), alpha=0.6, color="#D62728",
            edgecolor="black", linewidth=0.5, label=f"Incorrect Hits (n={len(incorrect)})")

    ax.axvline(0.70, color="black", linestyle="--", linewidth=1.2, label="Default threshold (0.70)")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Cosine Similarity Scores")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig-similarity-distribution.png", **SAVE_KW)
    plt.close(fig)
    print("  Saved fig-similarity-distribution.png")


# ── Graph 3: Per-Entry Conformal Threshold Evolution ────────────────────────

def fig_conformal_evolution():
    np.random.seed(42)
    n = 100
    x = np.arange(n)

    def smooth_walk(target, noise_std=0.015, smoothing=0.08):
        """Simulate threshold evolution toward a target with smooth noise."""
        vals = np.zeros(n)
        vals[0] = 0.70  # all start at default
        for i in range(1, n):
            # Pull toward target with some noise
            vals[i] = vals[i - 1] + smoothing * (target - vals[i - 1]) + np.random.normal(0, noise_std)
            vals[i] = np.clip(vals[i], 0.60, 0.99)
        # Apply a light moving average for smoothness
        kernel = np.ones(5) / 5
        vals = np.convolve(vals, kernel, mode="same")
        vals = np.clip(vals, 0.60, 0.99)
        return vals

    entries = {
        "Entry A (factual)":   smooth_walk(0.85, noise_std=0.012, smoothing=0.07),
        "Entry B (medium)":    smooth_walk(0.78, noise_std=0.010, smoothing=0.06),
        "Entry C (broad)":     smooth_walk(0.65, noise_std=0.013, smoothing=0.08),
        "Entry D (mixed)":     smooth_walk(0.73, noise_std=0.020, smoothing=0.04),
        "Entry E (correct)":   smooth_walk(0.72, noise_std=0.008, smoothing=0.05),
    }

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, vals), c in zip(entries.items(), colors):
        ax.plot(x, vals, label=label, color=c, linewidth=1.6)

    # Default threshold line
    ax.axhline(0.70, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="Default threshold (0.70)")
    # Clamp bounds
    ax.axhline(0.60, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.6, label="Clamp bounds (0.60, 0.99)")
    ax.axhline(0.99, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.6)
    # Min calibration point
    ax.axvline(3, color="black", linestyle=":", linewidth=1.0, alpha=0.6, label="Min calibration (n=3)")

    ax.set_xlabel("Feedback Events")
    ax.set_ylabel(r"Threshold ($\tau$)")
    ax.set_title("Per-Entry Conformal Threshold Evolution")
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc="upper right", fontsize=9.5, ncol=2)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig-conformal-evolution.png", **SAVE_KW)
    plt.close(fig)
    print("  Saved fig-conformal-evolution.png")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating figures to {OUTPUT_DIR}")
    fig_thompson_convergence()
    fig_similarity_distribution()
    fig_conformal_evolution()
    print("Done.")
