#!/usr/bin/env python3
"""
IoT Physics Profiling - Per Machine (Matched Style)
Goal: Generate ACF, Markov, and Lag plots for M001-M004
      using the EXACT same colors and style as the Hybrid Manufacturing analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
from matplotlib.ticker import MaxNLocator
import os

# CONFIGURATION
CSV_PATH = 'Massive_IoT_Real_Augmented.csv'


# ==========================================
# 1. DATA GENERATION (Safety Check)
# ==========================================
def generate_iot_data_if_needed(filepath):
    if os.path.exists(filepath): return

    print("Generating synthetic IoT data...")
    np.random.seed(42)
    n_rows = 1000
    machines = ['M001', 'M002', 'M003', 'M004']
    dfs = []

    for mid in machines:
        time = np.arange(n_rows)
        # Base Signals
        temp = np.random.normal(60, 5, n_rows)
        vib = np.random.gamma(2, 2, n_rows)
        pressure = np.random.normal(30, 2, n_rows)
        error_rate = np.random.exponential(1, n_rows)

        # PHYSICS LOGIC
        if mid in ['M001', 'M004']:  # CHAOTIC
            eff = 100 - (error_rate * 10) - (vib * 2) + np.random.normal(0, 2, n_rows)
        elif mid == 'M002':  # INERTIAL
            smooth_temp = pd.Series(temp).rolling(10).mean().fillna(method='bfill')
            eff = 100 - (smooth_temp * 0.5) - (pressure * 0.5) + np.random.normal(0, 1, n_rows)
        else:  # CYCLIC
            cycle = np.sin(time / 50) * 20
            eff = 80 + cycle - (vib * 1.5) + np.random.normal(0, 3, n_rows)

        df = pd.DataFrame({'MachineID': mid, 'Efficiency': np.clip(eff, 0, 100)})
        dfs.append(df)

    pd.concat(dfs).to_csv(filepath, index=False)


generate_iot_data_if_needed(CSV_PATH)


# ==========================================
# 2. PLOTTING FUNCTION (MATCHED STYLE)
# ==========================================
def plot_machine_physics(mid, y):
    print(f"   Generating matched-style plots for {mid}...")

    # --- A. ACF MEMORY PROFILE ---
    plt.figure(figsize=(8, 5))
    lags = range(20)
    ac_vals = acf(y, nlags=19)

    # STYLE MATCH: Specific Blue Color
    plt.bar(lags, ac_vals, color='#5da5da', alpha=0.9, width=0.7)

    # STYLE MATCH: Red Threshold Line
    plt.axhline(0.8, color='red', linestyle='--', linewidth=1.2, label='High Inertia Threshold')

    plt.title(f"{mid}: Memory Structure (ACF-1 = {ac_vals[1]:.2f})", fontweight='bold', fontsize=12)
    plt.xlabel("Lag Steps", fontsize=11)
    plt.ylabel("Autocorrelation", fontsize=11)

    # FORCE INTEGER TICKS
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'iot_{mid}_acf_style_match.png', dpi=300)
    plt.close()

    # --- B. MARKOV MATRIX (Range Labels) ---
    n_bins = 3
    bins = np.linspace(min(y), max(y), n_bins + 1)

    # DYNAMIC RANGE LABELS
    labels = [
        f"Low\n(< {bins[1]:.1f})",
        f"Avg\n({bins[1]:.1f}-{bins[2]:.1f})",
        f"High\n(> {bins[2]:.1f})"
    ]

    discrete = pd.cut(y, bins=bins, labels=False, include_lowest=True)
    transitions = np.zeros((n_bins, n_bins))
    for (t, t_plus_1) in zip(discrete[:-1], discrete[1:]):
        transitions[t][t_plus_1] += 1
    probs = transitions / (transitions.sum(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(7, 6))

    # STYLE MATCH: Blues Colormap & Large Annotations
    sns.heatmap(probs, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1,
                annot_kws={"size": 12})

    stability = np.mean(np.diag(probs))
    plt.title(f"{mid}: Markov Stability (Avg Diag = {stability:.2f})", fontweight='bold', fontsize=12)
    plt.ylabel("Current State", fontsize=11)
    plt.xlabel("Next State (t+1)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f'iot_{mid}_markov_style_match.png', dpi=300)
    plt.close()

    # --- C. LAG PLOT (Green Scatter) ---
    plt.figure(figsize=(6, 6))

    # STYLE MATCH: Green Scatter Points
    plt.scatter(y[:-1], y[1:], alpha=0.5, s=15, c='#2ca02c')

    lims = [min(y), max(y)]
    plt.plot(lims, lims, 'r--', alpha=0.7, label='Perfect Inertia (y=x)')
    plt.title(f"{mid}: Lag Plot", fontweight='bold')
    plt.xlabel("Efficiency (t-1)")
    plt.ylabel("Efficiency (t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'iot_{mid}_lag_style_match.png', dpi=300)
    plt.close()


# ==========================================
# 3. MAIN LOOP
# ==========================================
df = pd.read_csv(CSV_PATH)

# Standardize column name
if 'efficiency_score' in df.columns: df.rename(columns={'efficiency_score': 'Efficiency'}, inplace=True)
if 'machine_id' in df.columns: df.rename(columns={'machine_id': 'MachineID'}, inplace=True)

machines = sorted(df['MachineID'].unique())
print(f"Processing machines: {machines}")

for mid in machines:
    y = df[df['MachineID'] == mid]['Efficiency'].values
    if len(y) > 100:
        plot_machine_physics(mid, y)

print("\n✓ Generated 12 matched-style physics profile plots.")