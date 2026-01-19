#!/usr/bin/env python3
"""
Manufacturing Data Physics Profiling (Improved Labels)
Goal: Generate polished "Physics Proof" plots with:
  1. Range values in Markov labels (e.g., "Low (<85)").
  2. Integer X-axis for ACF plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
from matplotlib.ticker import MaxNLocator
import os

# CONFIGURATION
CSV_PATH = 'Manufacturing_dataset.csv'


def load_and_inject_physics(filepath):
    if not os.path.exists(filepath):
        print(f"❌ File '{filepath}' not found.")
        return None

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        df = pd.read_csv(filepath, encoding='latin1')

    # Standardize Names
    column_map = {
        'Machine Speed (RPM)': 'RPM',
        'Temperature (Â°C)': 'Temp', 'Temperature (°C)': 'Temp',
        'Vibration Level (mm/s)': 'Vibration',
        'Energy Consumption (kWh)': 'Energy'
    }

    df_clean = df.copy()
    for actual_col in df.columns:
        for map_key, map_val in column_map.items():
            if map_key[:10] in actual_col:
                df_clean.rename(columns={actual_col: map_val}, inplace=True)

    features = ['RPM', 'Temp', 'Vibration', 'Energy']
    features = [f for f in features if f in df_clean.columns]
    df_hybrid = df_clean[features].dropna().reset_index(drop=True)

    # Inject Physics (Smoothing + Formula)
    for col in features:
        df_hybrid[col] = df_hybrid[col].rolling(3).mean()
    df_hybrid = df_hybrid.dropna().reset_index(drop=True)

    # Normalize
    n_temp = (df_hybrid['Temp'] - df_hybrid['Temp'].mean()) / df_hybrid['Temp'].std()
    n_vib = (df_hybrid['Vibration'] - df_hybrid['Vibration'].mean()) / df_hybrid['Vibration'].std()
    n_rpm = (df_hybrid['RPM'] - df_hybrid['RPM'].mean()) / df_hybrid['RPM'].std()
    n_nrg = (df_hybrid['Energy'] - df_hybrid['Energy'].mean()) / df_hybrid['Energy'].std()

    # The Formula
    df_hybrid['Quality'] = 100 - (n_temp * 2.0) - (n_vib * 3.0) - (np.abs(n_rpm) * 1.0) - (
                n_nrg * 1.5) + np.random.normal(0, 0.5, len(df_hybrid))

    return df_hybrid['Quality'].values


y = load_and_inject_physics(CSV_PATH)

if y is not None:
    print("Generating Improved Physics Plots...")

    # --- PLOT 1: ACF MEMORY PROFILE (Integer Axis) ---
    plt.figure(figsize=(8, 5))
    lags = range(20)
    ac_vals = acf(y, nlags=19)
    plt.bar(lags, ac_vals, color='#5da5da', alpha=0.9, width=0.7)
    plt.axhline(0.8, color='red', linestyle='--', linewidth=1.2, label='High Inertia Threshold')

    plt.title(f"Hybrid Data: Memory Structure (ACF-1 = {ac_vals[1]:.2f})", fontweight='bold', fontsize=12)
    plt.xlabel("Lag Steps", fontsize=11)
    plt.ylabel("Autocorrelation", fontsize=11)

    # FORCE INTEGER TICKS
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend()
    plt.tight_layout()
    plt.savefig('hybrid_physics_acf_improved.png', dpi=300)
    print("✓ Saved 'hybrid_physics_acf_improved.png'")
    plt.close()

    # --- PLOT 2: MARKOV MATRIX (Range Labels) ---
    # Binning logic
    n_bins = 3
    bins = np.linspace(min(y), max(y), n_bins + 1)

    # Create informative labels with ranges
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
    sns.heatmap(probs, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1,
                annot_kws={"size": 12})

    plt.title("Hybrid Data: Markov Stability\n(Diagonal Dominance = Predictability)", fontweight='bold', fontsize=12)
    plt.ylabel("Current State", fontsize=11)
    plt.xlabel("Next State (t+1)", fontsize=11)
    plt.tight_layout()
    plt.savefig('hybrid_physics_markov_improved.png', dpi=300)
    print("✓ Saved 'hybrid_physics_markov_improved.png'")
    plt.close()

    # --- PLOT 3: LAG PLOT ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y[:-1], y[1:], alpha=0.5, s=15, c='#2ca02c')
    lims = [min(y), max(y)]
    plt.plot(lims, lims, 'r--', alpha=0.7, label='Perfect Inertia (y=x)')
    plt.title(f"Hybrid Data: Lag Plot (Strong Inertia)", fontweight='bold')
    plt.xlabel("Quality (t-1)")
    plt.ylabel("Quality (t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hybrid_physics_lag_improved.png', dpi=300)
    print("✓ Saved 'hybrid_physics_lag_improved.png'")
    plt.close()