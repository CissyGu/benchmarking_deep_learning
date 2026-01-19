#!/usr/bin/env python3
"""
IoT Scale-Up (Final Visuals + Integer Epochs) - PyTorch
Scope:
  1. Load REAL dataset 'Real_Time_Production.csv'.
  2. Augment to 80k samples.
  3. Run 5 Experiments per Model per Target.
  4. Select BEST Run.
  5. Generate Plots with METRIC ANNOTATIONS & INTEGER EPOCHS.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # <--- NEW IMPORT FOR INTEGER TICKS
import seaborn as sns
import os
import warnings
import copy

warnings.filterwarnings('ignore')

# --- GPU SETUP ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ PyTorch detected GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ PyTorch detected Apple MPS")
else:
    device = torch.device("cpu")
    print("! No GPU detected. Using CPU.")

# CONFIGURATION
SEED_CSV_PATH = 'Real_Time_Production.csv'
OUTPUT_CSV_PATH = 'Massive_IoT_Real_Augmented.csv'
N_RUNS = 20
WINDOW_SIZE = 10
EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128

# --- MODEL CONFIG ---
UNITS = 32
DROPOUT_RATE = 0.2
L2_RATE = 0.001

print("=" * 70)
print(f"IoT SCALE-UP (Integer Epochs) - PYTORCH")
print("=" * 70)


# ==========================================
# 1. HYBRID DATA GENERATION
# ==========================================
def load_real_seed_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CRITICAL: '{filepath}' not found.")

    print(f"   -> Loading Real Data Seed: {filepath}")
    df = pd.read_csv(filepath)

    col_map = {
        'machine_id': 'MachineID', 'temperature': 'Temp', 'vibration_level': 'Vibration',
        'power_consumption': 'Power', 'pressure': 'Pressure', 'error_rate': 'ErrorRate',
        'efficiency_score': 'Efficiency', 'timestamp': 'Timestamp'
    }

    df = df.rename(columns=col_map)
    df = df.sort_values('Timestamp').fillna(method='ffill').fillna(method='bfill')
    return df


def generate_massive_data(seed_path, output_path):
    if os.path.exists(output_path):
        print(f"✓ Found existing massive dataset: {output_path}")
        return pd.read_csv(output_path)

    seed_df = load_real_seed_data(seed_path)
    augmented_dfs = []
    target_per_machine = 20000

    print("   -> Augmenting Real Data to 80,000 rows...")
    for mid in seed_df['MachineID'].unique():
        subset = seed_df[seed_df['MachineID'] == mid]
        if len(subset) == 0: continue

        aug_sample = subset.sample(n=target_per_machine, replace=True).copy()

        feature_cols = ['Temp', 'Vibration', 'Power', 'Pressure', 'ErrorRate']
        for col in feature_cols:
            col_std = aug_sample[col].std()
            noise = np.random.normal(0, col_std * 0.05, size=len(aug_sample))
            aug_sample[col] = aug_sample[col] + noise

        eff_noise = np.random.normal(0, 0.5, size=len(aug_sample))
        aug_sample['Efficiency'] = np.clip(aug_sample['Efficiency'] + eff_noise, 0, 100)
        aug_sample['Timestamp'] = pd.date_range('2025-01-01', periods=target_per_machine, freq='T')
        augmented_dfs.append(aug_sample)

    full_df = pd.concat(augmented_dfs).reset_index(drop=True)
    full_df.to_csv(output_path, index=False)
    return full_df


main_df = generate_massive_data(SEED_CSV_PATH, OUTPUT_CSV_PATH)


# ==========================================
# 2. SHARED UTILS
# ==========================================
def engineer_data(df_subset):
    df = df_subset.copy()
    df['Mech_Stress'] = df['Vibration'] * df['Pressure']
    df['Therm_Stress'] = df['Temp'] * df['Power']

    for c in ['Temp', 'Vibration', 'Power']:
        df[f'{c}_Smooth'] = df.groupby('MachineID')[c].transform(lambda x: x.rolling(3).mean())

    df = df.dropna().reset_index(drop=True)
    features = ['Temp_Smooth', 'Vibration_Smooth', 'Power_Smooth', 'Mech_Stress', 'Therm_Stress', 'ErrorRate']
    target = 'Efficiency'

    inv_eff = df[target].max() - df[target]
    df['Sample_Weight'] = 1.0 + (inv_eff / inv_eff.max()) * 9.0
    return df, features, target


def create_windows(df, X, y, w, time_steps=WINDOW_SIZE):
    Xs, ys, ws = [], [], []
    for mid in df['MachineID'].unique():
        indices = df.index[df['MachineID'] == mid].tolist()
        X_sub, y_sub, w_sub = X[indices], y[indices], w[indices]
        for i in range(len(X_sub) - time_steps):
            Xs.append(X_sub[i:(i + time_steps)])
            ys.append(y_sub[i + time_steps - 1])
            ws.append(w_sub[i + time_steps - 1])
    return np.array(Xs), np.array(ys), np.array(ws)


# ==========================================
# 3. PYTORCH MODELS
# ==========================================
class BaseIoTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dropout_final = nn.Dropout(DROPOUT_RATE)
        self.dense1 = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.dense_out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def combine_and_predict(self, deep_out, wide_input):
        d = self.dropout_final(deep_out)
        combined = torch.cat([d, wide_input], dim=1)
        z = self.dense1(combined)
        z = self.relu(z)
        z = self.dropout_final(z)
        return self.dense_out(z)


class ResNet(BaseIoTModel):
    def __init__(self, input_dim):
        super().__init__(input_dim, UNITS)
        self.conv1 = nn.Conv1d(input_dim, UNITS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(UNITS, UNITS, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        out = self.relu(self.conv1(x_t))
        skip = out
        out = self.conv2(out)
        out = out + skip
        deep = self.pool(out).squeeze(-1)
        wide = x[:, -1, :]
        return self.combine_and_predict(deep, wide)


class LSTMModel(BaseIoTModel):
    def __init__(self, input_dim):
        super().__init__(input_dim, UNITS)
        self.lstm = nn.LSTM(input_dim, UNITS, batch_first=True, dropout=DROPOUT_RATE)

    def forward(self, x):
        out, _ = self.lstm(x)
        deep = out[:, -1, :]
        wide = x[:, -1, :]
        return self.combine_and_predict(deep, wide)


class CNNLSTM(BaseIoTModel):
    def __init__(self, input_dim):
        super().__init__(input_dim, UNITS)
        self.conv = nn.Conv1d(input_dim, UNITS, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(UNITS, UNITS, batch_first=True, dropout=DROPOUT_RATE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        out = self.relu(self.conv(x_t))
        out = self.pool(out)
        out = out.transpose(1, 2)
        out, _ = self.lstm(out)
        deep = out[:, -1, :]
        wide = x[:, -1, :]
        return self.combine_and_predict(deep, wide)


class TransformerModel(BaseIoTModel):
    def __init__(self, input_dim):
        super().__init__(input_dim, UNITS)
        self.conv = nn.Conv1d(input_dim, UNITS, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=UNITS, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(UNITS, eps=1e-6)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        out = self.relu(self.conv(x_t))
        out = out.transpose(1, 2)
        attn_out, _ = self.attn(out, out, out)
        out = self.norm(out + attn_out)
        out = out.transpose(1, 2)
        deep = self.pool(out).squeeze(-1)
        wide = x[:, -1, :]
        return self.combine_and_predict(deep, wide)


class LSTMTrans(BaseIoTModel):
    def __init__(self, input_dim):
        super().__init__(input_dim, UNITS)
        self.lstm = nn.LSTM(input_dim, UNITS, batch_first=True, dropout=DROPOUT_RATE)
        self.attn = nn.MultiheadAttention(embed_dim=UNITS, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(UNITS, eps=1e-6)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        out = self.norm(out + attn_out)
        deep = out[:, -1, :]
        wide = x[:, -1, :]
        return self.combine_and_predict(deep, wide)


class TriHybrid(BaseIoTModel):
    def __init__(self, input_dim):
        super().__init__(input_dim, UNITS * 2)
        self.conv = nn.Conv1d(input_dim, UNITS, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.lstm = nn.LSTM(input_dim, UNITS, batch_first=True, dropout=DROPOUT_RATE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        c = self.relu(self.conv(x_t))
        c = self.pool(c).squeeze(-1)
        l, _ = self.lstm(x)
        l = l[:, -1, :]
        deep = torch.cat([c, l], dim=1)
        wide = x[:, -1, :]
        return self.combine_and_predict(deep, wide)


def get_model(name, input_dim):
    if name == "ResNet":
        return ResNet(input_dim)
    elif name == "LSTM":
        return LSTMModel(input_dim)
    elif name == "CNN-LSTM":
        return CNNLSTM(input_dim)
    elif name == "Transformer":
        return TransformerModel(input_dim)
    elif name == "LSTM-Trans":
        return LSTMTrans(input_dim)
    elif name == "Tri-Hybrid":
        return TriHybrid(input_dim)
    return None


# ==========================================
# 4. TRAINING HELPER
# ==========================================
def train_model(model, train_loader, val_loader, epochs, patience):
    criterion = nn.HuberLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=L2_RATE)

    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for X_b, y_b, w_b in train_loader:
            X_b, y_b, w_b = X_b.to(device), y_b.to(device), w_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            raw_loss = criterion(outputs, y_b)
            loss = (raw_loss * w_b).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)
            total_samples += X_b.size(0)

        epoch_train_loss = running_loss / total_samples
        history['train_loss'].append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for X_b, y_b, w_b in val_loader:
                X_b, y_b, w_b = X_b.to(device), y_b.to(device), w_b.to(device)
                outputs = model(X_b)
                raw_loss = criterion(outputs, y_b)
                loss = (raw_loss * w_b).mean()
                val_loss += loss.item() * X_b.size(0)
                val_samples += X_b.size(0)

        epoch_val_loss = val_loss / val_samples
        history['val_loss'].append(epoch_val_loss)

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_wts)
    return model, history


# ==========================================
# 5. SHARED SCALE PLOTTING FUNCTIONS
# ==========================================
def get_global_limits(best_runs_data, type='loss'):
    all_vals = []
    for data in best_runs_data.values():
        if type == 'loss':
            all_vals.extend(data['history']['train_loss'])
            all_vals.extend(data['history']['val_loss'])
        elif type == 'pred':
            all_vals.extend(data['true'])
            all_vals.extend(data['pred'])

    if not all_vals: return 0, 1
    vmin, vmax = min(all_vals), max(all_vals)
    span = vmax - vmin
    return vmin - (span * 0.05), vmax + (span * 0.05)


def plot_subplots_learning_curves(best_runs_data, target, model_names):
    if not best_runs_data: return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    y_min, y_max = get_global_limits(best_runs_data, type='loss')

    for i, m_name in enumerate(model_names):
        ax = axes[i]
        if m_name in best_runs_data:
            data = best_runs_data[m_name]
            hist = data['history']

            ax.plot(hist['train_loss'], label='Train', color='royalblue', linewidth=2)
            ax.plot(hist['val_loss'], label='Val', color='darkorange', linewidth=2)

            ax.set_title(f"{m_name} (Run {data['run_id']})", fontweight='bold')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Huber Loss")
            ax.set_ylim(y_min, y_max)

            # FORCE INTEGER TICKS
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

    plt.suptitle(f"{target}: Learning Curves (Shared Y-Scale)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"subplots_learning_curve_{target}.png")
    plt.close()


def plot_subplots_predictions(best_runs_data, target, model_names):
    if not best_runs_data: return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    y_min, y_max = get_global_limits(best_runs_data, type='pred')
    limit = 100

    for i, m_name in enumerate(model_names):
        ax = axes[i]
        if m_name in best_runs_data:
            data = best_runs_data[m_name]
            true = data['true'][:limit]
            pred = data['pred'][:limit]

            ax.plot(true, label='Actual', color='black', alpha=0.6, linewidth=1.5)
            ax.plot(pred, label='Predicted', color='dodgerblue', linestyle='--', linewidth=1.5)

            # Annotations
            r2 = data['r2']
            rmse = data['rmse']
            mae = data['mae']
            text_str = f'R²: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}'
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f"{m_name} (Run {data['run_id']})", fontweight='bold')
            ax.set_xlabel("Sample")
            ax.set_ylabel("Efficiency")
            ax.set_ylim(y_min, y_max)
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

    plt.suptitle(f"{target}: Predictions (Shared Y-Scale)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"subplots_prediction_{target}.png")
    plt.close()


# ==========================================
# 6. MAIN EXPERIMENT LOOP
# ==========================================
model_names = ["ResNet", "LSTM", "CNN-LSTM", "Transformer", "LSTM-Trans", "Tri-Hybrid"]
targets = ['M001', 'M002', 'M003', 'M004', 'GLOBAL']

summary_lines = []
summary_lines.append(f"{'Experiment':<12} | {'Model':<15} | {'R2 (Mean ± Std)':<22} | {'RMSE':<12} | {'MAE':<12}")
summary_lines.append("-" * 85)

for target in targets:
    print(f"\n\n>>> STARTING EXPERIMENT: {target} <<<")

    if target == 'GLOBAL':
        df_target = main_df.copy()
    else:
        df_target = main_df[main_df['MachineID'] == target].copy()

    df_proc, feats, targ_col = engineer_data(df_target)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_sc = scaler_X.fit_transform(df_proc[feats])
    y_sc = scaler_y.fit_transform(df_proc[[targ_col]])
    w_sc = df_proc['Sample_Weight'].values

    X_data, y_data, w_data = create_windows(df_proc, X_sc, y_sc, w_sc)

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    w_tensor = torch.tensor(w_data, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_data.shape[2]
    logs = []

    best_runs_for_target = {}

    for m_name in model_names:
        print(f"   Model: {m_name:<12}", end="")

        best_run_score = -float('inf')
        best_run_data = {}

        for run in range(1, N_RUNS + 1):
            model = get_model(m_name, input_dim)
            model = model.to(device)

            model, history = train_model(model, train_loader, val_loader, EPOCHS, PATIENCE)

            model.eval()
            preds = []
            trues = []
            with torch.no_grad():
                for X_b, y_b, _ in val_loader:
                    X_b = X_b.to(device)
                    outputs = model(X_b)
                    preds.append(outputs.cpu().numpy())
                    trues.append(y_b.cpu().numpy())

            pred_arr = np.concatenate(preds)
            true_arr = np.concatenate(trues)
            pred_inv = scaler_y.inverse_transform(pred_arr)
            true_inv = scaler_y.inverse_transform(true_arr)

            r2 = r2_score(true_inv, pred_inv)
            rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
            mae = mean_absolute_error(true_inv, pred_inv)

            logs.append({'Run': run, 'Model': m_name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
            print(f".", end="")

            if r2 > best_run_score:
                best_run_score = r2
                best_run_data = {
                    'run_id': run,
                    'history': history,
                    'true': true_inv,
                    'pred': pred_inv,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                }

            del model
            torch.cuda.empty_cache()

        print(" Done.")
        if best_run_data:
            best_runs_for_target[m_name] = best_run_data

    # --- GENERATE PLOTS ---
    print(f"   -> Generating Plots for {target}...")
    plot_subplots_learning_curves(best_runs_for_target, target, model_names)
    plot_subplots_predictions(best_runs_for_target, target, model_names)

    # --- SAVE LOGS ---
    df_res = pd.DataFrame(logs)
    df_res.to_csv(f"stability_log_{target}.csv", index=False)

    stats = df_res.groupby('Model').agg(['mean', 'std'])
    stats = stats.sort_values(('R2', 'mean'), ascending=False)

    print(f"\n   --- {target} RESULTS ---")
    for m_name, row in stats.iterrows():
        r2_str = f"{row[('R2', 'mean')]:.3f} ± {row[('R2', 'std')]:.3f}"
        rmse_str = f"{row[('RMSE', 'mean')]:.3f}"
        mae_str = f"{row[('MAE', 'mean')]:.3f}"

        print(f"   {m_name:<15} : R2={r2_str} | RMSE={rmse_str} | MAE={mae_str}")
        summary_lines.append(f"{target:<12} | {m_name:<15} | {r2_str:<22} | {rmse_str:<12} | {mae_str:<12}")

    # --- GENERATE BOXPLOTS ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.boxplot(x='Model', y='R2', data=df_res, palette='viridis', ax=axes[0])
    axes[0].set_title(f"{target} - R2 Score", fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)

    sns.boxplot(x='Model', y='RMSE', data=df_res, palette='magma', ax=axes[1])
    axes[1].set_title(f"{target} - RMSE", fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    sns.boxplot(x='Model', y='MAE', data=df_res, palette='plasma', ax=axes[2])
    axes[2].set_title(f"{target} - MAE", fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'boxplot_{target}_metrics.png')
    plt.close()

with open("FINAL_SUMMARY_REPORT.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\n\n" + "=" * 50)
print("ALL EXPERIMENTS COMPLETED.")
print("Outputs Generated Per Target:")
print("  1. stability_log_*.csv")
print("  2. boxplot_*_metrics.png (R2, RMSE, MAE)")
print("  3. subplots_learning_curve_*.png (Shared Scale)")
print("  4. subplots_prediction_*.png (Shared Scale + Metrics)")
print("=" * 50)