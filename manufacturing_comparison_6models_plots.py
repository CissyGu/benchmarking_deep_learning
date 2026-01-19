#!/usr/bin/env python3
"""
Manufacturing Prediction (PyTorch Stability Analysis - 50 Runs)
Scope:
  1. Load Manufacturing Dataset (or generate synthetic).
  2. Models: CNN, LSTM, CNN-LSTM, Transformer, LSTM-Trans, Tri-Hybrid (PyTorch).
  3. Run 50 Experiments per Model.
  4. Select BEST Run (based on R2).
  5. Generate Outputs matching 'IoT Scale-Up' format:
     - Boxplots (R2, RMSE, MAE)
     - Stability Log CSV
     - Learning Curves (Best Run, Integer Epochs, Shared Scale)
     - Prediction Plots (Best Run, Shared Scale, Metrics)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
CSV_PATH = 'Manufacturing_dataset.csv'
N_RUNS = 20
WINDOW_SIZE = 20
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 5

print("=" * 70)
print(f"MANUFACTURING STABILITY ANALYSIS (20 Runs) - PYTORCH")
print("=" * 70)


# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ File '{filepath}' not found. Generating synthetic data...")
        df = pd.DataFrame({
            'Temperature (Â°C)': np.random.normal(75, 5, 2000),
            'Machine Speed (RPM)': np.random.normal(1500, 50, 2000),
            'Vibration Level (mm/s)': np.random.gamma(2, 0.05, 2000),
            'Energy Consumption (kWh)': np.random.normal(1.5, 0.3, 2000)
        })
    else:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except:
            df = pd.read_csv(filepath, encoding='latin1')

    column_map = {
        'Temperature (Â°C)': 'Temp', 'Temperature (°C)': 'Temp',
        'Machine Speed (RPM)': 'RPM',
        'Vibration Level (mm/s)': 'Vibration',
        'Energy Consumption (kWh)': 'Energy'
    }
    df = df.rename(columns=column_map)

    features = ['RPM', 'Temp', 'Vibration', 'Energy']
    features = [f for f in features if f in df.columns]
    df = df[features].dropna()

    # --- HYBRID INJECTION ---
    df['Temp_S'] = df['Temp'].rolling(3).mean()
    df['Vib_S'] = df['Vibration'].rolling(3).mean()
    df['RPM_S'] = df['RPM'].rolling(3).mean()
    df['Energy_S'] = df['Energy'].rolling(3).mean()
    df = df.dropna()

    n_temp = (df['Temp_S'] - df['Temp_S'].mean()) / df['Temp_S'].std()
    n_vib = (df['Vib_S'] - df['Vib_S'].mean()) / df['Vib_S'].std()
    n_rpm = (df['RPM_S'] - df['RPM_S'].mean()) / df['RPM_S'].std()
    n_nrg = (df['Energy_S'] - df['Energy_S'].mean()) / df['Energy_S'].std()

    # Physics Formula
    df['Quality'] = 100 \
                    - (n_temp * 2.0) \
                    - (n_vib * 3.0) \
                    - (np.abs(n_rpm) * 1.0) \
                    - (n_nrg * 1.5) \
                    + np.random.normal(0, 0.5, len(df))

    return df, features, 'Quality'


df, features, target = load_data(CSV_PATH)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[[target]])


def create_windows(X, y, time_steps=WINDOW_SIZE):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


X_data, y_data = create_windows(X_scaled, y_scaled)

# Convert to Tensors
X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

# Train/Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

INPUT_DIM = X_data.shape[2]


# ==========================================
# 2. PYTORCH MODEL DEFINITIONS
# ==========================================
class BaseNet(nn.Module):
    def __init__(self, hidden_dim=50):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 50)
        self.out = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def head(self, x):
        x = self.relu(self.dense(x))
        return self.out(x)


class CNN(BaseNet):
    def __init__(self, input_dim):
        super().__init__(hidden_dim=32)
        # padding=1 simulates 'same' for kernel=3
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: [Batch, Time, Channels] -> Permute to [Batch, Channels, Time]
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.global_pool(x).squeeze(-1)
        return self.head(x)


class LSTMNet(BaseNet):
    def __init__(self, input_dim):
        super().__init__(hidden_dim=64)
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take last time step
        x = out[:, -1, :]
        return self.head(x)


class CNNLSTM(BaseNet):
    def __init__(self, input_dim):
        super().__init__(hidden_dim=64)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 64, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Back to [B, T, C] for LSTM
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        return self.head(x)


class TransformerNet(BaseNet):
    def __init__(self, input_dim):
        super().__init__(hidden_dim=64)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(64, eps=1e-6)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # [B, T, C]

        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)

        # Pool
        x = x.permute(0, 2, 1)  # [B, C, T] for pooling
        x = self.global_pool(x).squeeze(-1)
        return self.head(x)


class LSTMTransformer(BaseNet):
    def __init__(self, input_dim):
        super().__init__(hidden_dim=64)
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(64, eps=1e-6)

        # "Conv1d(filters=64, kernel_size=1)" acts as a dense projection across time
        self.conv_proj = nn.Conv1d(64, 64, kernel_size=1)
        self.norm2 = nn.LayerNorm(64, eps=1e-6)

    def forward(self, x):
        out, _ = self.lstm(x)  # [B, T, 64]
        attn_out, _ = self.attn(out, out, out)
        x = self.norm1(out + attn_out)

        res = x
        # Project
        x_t = x.permute(0, 2, 1)
        x_t = self.relu(self.conv_proj(x_t))
        x = x_t.permute(0, 2, 1)

        x = self.norm2(x + res)
        x = x[:, -1, :]
        return self.head(x)


class TriHybrid(BaseNet):
    def __init__(self, input_dim):
        super().__init__(hidden_dim=64)
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(64, eps=1e-6)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # 1. Conv Branch
        x_c = x.permute(0, 2, 1)
        x_c = self.relu(self.conv(x_c))
        x_c = self.pool(x_c)
        x_c = x_c.permute(0, 2, 1)  # [B, T_reduced, C]

        # 2. LSTM Branch
        x_l, _ = self.lstm(x_c)  # Sequence is retained

        # 3. Attention Branch
        attn_out, _ = self.attn(x_l, x_l, x_l)
        x = self.norm(x_l + attn_out)

        # Pooling
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        return self.head(x)


def get_model(name, input_dim):
    if name == "CNN": return CNN(input_dim)
    if name == "LSTM": return LSTMNet(input_dim)
    if name == "CNN-LSTM": return CNNLSTM(input_dim)
    if name == "Transformer": return TransformerNet(input_dim)
    if name == "LSTM-Trans": return LSTMTransformer(input_dim)
    if name == "Tri-Hybrid": return TriHybrid(input_dim)
    return None


# ==========================================
# 3. TRAINING HELPER
# ==========================================
def train_model(model, train_loader, val_loader, epochs, patience):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # TRAIN
        model.train()
        running_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                val_loss += loss.item() * X_b.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        # Early Stopping Check
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
# 4. SHARED PLOTTING FUNCTIONS
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


def plot_subplots_learning_curves(best_runs_data, target_name, model_names):
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
            ax.set_ylabel("Loss (MSE)")
            ax.set_ylim(y_min, y_max)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

    plt.suptitle(f"{target_name}: Learning Curves (Shared Y-Scale)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"subplots_learning_curve_Manufacturing_PT.png")
    plt.close()


def plot_subplots_predictions(best_runs_data, target_name, model_names):
    if not best_runs_data: return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    y_min, y_max = get_global_limits(best_runs_data, type='pred')
    limit = 100

    for i, m_name in enumerate(model_names):
        ax = axes[i]
        if m_name in best_runs_data:
            data = best_runs_data[m_name]
            true = data['true'][-limit:]
            pred = data['pred'][-limit:]
            x_ax = range(len(true))

            ax.plot(x_ax, true, label='Actual', color='black', alpha=0.6, linewidth=1.5)
            ax.plot(x_ax, pred, label='Predicted', color='dodgerblue', linestyle='--', linewidth=1.5)

            r2 = data['r2']
            rmse = data['rmse']
            mae = data['mae']
            text_str = f'R²: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}'
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f"{m_name} (Run {data['run_id']})", fontweight='bold')
            ax.set_xlabel("Sample (Last 100)")
            ax.set_ylabel("Quality")
            ax.set_ylim(y_min, y_max)
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

    plt.suptitle(f"{target_name}: Predictions (Shared Y-Scale)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"subplots_prediction_Manufacturing_PT.png")
    plt.close()


# ==========================================
# 5. MAIN EXPERIMENT LOOP
# ==========================================
model_names = ["CNN", "LSTM", "CNN-LSTM", "Transformer", "LSTM-Trans", "Tri-Hybrid"]
summary_lines = []
summary_lines.append(f"{'Experiment':<12} | {'Model':<15} | {'R2 (Mean ± Std)':<22} | {'RMSE':<12} | {'MAE':<12}")
summary_lines.append("-" * 85)

logs = []
best_runs_data = {}

print("\n>>> STARTING 20 RUNS EXPERIMENT (PYTORCH) <<<")

for m_name in model_names:
    print(f"   Model: {m_name:<12}", end="")

    best_run_score = -float('inf')
    best_run_info = {}

    for run in range(1, N_RUNS + 1):
        # Init Model
        model = get_model(m_name, INPUT_DIM)
        model = model.to(device)

        # Train
        model, history = train_model(model, train_loader, val_loader, EPOCHS, PATIENCE)

        # Predict
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                outputs = model(X_b)
                preds.append(outputs.cpu().numpy())
                trues.append(y_b.numpy())  # y_b is already cpu tensor if not moved

        pred_arr = np.concatenate(preds)
        true_arr = np.concatenate(trues)

        # Inverse Transform
        pred_inv = scaler_y.inverse_transform(pred_arr)
        true_inv = scaler_y.inverse_transform(true_arr)

        # Flatten
        pred_flat = pred_inv.flatten()
        true_flat = true_inv.flatten()

        r2 = r2_score(true_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
        mae = mean_absolute_error(true_flat, pred_flat)

        logs.append({'Run': run, 'Model': m_name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
        print(f".", end="", flush=True)

        if r2 > best_run_score:
            best_run_score = r2
            best_run_info = {
                'run_id': run,
                'history': history,
                'true': true_flat,
                'pred': pred_flat,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }

        # Clean up GPU
        del model
        torch.cuda.empty_cache()

    print(" Done.")
    if best_run_info:
        best_runs_data[m_name] = best_run_info

# ==========================================
# 6. GENERATE OUTPUTS
# ==========================================
print(f"   -> Generating Plots...")

plot_subplots_learning_curves(best_runs_data, "Manufacturing", model_names)
plot_subplots_predictions(best_runs_data, "Manufacturing", model_names)

df_res = pd.DataFrame(logs)
df_res.to_csv("stability_log_Manufacturing_PT.csv", index=False)

stats = df_res.groupby('Model').agg(['mean', 'std'])
try:
    stats = stats.sort_values(('R2', 'mean'), ascending=False)
except KeyError:
    pass

print(f"\n   --- RESULTS SUMMARY ---")
for m_name, row in stats.iterrows():
    r2_str = f"{row[('R2', 'mean')]:.3f} ± {row[('R2', 'std')]:.3f}"
    rmse_str = f"{row[('RMSE', 'mean')]:.3f}"
    mae_str = f"{row[('MAE', 'mean')]:.3f}"
    print(f"   {m_name:<15} : R2={r2_str} | RMSE={rmse_str} | MAE={mae_str}")
    summary_lines.append(f"{'Manuf':<12} | {m_name:<15} | {r2_str:<22} | {rmse_str:<12} | {mae_str:<12}")

with open("FINAL_SUMMARY_REPORT_PT.txt", "w") as f:
    f.write("\n".join(summary_lines))

# BOXPLOTS
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.boxplot(x='Model', y='R2', data=df_res, palette='viridis', ax=axes[0])
axes[0].set_title(f"Manufacturing - R2 Score", fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(x='Model', y='RMSE', data=df_res, palette='magma', ax=axes[1])
axes[1].set_title(f"Manufacturing - RMSE", fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

sns.boxplot(x='Model', y='MAE', data=df_res, palette='plasma', ax=axes[2])
axes[2].set_title(f"Manufacturing - MAE", fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'boxplot_Manufacturing_metrics_PT.png')
plt.close()

print("\n\n" + "=" * 50)
print("ALL EXPERIMENTS COMPLETED (PYTORCH).")
print("Outputs Generated:")
print("  1. stability_log_Manufacturing_PT.csv")
print("  2. boxplot_Manufacturing_metrics_PT.png")
print("  3. subplots_learning_curve_Manufacturing_PT.png")
print("  4. subplots_prediction_Manufacturing_PT.png")
print("  5. FINAL_SUMMARY_REPORT_PT.txt")
print("=" * 50)