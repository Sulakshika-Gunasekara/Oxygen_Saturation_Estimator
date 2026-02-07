#!/usr/bin/env python3
"""
Heavy Transformer baseline for SpO2 regression using rPPG features.

Inputs (choose ONE):
  --npz <file.npz>                        # single dataset file
  --npz_dir <folder> [--pattern PATTERN]  # multiple dataset files (default "*.npz")

Train/Test split:
  - Default: 70/30 (train/test)
  - Default split mode: group-wise by file (to reduce leakage across files)

Outputs (saved to --out_dir):
  - epoch_loss_summary.png           : Train vs Test loss curve (MSE)
  - epoch_accuracy_summary.png       : Train vs Test accuracy curve (|error| <= --acc_tol)
  - epoch_error_summary.png          : Train vs Test error curve (MAE)
  - hardware_requirements.png        : Inference hardware summary (params/size/latency/RAM estimate)
  - scatter_fit.png                  : Scientific Scatter Plot (GT vs Pred on test split)
  - bland_altman.png                 : Bland-Altman Plot (test split)
  - train_test_results.csv           : Final test metrics + efficiency metrics
  - heavy_tf_summary.json            : Run config + final metrics (JSON)
  - tf_best_model.pt                 : Best model weights (based on lowest test loss)
"""

import os
import glob
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------- Utilities -----------------------------
def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_size_bytes(model: nn.Module, path: str) -> int:
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)


def estimate_inference_ram_mb(params: int, bytes_per_param: int = 4, multiplier: float = 3.0) -> float:
    """
    Very rough estimate for inference memory at batch size 1:
      params * bytes_per_param * multiplier

    multiplier>1 attempts to account for buffers/activations/overhead.
    """
    return (params * bytes_per_param * float(multiplier)) / (1024.0 ** 2)


# ----------------------------- Plotting Functions -----------------------------
def _safe_makedirs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def plot_loss_curve(history: dict, out_dir: str):
    """
    Generates 'epoch_loss_summary.png'.
    Plots Train Loss (MSE) and Test Loss (MSE) across epochs.
    """
    try:
        _safe_makedirs(out_dir)
        epochs = np.array(history["epochs"], dtype=int)

        train_loss = np.array(history["train_loss"], dtype=float)
        test_loss = np.array(history["test_loss"], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label="Train MSE", color="blue")
        plt.plot(epochs, test_loss, label="Test MSE", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Train vs Test MSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "epoch_loss_summary.png"))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'epoch_loss_summary.png')}")
    except Exception as e:
        print(f"Failed to plot loss curve: {e}")


def plot_accuracy_curve(history: dict, out_dir: str, acc_tol: float):
    """
    Generates 'epoch_accuracy_summary.png'.
    Accuracy is defined as: mean(|pred - true| <= acc_tol)
    """
    try:
        _safe_makedirs(out_dir)
        epochs = np.array(history["epochs"], dtype=int)

        train_acc = np.array(history["train_acc"], dtype=float)
        test_acc = np.array(history["test_acc"], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, label=f"Train Acc (|err|≤{acc_tol:g})", color="blue")
        plt.plot(epochs, test_acc, label=f"Test Acc (|err|≤{acc_tol:g})", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train vs Test Accuracy (Tolerance-based)")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "epoch_accuracy_summary.png"))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'epoch_accuracy_summary.png')}")
    except Exception as e:
        print(f"Failed to plot accuracy curve: {e}")


def plot_error_curve(history: dict, out_dir: str):
    """
    Generates 'epoch_error_summary.png'.
    Plots Train MAE and Test MAE across epochs.
    """
    try:
        _safe_makedirs(out_dir)
        epochs = np.array(history["epochs"], dtype=int)

        train_mae = np.array(history["train_mae"], dtype=float)
        test_mae = np.array(history["test_mae"], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_mae, label="Train MAE", color="blue")
        plt.plot(epochs, test_mae, label="Test MAE", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Train vs Test Error (MAE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "epoch_error_summary.png"))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'epoch_error_summary.png')}")
    except Exception as e:
        print(f"Failed to plot error curve: {e}")


def plot_scientific_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str):
    """
    Generates 'scatter_fit.png'.
    GT vs Prediction with Identity Line.
    """
    try:
        _safe_makedirs(out_dir)
        plt.figure(figsize=(8, 8))

        dmin = float(min(y_true.min(), y_pred.min()))
        dmax = float(max(y_true.max(), y_pred.max()))
        plt.plot([dmin, dmax], [dmin, dmax], "k--", lw=2, label="Identity Line (y=x)")

        plt.scatter(y_true, y_pred, alpha=0.6, s=15, color="blue", label="Predictions")

        plt.xlabel("Ground Truth SpO2")
        plt.ylabel("Model Prediction SpO2")
        plt.title(f"Scientific Scatter Plot (Test)\nMAE: {mean_absolute_error(y_true, y_pred):.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "scatter_fit.png"))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'scatter_fit.png')}")
    except Exception as e:
        print(f"Failed to plot scatter: {e}")


def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str):
    """
    Generates 'bland_altman.png'.
    Mean vs Difference with Bias and Limits of Agreement.
    """
    try:
        _safe_makedirs(out_dir)
        mean_val = (y_true + y_pred) / 2.0
        diff_val = y_pred - y_true  # Pred - Ref

        bias = float(np.mean(diff_val))
        sd = float(np.std(diff_val))
        upper_loa = bias + 1.96 * sd
        lower_loa = bias - 1.96 * sd

        plt.figure(figsize=(10, 6))
        plt.scatter(mean_val, diff_val, alpha=0.5, s=15, color="purple")

        plt.axhline(bias, color="black", linestyle="-", lw=2, label=f"Bias ({bias:.2f})")
        plt.axhline(upper_loa, color="red", linestyle="--", label=f"+1.96 SD ({upper_loa:.2f})")
        plt.axhline(lower_loa, color="red", linestyle="--", label=f"-1.96 SD ({lower_loa:.2f})")
        plt.fill_between([float(mean_val.min()), float(mean_val.max())], lower_loa, upper_loa, color="red", alpha=0.05)

        plt.xlabel("Average of (Predicted + Reference)")
        plt.ylabel("Difference (Predicted - Reference)")
        plt.title("Bland-Altman Plot (Test)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "bland_altman.png"))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'bland_altman.png')}")
    except Exception as e:
        print(f"Failed to plot Bland-Altman: {e}")


def plot_hardware_requirements_box(
    params: int,
    disk_mb: float,
    latency_ms: float,
    est_ram_mb: float,
    out_dir: str,
    filename: str = "hardware_requirements.png",
):
    """
    Creates a figure similar to the sample "HARDWARE REQUIREMENTS (Inference)" box.
    """
    try:
        _safe_makedirs(out_dir)

        lines = [
            "HARDWARE REQUIREMENTS (Inference)",
            "-" * 32,
            f"Parameters      : {params:,}",
            f"Disk Size       : {disk_mb:.2f} MB",
            f"Latency (CPU)   : {latency_ms:.2f} ms/seq",
            "-" * 32,
            f"Est. RAM (B=1)  : ~{est_ram_mb:.2f} MB",
        ]
        text = "\n".join(lines)

        fig = plt.figure(figsize=(8.0, 3.0))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontfamily="monospace",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="black", linewidth=1.5),
        )

        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Failed to plot hardware requirements box: {e}")


# ----------------------------- Dataset -----------------------------
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean=None, std=None, fit_stats: bool = False):
        if fit_stats or (mean is None or std is None):
            mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
            std = X.reshape(-1, X.shape[-1]).std(axis=0)
            std = np.where(std == 0, 1.0, std)
        self.mean = mean
        self.std = std
        self.X = (X - self.mean) / (self.std + 1e-6)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        xi = torch.from_numpy(self.X[idx]).float()
        yi = torch.tensor(self.y[idx]).float()
        return xi, yi


# ----------------------------- Data Loading -----------------------------
def load_npz_file(path: str):
    z = np.load(path, allow_pickle=True)
    X = z["features"].astype(np.float32)  # (N, T, C)
    y = z["labels"].astype(np.float32).reshape(-1)
    return X, y


def load_multiple_npz(files: list, seq_len: int = None):
    """
    Load multiple .npz files and combine them into single arrays.
    Returns: X_all, y_all, target_T, C
    """
    if len(files) == 0:
        raise ValueError("No files provided to load_multiple_npz")
    
    X_list, y_list, T_list, C_list = [], [], [], []
    for fp in files:
        Xi, yi = load_npz_file(fp)
        X_list.append(Xi)
        y_list.append(yi)
        T_list.append(Xi.shape[1])
        C_list.append(Xi.shape[2])
    
    if len(set(C_list)) != 1:
        raise ValueError(f"Inconsistent channels: {C_list}")
    C = C_list[0]
    
    target_T = seq_len if seq_len is not None else min(T_list)
    
    X_all_list, y_all_list = [], []
    for Xi, yi in zip(X_list, y_list):
        Ti = Xi.shape[1]
        if Ti > target_T:
            start = (Ti - target_T) // 2
            Xi_c = Xi[:, start : start + target_T, :]
        elif Ti < target_T:
            pad_len = target_T - Ti
            pad = np.repeat(Xi[:, -1:, :], pad_len, axis=1)
            Xi_c = np.concatenate([Xi, pad], axis=1)
        else:
            Xi_c = Xi
        
        X_all_list.append(Xi_c)
        y_all_list.append(yi)
    
    X_all = np.concatenate(X_all_list, axis=0)
    y_all = np.concatenate(y_all_list, axis=0)
    
    return X_all, y_all, target_T, C


def load_dual_datasets(data_dir: str, seq_len: int = None):
    """
    Load Dataset A (*_sequences.npz) and Dataset B (*.npz excluding sequences).
    Returns: X_A, y_A, X_B, y_B, N_A, N_B, target_T, C
    """
    print(f"\nLoading dual datasets from: {data_dir}")
    
    # Load Dataset A: *_sequences.npz
    print("\n=== Loading Dataset A (*_sequences.npz) ===")
    files_A = sorted(glob.glob(os.path.join(data_dir, "**", "*_sequences.npz"), recursive=True))
    if len(files_A) == 0:
        raise FileNotFoundError(f"No *_sequences.npz files found in {data_dir}")
    print(f"Found {len(files_A)} sequence file(s)")
    
    X_A, y_A, T_A, C_A = load_multiple_npz(files_A, seq_len)
    N_A = len(y_A)
    print(f"Dataset A size: {N_A}")
    
    # Load Dataset B: *.npz (excluding *_sequences.npz)
    print("\n=== Loading Dataset B (*.npz excluding sequences) ===")
    all_npz = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    files_B = [f for f in all_npz if not f.endswith("_sequences.npz")]
    if len(files_B) == 0:
        raise FileNotFoundError(f"No non-sequence .npz files found in {data_dir}")
    print(f"Found {len(files_B)} non-sequence file(s)")
    
    X_B, y_B, T_B, C_B = load_multiple_npz(files_B, seq_len)
    N_B = len(y_B)
    print(f"Dataset B size: {N_B}")
    
    # Verify consistency
    if C_A != C_B:
        raise ValueError(f"Channel mismatch: Dataset A has {C_A} channels, Dataset B has {C_B} channels")
    if T_A != T_B:
        raise ValueError(f"Sequence length mismatch: Dataset A has T={T_A}, Dataset B has T={T_B}")
    
    return X_A, y_A, X_B, y_B, N_A, N_B, T_A, C_A


# ----------------------------- Train/Test Split -----------------------------
def build_dual_dataset_splits(N_A: int, N_B: int, train_ratio: float = 0.30, test_ratio: float = 0.70, seed: int = 42):
    """
    Split Dataset A and Dataset B independently with 30/70 train/test ratio.
    Returns: train_idx_A, test_idx_A, train_idx_B, test_idx_B
    """
    # Split Dataset A
    idx_A_all = np.arange(N_A)
    train_idx_A, test_idx_A = train_test_split(
        idx_A_all, test_size=test_ratio, random_state=seed, shuffle=True
    )
    
    # Split Dataset B
    idx_B_all = np.arange(N_B)
    train_idx_B, test_idx_B = train_test_split(
        idx_B_all, test_size=test_ratio, random_state=seed, shuffle=True
    )
    
    return train_idx_A, test_idx_A, train_idx_B, test_idx_B


# ----------------------------- Model -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class HeavyTransformerRegressor(nn.Module):
    def __init__(
        self,
        in_ch: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_ch, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)


# ----------------------------- Evaluation Helpers -----------------------------
@torch.no_grad()
def eval_epoch_metrics(
    model: nn.Module,
    dl: DataLoader,
    device: str,
    loss_fn: nn.Module,
    acc_tol: float,
    return_preds: bool = False,
):
    """
    Computes epoch-level metrics on a loader:
      - mse, rmse, mae
      - tol-accuracy: mean(|err| <= acc_tol)

    If return_preds=True, also returns concatenated y_true, y_pred arrays.
    """
    model.eval()

    total_sq = 0.0
    total_abs = 0.0
    total_acc = 0.0
    n = 0

    losses = []
    y_true_all = []
    y_pred_all = []

    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        err = pred - yb

        total_sq += float((err ** 2).sum().item())
        total_abs += float(err.abs().sum().item())
        total_acc += float((err.abs() <= acc_tol).sum().item())
        n += int(yb.numel())

        losses.append(float(loss_fn(pred, yb).item()))

        if return_preds:
            y_true_all.append(yb.detach().cpu().numpy().reshape(-1))
            y_pred_all.append(pred.detach().cpu().numpy().reshape(-1))

    mse = total_sq / max(n, 1)
    rmse_v = float(np.sqrt(mse))
    mae_v = total_abs / max(n, 1)
    acc_v = total_acc / max(n, 1)
    loss_mean = float(np.mean(losses)) if len(losses) > 0 else float(mse)

    if return_preds:
        y_true = np.concatenate(y_true_all) if len(y_true_all) else np.array([])
        y_pred = np.concatenate(y_pred_all) if len(y_pred_all) else np.array([])
        return loss_mean, mse, rmse_v, mae_v, acc_v, y_true, y_pred

    return loss_mean, mse, rmse_v, mae_v, acc_v


# ----------------------------- Training (30/70 Dual-Dataset) -----------------------------
def run_train_test_dual(X_A: np.ndarray, y_A: np.ndarray, X_B: np.ndarray, y_B: np.ndarray, N_A: int, N_B: int, T: int, C: int, args, device: str):
    print("\n===== Dual-Dataset Train/Test Split (30/70) =====")
    
    # Split both datasets independently
    train_idx_A, test_idx_A, train_idx_B, test_idx_B = build_dual_dataset_splits(
        N_A=N_A,
        N_B=N_B,
        train_ratio=0.30,
        test_ratio=0.70,
        seed=args.seed
    )
    
    print(f"\n=== Splitting Dataset A ===")
    print(f"Train from A: {len(train_idx_A)} ({len(train_idx_A)/N_A*100:.1f}%)")
    print(f"Test from A:  {len(test_idx_A)} ({len(test_idx_A)/N_A*100:.1f}%)")
    
    print(f"\n=== Splitting Dataset B ===")
    print(f"Train from B: {len(train_idx_B)} ({len(train_idx_B)/N_B*100:.1f}%)")
    print(f"Test from B:  {len(test_idx_B)} ({len(test_idx_B)/N_B*100:.1f}%)")
    
    # Combine datasets
    X_tr = np.concatenate([X_A[train_idx_A], X_B[train_idx_B]], axis=0)
    y_tr = np.concatenate([y_A[train_idx_A], y_B[train_idx_B]], axis=0)
    
    X_te = np.concatenate([X_A[test_idx_A], X_B[test_idx_B]], axis=0)
    y_te = np.concatenate([y_A[test_idx_A], y_B[test_idx_B]], axis=0)
    
    print(f"\n=== Combined Datasets ===")
    print(f"Total train samples: {len(y_tr)} (A: {len(train_idx_A)}, B: {len(train_idx_B)})")
    print(f"Total test samples:  {len(y_te)} (A: {len(test_idx_A)}, B: {len(test_idx_B)})")
    print(f"Sequence length T={T}, Channels C={C}")


    train_ds = SequenceDataset(X_tr, y_tr, fit_stats=True)
    test_ds = SequenceDataset(X_te, y_te, mean=train_ds.mean, std=train_ds.std)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = HeavyTransformerRegressor(
        in_ch=C,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=args.sched_patience)

    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "train_rmse": [],
        "test_rmse": [],
        "train_mae": [],
        "test_mae": [],
        "train_acc": [],
        "test_acc": [],
    }

    best_test = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        # --- Train for 1 epoch (also compute train metrics from the epoch batches) ---
        total_sq = 0.0
        total_abs = 0.0
        total_acc = 0.0
        n = 0
        batch_losses = []

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            err = (pred.detach() - yb)
            total_sq += float((err ** 2).sum().item())
            total_abs += float(err.abs().sum().item())
            total_acc += float((err.abs() <= args.acc_tol).sum().item())
            n += int(yb.numel())

            batch_losses.append(float(loss.item()))

        train_mse = total_sq / max(n, 1)
        train_rmse = float(np.sqrt(train_mse))
        train_mae = total_abs / max(n, 1)
        train_acc = total_acc / max(n, 1)
        train_loss_mean = float(np.mean(batch_losses)) if len(batch_losses) else float(train_mse)

        # --- Evaluate on test split each epoch ---
        test_loss_mean, test_mse, test_rmse, test_mae, test_acc = eval_epoch_metrics(
            model=model,
            dl=test_dl,
            device=device,
            loss_fn=loss_fn,
            acc_tol=args.acc_tol,
            return_preds=False,
        )

        scheduler.step(test_loss_mean)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss(MSE): {train_loss_mean:.4f} | test_loss(MSE): {test_loss_mean:.4f} | "
            f"train_MAE: {train_mae:.3f} | test_MAE: {test_mae:.3f} | "
            f"train_acc: {train_acc:.3f} | test_acc: {test_acc:.3f}"
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss_mean)
        history["test_loss"].append(test_loss_mean)
        history["train_rmse"].append(train_rmse)
        history["test_rmse"].append(test_rmse)
        history["train_mae"].append(train_mae)
        history["test_mae"].append(test_mae)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        # Early stopping / best model tracking (based on test loss)
        if test_loss_mean < best_test - 1e-6:
            best_test = test_loss_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final test predictions (for plots/metrics) ---
    _, _, _, _, _, y_true_test, y_pred_test = eval_epoch_metrics(
        model=model,
        dl=test_dl,
        device=device,
        loss_fn=loss_fn,
        acc_tol=args.acc_tol,
        return_preds=True,
    )

    # Final metrics
    final_mae = mean_absolute_error(y_true_test, y_pred_test) if len(y_true_test) else float("nan")
    final_rmse = rmse(y_true_test, y_pred_test) if len(y_true_test) else float("nan")
    final_r2 = r2_score(y_true_test, y_pred_test) if len(y_true_test) > 1 else float("nan")
    acc1 = within_tolerance(y_true_test, y_pred_test, 1.0) if len(y_true_test) else float("nan")
    acc2 = within_tolerance(y_true_test, y_pred_test, 2.0) if len(y_true_test) else float("nan")
    acc3 = within_tolerance(y_true_test, y_pred_test, 3.0) if len(y_true_test) else float("nan")

    # Efficiency metrics
    params = count_parameters(model)
    model_path = os.path.join(args.out_dir, "tf_best_model.pt")
    size_bytes = save_model_size_bytes(model, model_path)

    # Latency (CPU) - ms/sequence
    # Use a single example from the test set if possible, otherwise from train set.
    sample_np = X_te[0] if len(X_te) else X_tr[0]
    dummy_cpu = torch.from_numpy(sample_np).unsqueeze(0).float().to("cpu")

    model_cpu = model.to("cpu").eval()
    with torch.no_grad():
        _ = model_cpu(dummy_cpu)  # warm-up
        t0 = time.perf_counter()
        for _ in range(args.latency_reps):
            _ = model_cpu(dummy_cpu)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / max(args.latency_reps, 1) * 1000.0

    model_mb = size_bytes / (1024.0 ** 2)
    est_ram_mb = estimate_inference_ram_mb(params=params, bytes_per_param=4, multiplier=args.ram_multiplier)

    summary = {
        "test_MAE": float(final_mae),
        "test_RMSE": float(final_rmse),
        "test_R2": float(final_r2),
        "Acc@1": float(acc1),
        "Acc@2": float(acc2),
        "Acc@3": float(acc3),
        "Params": int(params),
        "Model_MB": float(model_mb),
        "Latency_ms": float(latency_ms),
        "Est_RAM_MB_B1": float(est_ram_mb),
        "model_path": model_path,
    }

    return summary, history, y_true_test, y_pred_test, (params, model_mb, latency_ms, est_ram_mb)


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", type=str, default=r"D:\RESEARCH\Oxygen_Saturation_Estimator", 
                        help="Directory containing .npz files")
    parser.add_argument("--seq_len", type=int, default=None)

    # Split seed
    parser.add_argument("--seed", type=int, default=42)

    # Transformer config
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training config
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sched_patience", type=int, default=8)
    parser.add_argument("--early_stop_patience", type=int, default=30)

    # Metrics/plots
    parser.add_argument("--acc_tol", type=float, default=2.0, help="Tolerance for accuracy curve: |pred-true|<=acc_tol.")
    parser.add_argument("--latency_reps", type=int, default=30)
    parser.add_argument("--ram_multiplier", type=float, default=3.0, help="Multiplier for RAM estimate from parameter bytes.")

    parser.add_argument("--out_dir", type=str, default="./heavy_tf_30_70_dual_dataset")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    set_seed(args.seed)

    print("Loading dual datasets...")
    X_A, y_A, X_B, y_B, N_A, N_B, T, C = load_dual_datasets(data_dir=args.data_dir, seq_len=args.seq_len)
    print(f"Dataset A: {N_A} samples, Dataset B: {N_B} samples")
    print(f"Total samples: {N_A + N_B}, Sequence length T={T}, Channels C={C}")

    # Run train/test with dual datasets
    summary, history, y_true_test, y_pred_test, hw_tuple = run_train_test_dual(X_A, y_A, X_B, y_B, N_A, N_B, T, C, args, device)

    # Save CSV
    df = pd.DataFrame([{
        "split": "30:70",  # Updated to reflect dual-dataset 30/70 split
        "test_MAE": summary["test_MAE"],
        "test_RMSE": summary["test_RMSE"],
        "test_R2": summary["test_R2"],
        "Acc@1": summary["Acc@1"],
        "Acc@2": summary["Acc@2"],
        "Acc@3": summary["Acc@3"],
        "Params": summary["Params"],
        "Model_MB": summary["Model_MB"],
        "Latency_ms": summary["Latency_ms"],
        "Est_RAM_MB_B1": summary["Est_RAM_MB_B1"],
    }])

    csv_path = os.path.join(args.out_dir, "train_test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results CSV to: {csv_path}")

    # Save JSON summary
    json_path = os.path.join(args.out_dir, "heavy_tf_summary.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "dataset_A_samples": int(N_A),
                "dataset_B_samples": int(N_B),
                "total_samples": int(N_A + N_B),
                "train_ratio": 0.30,
                "test_ratio": 0.70,
                "sequence_length_T": int(T),
                "channels_C": int(C),
                "device_used_for_training": device,
                "seed": int(args.seed),
                "acc_tol_for_curve": float(args.acc_tol),
                "transformer_config": {
                    "d_model": int(args.d_model),
                    "nhead": int(args.nhead),
                    "num_layers": int(args.num_layers),
                    "dim_ff": int(args.dim_ff),
                    "dropout": float(args.dropout),
                },
                "training_config": {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "sched_patience": int(args.sched_patience),
                    "early_stop_patience": int(args.early_stop_patience),
                },
                "final_metrics": summary,
            },
            f,
            indent=2,
        )
    print(f"Saved JSON summary to: {json_path}")

    print("\n========== FINAL TEST SUMMARY (70/30) ==========")
    for k in ["test_MAE", "test_RMSE", "test_R2", "Acc@1", "Acc@2", "Acc@3"]:
        print(f"{k:10s}: {summary[k]:.4f}")
    print(f"Params     : {summary['Params']:,}")
    print(f"Model_MB   : {summary['Model_MB']:.3f} MB")
    print(f"Latency_ms : {summary['Latency_ms']:.3f} ms/seq")
    print("================================================\n")

    # Plots
    print("Generating plots...")
    plot_loss_curve(history, args.out_dir)
    plot_accuracy_curve(history, args.out_dir, acc_tol=args.acc_tol)
    plot_error_curve(history, args.out_dir)

    if len(y_true_test) > 0:
        plot_scientific_scatter(y_true_test, y_pred_test, args.out_dir)
        plot_bland_altman(y_true_test, y_pred_test, args.out_dir)

    # Hardware requirements box
    params, model_mb, latency_ms, est_ram_mb = hw_tuple
    plot_hardware_requirements_box(
        params=params,
        disk_mb=model_mb,
        latency_ms=latency_ms,
        est_ram_mb=est_ram_mb,
        out_dir=args.out_dir,
        filename="hardware_requirements.png",
    )

    print("✅ Done. All results and plots saved in:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()