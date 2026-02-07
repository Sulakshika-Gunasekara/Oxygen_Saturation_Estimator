#!/usr/bin/env python3
# heavy_cnn2d_cv3_with_extra_graphs.py
# Same as your FIRST model (heavy_cnn2d_cv3.py) but with extra validation + hardware graphs added.

import os
import glob
import json
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------- Utilities -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def within_tolerance(y_true, y_pred, tol):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_size_bytes(model, path):
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)


# ----------------------------- Model -----------------------------
class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dropout=0.0):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        if self.proj is not None:
            identity = self.proj(identity)

        out = F.relu(out + identity)
        return out


class HeavyCNN2D_v2(nn.Module):
    """
    Heavier 2D CNN for regression with residual blocks and a stronger head.

    Input:  (B, 1, T, C)
    Output: (B,)
    """
    def __init__(self, base=48, dropout=0.10, pool_out=4, head_hidden=256):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, kernel_size=(7, 3), padding=(3, 1), bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Stages (downsample with maxpool)
        self.stage1 = nn.Sequential(
            ResidualBlock2D(base, base, k=3, dropout=dropout),
            ResidualBlock2D(base, base, k=3, dropout=dropout),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.stage2 = nn.Sequential(
            ResidualBlock2D(base, base * 2, k=3, dropout=dropout),
            ResidualBlock2D(base * 2, base * 2, k=3, dropout=dropout),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.stage3 = nn.Sequential(
            ResidualBlock2D(base * 2, base * 4, k=3, dropout=dropout),
            ResidualBlock2D(base * 4, base * 4, k=3, dropout=dropout),
        )

        # Keep more information than GAP(1,1)
        self.pool_out = pool_out
        self.agg = nn.AdaptiveAvgPool2d((pool_out, pool_out))

        # Stronger MLP head
        feat_dim = (base * 4) * pool_out * pool_out
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, x):
        x = self.stem(x)         # (B, base, T, C)
        x = self.stage1(x)
        x = self.pool1(x)        # downsample
        x = self.stage2(x)
        x = self.pool2(x)        # downsample
        x = self.stage3(x)

        x = self.agg(x)          # (B, base*4, pool_out, pool_out)
        x = x.flatten(1)
        x = self.head(x).squeeze(-1)
        return x


# ----------------------------- Dataset -----------------------------
class SequenceDataset2D(Dataset):
    """
    Takes X shaped (N, T, C) and returns tensors shaped (1, T, C).
    Normalization: per-channel over all samples and timesteps.
    Optional y normalization (fit on train, reuse on val).
    """
    def __init__(self, X, y, x_mean=None, x_std=None, fit_x_stats=False,
                 y_mean=None, y_std=None, fit_y_stats=False, use_y_norm=True):
        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape(-1)

        # X stats
        if fit_x_stats or (x_mean is None or x_std is None):
            x_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
            x_std = X.reshape(-1, X.shape[-1]).std(axis=0)
            x_std = np.where(x_std == 0, 1.0, x_std)

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.X = (X - self.x_mean) / (self.x_std + 1e-6)

        # y stats
        self.use_y_norm = bool(use_y_norm)
        if self.use_y_norm:
            if fit_y_stats or (y_mean is None or y_std is None):
                y_mean = float(y.mean())
                y_std = float(y.std())
                if y_std == 0:
                    y_std = 1.0
            self.y_mean = float(y_mean)
            self.y_std = float(y_std)
            self.y = ((y - self.y_mean) / (self.y_std + 1e-6)).astype(np.float32)
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xi = torch.from_numpy(self.X[idx]).float().unsqueeze(0)  # (1, T, C)
        yi = torch.tensor(self.y[idx]).float()
        return xi, yi


# ----------------------------- Data loading (Dual-Dataset) -----------------------------
def load_npz_file(path):
    z = np.load(path, allow_pickle=True)
    X = z["features"].astype(np.float32)  # (N, T, C)
    y = z["labels"].astype(np.float32).reshape(-1)
    return X, y


def load_dual_datasets(data_dir, seq_len=None):
    """
    Load two datasets:
    - Dataset A: *_sequences.npz files
    - Dataset B: *.npz files (excluding *_sequences.npz)
    Returns combined data with proper alignment.
    """
    from sklearn.model_selection import train_test_split
    
    # Load Dataset A: *_sequences.npz
    print("\n=== Loading Dataset A (*_sequences.npz) ===")
    pattern_A = "*_sequences.npz"
    files_A = sorted(glob.glob(os.path.join(data_dir, "**", pattern_A), recursive=True))
    
    if not files_A:
        raise FileNotFoundError(f"No '{pattern_A}' files found under '{data_dir}'")
    
    X_A_list, y_A_list, T_list, C_list = [], [], [], []
    for fp in files_A:
        Xi, yi = load_npz_file(fp)
        X_A_list.append(Xi)
        y_A_list.append(yi)
        T_list.append(Xi.shape[1])
        C_list.append(Xi.shape[2])
    
    # Load Dataset B: *.npz (excluding *_sequences.npz)
    print("\n=== Loading Dataset B (*.npz excluding sequences) ===")
    all_npz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    files_B = [f for f in all_npz_files if not f.endswith("_sequences.npz")]
    
    if not files_B:
        raise FileNotFoundError(f"No non-sequence .npz files found in '{data_dir}'")
    
    X_B_list, y_B_list = [], []
    for fp in files_B:
        Xi, yi = load_npz_file(fp)
        X_B_list.append(Xi)
        y_B_list.append(yi)
        T_list.append(Xi.shape[1])
        C_list.append(Xi.shape[2])
    
    if len(set(C_list)) != 1:
        raise ValueError(f"Inconsistent channels across files: {C_list}")
    C = C_list[0]
    
    target_T = seq_len if seq_len is not None else min(T_list)
    
    # Process and combine datasets
    def process_dataset(X_list, y_list, target_T):
        X_proc_list, y_proc_list = [], []
        for Xi, yi in zip(X_list, y_list):
            Ti = Xi.shape[1]
            if Ti > target_T:
                start = (Ti - target_T) // 2
                Xi_c = Xi[:, start:start + target_T, :]
            elif Ti < target_T:
                pad_len = target_T - Ti
                pad = np.repeat(Xi[:, -1:, :], pad_len, axis=1)
                Xi_c = np.concatenate([Xi, pad], axis=1)
            else:
                Xi_c = Xi
            X_proc_list.append(Xi_c)
            y_proc_list.append(yi)
        return X_proc_list, y_proc_list
    
    X_A_proc, y_A_proc = process_dataset(X_A_list, y_A_list, target_T)
    X_B_proc, y_B_proc = process_dataset(X_B_list, y_B_list, target_T)
    
    X_A = np.concatenate(X_A_proc, axis=0)
    y_A = np.concatenate(y_A_proc, axis=0)
    X_B = np.concatenate(X_B_proc, axis=0)
    y_B = np.concatenate(y_B_proc, axis=0)
    
    N_A, N_B = len(y_A), len(y_B)
    print(f"Dataset A size: {N_A}")
    print(f"Dataset B size: {N_B}")
    
    return X_A, y_A, X_B, y_B, target_T, C, N_A, N_B


# ----------------------------- Dual-Dataset Splits (15/15/70) -----------------------------
def build_dual_dataset_splits(X_A, y_A, X_B, y_B, train_ratio=0.15, val_ratio=0.15, test_ratio=0.70):
    """
    Split both datasets A and B into train/val/test with specified ratios.
    Returns combined train, val, test data.
    """
    from sklearn.model_selection import train_test_split
    
    N_A, N_B = len(y_A), len(y_B)
    
    # Split Dataset A
    print(f"\n=== Splitting Dataset A ===")
    idx_A_all = np.arange(N_A)
    train_val_idx_A, test_idx_A = train_test_split(
        idx_A_all, test_size=test_ratio, random_state=42, shuffle=True
    )
    train_idx_A, val_idx_A = train_test_split(
        train_val_idx_A, train_size=0.5, random_state=42, shuffle=True
    )
    print(f"Train from A: {len(train_idx_A)} ({len(train_idx_A)/N_A*100:.1f}%)")
    print(f"Val from A:   {len(val_idx_A)} ({len(val_idx_A)/N_A*100:.1f}%)")
    print(f"Test from A:  {len(test_idx_A)} ({len(test_idx_A)/N_A*100:.1f}%)")
    
    # Split Dataset B
    print(f"\n=== Splitting Dataset B ===")
    idx_B_all = np.arange(N_B)
    train_val_idx_B, test_idx_B = train_test_split(
        idx_B_all, test_size=test_ratio, random_state=42, shuffle=True
    )
    train_idx_B, val_idx_B = train_test_split(
        train_val_idx_B, train_size=0.5, random_state=42, shuffle=True
    )
    print(f"Train from B: {len(train_idx_B)} ({len(train_idx_B)/N_B*100:.1f}%)")
    print(f"Val from B:   {len(val_idx_B)} ({len(val_idx_B)/N_B*100:.1f}%)")
    print(f"Test from B:  {len(test_idx_B)} ({len(test_idx_B)/N_B*100:.1f}%)")
    
    # Combine datasets
    X_train = np.concatenate([X_A[train_idx_A], X_B[train_idx_B]], axis=0)
    y_train = np.concatenate([y_A[train_idx_A], y_B[train_idx_B]], axis=0)
    
    X_val = np.concatenate([X_A[val_idx_A], X_B[val_idx_B]], axis=0)
    y_val = np.concatenate([y_A[val_idx_A], y_B[val_idx_B]], axis=0)
    
    X_test = np.concatenate([X_A[test_idx_A], X_B[test_idx_B]], axis=0)
    y_test = np.concatenate([y_A[test_idx_A], y_B[test_idx_B]], axis=0)
    
    print(f"\n=== Combined Datasets ===")
    print(f"Total train samples: {len(y_train)} (A: {len(train_idx_A)}, B: {len(train_idx_B)})")
    print(f"Total val samples:   {len(y_val)} (A: {len(val_idx_A)}, B: {len(val_idx_B)})")
    print(f"Total test samples:  {len(y_test)} (A: {len(test_idx_A)}, B: {len(test_idx_B)})")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ----------------------------- Loss picker -----------------------------
def make_loss(name: str, huber_beta: float = 1.0):
    name = name.lower().strip()
    if name in ("mse", "mseloss"):
        return nn.MSELoss()
    if name in ("huber", "smoothl1", "smoothl1loss"):
        return nn.SmoothL1Loss(beta=huber_beta)
    if name in ("mae", "l1", "l1loss"):
        return nn.L1Loss()
    raise ValueError(f"Unknown loss: {name}")


# ----------------------------- Training per fold -----------------------------
def run_fold(fold, train_idx, val_idx, X, y, T, C, args, device):
    print(f"\n===== Fold {fold} =====")
    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    # Fit X and y stats on train; reuse on val
    train_ds = SequenceDataset2D(
        X_tr, y_tr,
        fit_x_stats=True,
        fit_y_stats=True,
        use_y_norm=args.y_norm
    )
    val_ds = SequenceDataset2D(
        X_va, y_va,
        x_mean=train_ds.x_mean, x_std=train_ds.x_std, fit_x_stats=False,
        y_mean=train_ds.y_mean, y_std=train_ds.y_std, fit_y_stats=False,
        use_y_norm=args.y_norm
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = HeavyCNN2D_v2(
        base=args.base,
        dropout=args.dropout,
        pool_out=args.pool_out,
        head_hidden=args.head_hidden
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = make_loss(args.loss, huber_beta=args.huber_beta)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.sched_patience
    )

    history = []
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))

        # Convert normalized loss into RMSE in original units if y_norm enabled
        if args.y_norm:
            va_rmse_orig = float(np.sqrt(max(va_loss, 0.0)) * train_ds.y_std)
        else:
            va_rmse_orig = float(np.sqrt(max(va_loss, 0.0)))

        scheduler.step(va_loss)

        print(f"Fold {fold} | Epoch {epoch}/{args.epochs} "
              f"| train loss: {tr_loss:.5f} | val RMSE (orig): {va_rmse_orig:.5f}")

        history.append({
            "fold": fold,
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_rmse": va_rmse_orig,
        })

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping on fold {fold} at epoch {epoch} "
                  f"(no improvement for {args.early_stop_patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Out-of-fold predictions (DENORMALIZE back to original units if needed)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            pred_n = model(xb.to(device)).cpu().numpy().reshape(-1)
            yb_n = yb.cpu().numpy().reshape(-1)

            if args.y_norm:
                pred = pred_n * train_ds.y_std + train_ds.y_mean
                yt = yb_n * train_ds.y_std + train_ds.y_mean
            else:
                pred = pred_n
                yt = yb_n

            y_true.append(yt)
            y_pred.append(pred)

    y_true = np.concatenate(y_true).reshape(-1).astype(np.float32)
    y_pred = np.concatenate(y_pred).reshape(-1).astype(np.float32)

    # Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    r_mse = rmse(y_true, y_pred)
    acc1 = within_tolerance(y_true, y_pred, 1.0)
    acc2 = within_tolerance(y_true, y_pred, 2.0)
    acc3 = within_tolerance(y_true, y_pred, 3.0)

    params = count_parameters(model)
    model_path = os.path.join(args.out_dir, f"fold{fold}_model.pt")
    size_bytes = save_model_size_bytes(model, model_path)

    # CPU latency estimate (single sample)
    dummy_input, _ = val_ds[0]            # (1, T, C)
    dummy = dummy_input.unsqueeze(0).to("cpu")  # (1, 1, T, C)

    model_cpu = model.to("cpu")
    with torch.no_grad():
        _ = model_cpu(dummy)  # warmup
        t0 = time.perf_counter()
        for _ in range(args.latency_reps):
            _ = model_cpu(dummy)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / args.latency_reps * 1000.0

    summary = {
        "fold": fold,
        "val_MAE": mae,
        "val_RMSE": r_mse,
        "val_R2": r2,
        "Acc@1": acc1,
        "Acc@2": acc2,
        "Acc@3": acc3,
        "Params": params,
        "Model_MB": size_bytes / (1024 ** 2),
        "Latency_ms": latency_ms,
        "epochs_ran": int(history[-1]["epoch"]) if history else 0
    }

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.out_dir, f"fold{fold}_history.csv"), index=False)

    print(f"Fold {fold} summary:", summary)
    return summary, hist_df, y_true, y_pred, val_idx


# ----------------------------- Plotting -----------------------------
def plot_training_stability(all_histories, out_dir):
    if not all_histories:
        return

    max_epoch = int(max(h["epoch"].max() for h in all_histories))
    folds = len(all_histories)

    train_mat = np.full((folds, max_epoch), np.nan, dtype=np.float32)
    valrmse_mat = np.full((folds, max_epoch), np.nan, dtype=np.float32)

    for i, h in enumerate(all_histories):
        for _, row in h.iterrows():
            e = int(row["epoch"]) - 1
            train_mat[i, e] = float(row["train_loss"])
            valrmse_mat[i, e] = float(row["val_rmse"])

    epochs = np.arange(1, max_epoch + 1)

    tr_mean = np.nanmean(train_mat, axis=0)
    tr_std = np.nanstd(train_mat, axis=0)

    va_mean = np.nanmean(valrmse_mat, axis=0)
    va_std = np.nanstd(valrmse_mat, axis=0)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, tr_mean, label="Train Loss")
    plt.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)

    plt.plot(epochs, va_mean, label="Validation RMSE (orig units)")
    plt.fill_between(epochs, va_mean - va_std, va_mean + va_std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Stability (mean ± std across {folds} folds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epoch_loss_summary.png"), dpi=200)
    plt.close()


def plot_scatter_fit(y_true, y_pred, out_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    if len(y_true) > 0:
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)

    plt.xlabel("Reference SpO2 (Ground Truth)")
    plt.ylabel("Predicted SpO2")
    plt.title("Scatter Fit: Prediction vs Reference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_fit.png"), dpi=200)
    plt.close()


def plot_pred_vs_true_hexbin(y_true, y_pred, out_dir):
    plt.figure(figsize=(6.5, 6))
    plt.hexbin(y_true, y_pred, gridsize=35, mincnt=1)
    if len(y_true) > 0:
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title("Pred vs True (Hexbin density)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_vs_true_density.png"), dpi=200)
    plt.close()


def plot_bland_altman(y_true, y_pred, out_dir):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    avg = (y_true + y_pred) / 2.0
    diff = (y_pred - y_true)

    bias = float(np.mean(diff)) if len(diff) > 0 else 0.0
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    loa_hi = bias + 1.96 * sd
    loa_lo = bias - 1.96 * sd

    plt.figure(figsize=(7, 5))
    plt.scatter(avg, diff, alpha=0.6)

    plt.axhline(bias, color="k", linewidth=1.5, label=f"Bias = {bias:.2f}")
    plt.axhline(loa_hi, color="r", linestyle="--", linewidth=1.5, label=f"+1.96σ = {loa_hi:.2f}")
    plt.axhline(loa_lo, color="r", linestyle="--", linewidth=1.5, label=f"-1.96σ = {loa_lo:.2f}")

    plt.xlabel("Average of (Predicted + Reference)")
    plt.ylabel("Difference (Predicted - Reference)")
    plt.title("Bland–Altman Plot")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bland_altman.png"), dpi=200)
    plt.close()


def plot_residual_hist(y_true, y_pred, out_dir):
    err = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    plt.figure(figsize=(7, 4.5))
    plt.hist(err, bins=40)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_hist.png"), dpi=200)
    plt.close()


def plot_residuals_vs_true(y_true, y_pred, out_dir):
    y_true = np.asarray(y_true, dtype=np.float32)
    err = np.asarray(y_pred, dtype=np.float32) - y_true
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, err, alpha=0.6)
    plt.axhline(0.0, color="k", linewidth=1.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residuals vs Ground Truth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuals_vs_true.png"), dpi=200)
    plt.close()


def plot_abs_error_vs_true(y_true, y_pred, out_dir):
    y_true = np.asarray(y_true, dtype=np.float32)
    abs_err = np.abs(np.asarray(y_pred, dtype=np.float32) - y_true)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, abs_err, alpha=0.6)
    plt.xlabel("Ground Truth")
    plt.ylabel("|Error|")
    plt.title("Absolute Error vs Ground Truth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "abs_error_vs_true.png"), dpi=200)
    plt.close()


def plot_hardware_requirements(df_summary: pd.DataFrame, out_dir: str):
    fold_labels = [f"F{int(f)}" for f in df_summary["fold"].tolist()]

    # Per-fold: individual bar plots
    for col, title, fname in [
        ("Params", "Trainable Parameters (per fold)", "hardware_params.png"),
        ("Model_MB", "Model Size (MB, per fold)", "hardware_model_mb.png"),
        ("Latency_ms", "CPU Latency (ms, per fold)", "hardware_latency_ms.png"),
    ]:
        plt.figure(figsize=(7.5, 4.5))
        plt.bar(fold_labels, df_summary[col].astype(float).values)
        plt.xlabel("Fold")
        plt.ylabel(col)
        plt.title(title)
        plt.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    # Combined (normalized so all three can be shown together)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(fold_labels))
    width = 0.25

    params = df_summary["Params"].astype(float).values
    size_mb = df_summary["Model_MB"].astype(float).values
    lat_ms = df_summary["Latency_ms"].astype(float).values

    def _norm(a):
        a = np.asarray(a, dtype=np.float64)
        denom = (a.max() - a.min()) if a.max() != a.min() else 1.0
        return (a - a.min()) / denom

    plt.bar(x - width, _norm(params), width, label="Params (norm)")
    plt.bar(x, _norm(size_mb), width, label="Model_MB (norm)")
    plt.bar(x + width, _norm(lat_ms), width, label="Latency_ms (norm)")
    plt.xticks(x, fold_labels)
    plt.ylabel("Normalized value")
    plt.title("Hardware Requirements (normalized, per fold)")
    plt.grid(True, alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hardware_requirements.png"), dpi=200)
    plt.close()

    # Mean ± std summary
    agg = df_summary[["Params", "Model_MB", "Latency_ms"]].agg(["mean", "std"])
    means = agg.loc["mean"].values.astype(float)
    stds = agg.loc["std"].values.astype(float)
    labels = ["Params", "Model_MB", "Latency_ms"]

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.title("Hardware Summary (mean ± std across folds)")
    plt.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hardware_summary.png"), dpi=200)
    plt.close()


# ----------------------------- Main (Dual-Dataset 15/15/70) -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Data loading (updated for dual-dataset)
    parser.add_argument("--data_dir", type=str, default=r"D:\RESEARCH\Oxygen_Saturation_Estimator",
                        help="Directory containing both *_sequences.npz and *.npz files")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.70)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)  # Increased from 1e-5 for stronger L2 regularization
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Model (reduced capacity to prevent overfitting)
    parser.add_argument("--base", type=int, default=40)  # Reduced from 48
    parser.add_argument("--dropout", type=float, default=0.25)  # Increased from 0.10 for stronger regularization
    parser.add_argument("--pool_out", type=int, default=4)
    parser.add_argument("--head_hidden", type=int, default=192)  # Reduced from 256

    # Loss / scaling
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber", "mae"])
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--y_norm", action="store_true", default=True,
                        help="Normalize targets using train mean/std (recommended).")

    # Scheduler / early stop
    parser.add_argument("--sched_patience", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=25)  # Reduced from 35 to stop earlier

    # Runtime / outputs
    parser.add_argument("--latency_reps", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="./cnn2d_15_15_70_dual_dataset")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dual datasets from: {args.data_dir}")
    X_A, y_A, X_B, y_B, T, C, N_A, N_B = load_dual_datasets(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
    )
    
    # Build dual-dataset splits
    X_train, y_train, X_val, y_val, X_test, y_test = build_dual_dataset_splits(
        X_A, y_A, X_B, y_B,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    print(f"Sequence length T={T}, Channels C={C}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    set_seed(42)

    # Train single model on train/val split
    print("\n===== Training 2D CNN Model =====")
    
    # Fit X and y stats on train; reuse on val/test
    train_ds = SequenceDataset2D(
        X_train, y_train,
        fit_x_stats=True,
        fit_y_stats=True,
        use_y_norm=args.y_norm
    )
    val_ds = SequenceDataset2D(
        X_val, y_val,
        x_mean=train_ds.x_mean, x_std=train_ds.x_std, fit_x_stats=False,
        y_mean=train_ds.y_mean, y_std=train_ds.y_std, fit_y_stats=False,
        use_y_norm=args.y_norm
    )
    test_ds = SequenceDataset2D(
        X_test, y_test,
        x_mean=train_ds.x_mean, x_std=train_ds.x_std, fit_x_stats=False,
        y_mean=train_ds.y_mean, y_std=train_ds.y_std, fit_y_stats=False,
        use_y_norm=args.y_norm
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = HeavyCNN2D_v2(
        base=args.base,
        dropout=args.dropout,
        pool_out=args.pool_out,
        head_hidden=args.head_hidden
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = make_loss(args.loss, huber_beta=args.huber_beta)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.sched_patience
    )

    history = []
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))

        # Convert normalized loss into RMSE in original units if y_norm enabled
        if args.y_norm:
            va_rmse_orig = float(np.sqrt(max(va_loss, 0.0)) * train_ds.y_std)
        else:
            va_rmse_orig = float(np.sqrt(max(va_loss, 0.0)))

        scheduler.step(va_loss)

        print(f"Epoch {epoch}/{args.epochs} | train loss: {tr_loss:.5f} | val RMSE (orig): {va_rmse_orig:.5f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_rmse": va_rmse_orig,
        })

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    params = count_parameters(model)
    model_path = os.path.join(args.out_dir, "best_model.pt")
    size_bytes = save_model_size_bytes(model, model_path)

    # CPU latency estimate
    dummy_input, _ = val_ds[0]
    dummy = dummy_input.unsqueeze(0).to("cpu")

    model_cpu = model.to("cpu")
    with torch.no_grad():
        _ = model_cpu(dummy)
        t0 = time.perf_counter()
        for _ in range(args.latency_reps):
            _ = model_cpu(dummy)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / args.latency_reps * 1000.0

    # Evaluate on train/val/test
    def evaluate_split(model, dataloader, device, train_ds, args):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in dataloader:
                pred_n = model(xb.to(device)).cpu().numpy().reshape(-1)
                yb_n = yb.cpu().numpy().reshape(-1)

                if args.y_norm:
                    pred = pred_n * train_ds.y_std + train_ds.y_mean
                    yt = yb_n * train_ds.y_std + train_ds.y_mean
                else:
                    pred = pred_n
                    yt = yb_n

                y_true.append(yt)
                y_pred.append(pred)

        y_true = np.concatenate(y_true).reshape(-1).astype(np.float32)
        y_pred = np.concatenate(y_pred).reshape(-1).astype(np.float32)

        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
        r_mse = rmse(y_true, y_pred)
        acc1 = within_tolerance(y_true, y_pred, 1.0)
        acc2 = within_tolerance(y_true, y_pred, 2.0)
        acc3 = within_tolerance(y_true, y_pred, 3.0)
        
        return {
            "MAE": mae,
            "RMSE": r_mse,
            "R2": r2,
            "Acc@1": acc1,
            "Acc@2": acc2,
            "Acc@3": acc3,
        }, y_true, y_pred

    model_gpu = model.to(device)
    train_metrics, train_true, train_pred = evaluate_split(model_gpu, train_dl, device, train_ds, args)
    val_metrics, val_true, val_pred = evaluate_split(model_gpu, val_dl, device, train_ds, args)
    test_metrics, test_true, test_pred = evaluate_split(model_gpu, test_dl, device, train_ds, args)

    # Save results
    summary = {
        "train_MAE": train_metrics["MAE"],
        "val_MAE": val_metrics["MAE"],
        "test_MAE": test_metrics["MAE"],
        "train_RMSE": train_metrics["RMSE"],
        "val_RMSE": val_metrics["RMSE"],
        "test_RMSE": test_metrics["RMSE"],
        "Params": params,
        "Model_MB": size_bytes / (1024 ** 2),
        "Latency_ms": latency_ms,
        "N_A": N_A,
        "N_B": N_B,
        "Train_total": len(y_train),
        "Val_total": len(y_val),
        "Test_total": len(y_test),
        "epochs_ran": int(history[-1]["epoch"]) if history else 0
    }

    # Save summary CSV
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(args.out_dir, "results_15_15_70_dual_dataset.csv"), index=False)

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

    # Plots
    # Training history
    if len(history) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train Loss")
        plt.plot([h["epoch"] for h in history], [h["val_rmse"] for h in history], label="Val RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "epoch_loss_summary.png"), dpi=200)
        plt.close()

    # Scatter plots
    plot_scatter_fit(test_true, test_pred, args.out_dir)
    plot_pred_vs_true_hexbin(test_true, test_pred, args.out_dir)
    plot_bland_altman(test_true, test_pred, args.out_dir)
    plot_residual_hist(test_true, test_pred, args.out_dir)
    plot_residuals_vs_true(test_true, test_pred, args.out_dir)
    plot_abs_error_vs_true(test_true, test_pred, args.out_dir)

    print("\n========== FINAL RESULTS (2D CNN Dual-Dataset 15/15/70) ==========")
    print(f"Dataset A: {N_A} samples, Dataset B: {N_B} samples")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"\nTrain MAE: {train_metrics['MAE']:.4f}, RMSE: {train_metrics['RMSE']:.4f}")
    print(f"Val   MAE: {val_metrics['MAE']:.4f}, RMSE: {val_metrics['RMSE']:.4f}")
    print(f"Test  MAE: {test_metrics['MAE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}")
    print(f"\nModel: {params:,} params, {size_bytes/(1024**2):.2f} MB, {latency_ms:.2f} ms/seq (CPU)")
    print("==================================================================\n")

    print("✅ Done. Results in:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
