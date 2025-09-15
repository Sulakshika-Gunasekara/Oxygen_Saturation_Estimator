# train.py  —  5-fold grouped CV with regularization, DropPath, and light augmentations
# Works with your *_sequences.npz files (features [N,T,F], labels [N]) in subfolders.

import os, re, glob, math, csv, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold

# ==================== CONFIG ====================
DATA_DIR = r"D:\pure\npz"     # folder that contains your *_sequences.npz (recursively)

# training & regularization
BATCH      = 32
EPOCHS     = 100
LR         = 3e-4
WD         = 2e-4
PATIENCE   = 12
DROP       = 0.18
DROP_PATH  = 0.10
USE_Y_ZSCORE = True
K_FOLDS    = 5

CLIP_NORM = 1.0 

# augmentation (applied to TRAIN ONLY, after feature normalization)
TIME_MASK_P   = 0.30    # prob to apply a time mask on a sample
TIME_MASK_MAX = 20      # max masked frames
FEAT_MASK_P   = 0.30    # prob to apply a feature band mask
FEAT_MASK_MAX = 8       # max masked features
JITTER_STD    = 0.01    # Gaussian noise std
SHIFT_MAX     = 5       # circular time shift (± frames)

# log/plots
LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ================== HELPERS =====================
def parse_subject_id(path: str) -> int:
    """
    Extract a 2-digit subject id from filename or its parent folder.
    Accepts names like '01-06_sequences.npz', 'S01-06_sequences.npz',
    or parent folders like '...\01-06\file.npz'. Returns -1 if not found.
    """
    name = os.path.basename(path)

    # Prefer a 2-digit number at the very start of the file name
    m = re.match(r'^\s*(\d{2})\D', name)
    if m:
        return int(m.group(1))

    # Else, any 2-digit chunk near the start of filename
    m = re.search(r'(\d{2})', name[:12])
    if m:
        return int(m.group(1))

    # Try parent folder
    parent = os.path.basename(os.path.dirname(path))
    m = re.match(r'^\s*(\d{2})\D', parent)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d{2})', parent[:12])
    if m:
        return int(m.group(1))

    return -1


class NPZConcat(Dataset):
    def __init__(self, root_dir: str):
        self.files = sorted(
            glob.glob(os.path.join(root_dir, "**", "*_sequences.npz"), recursive=True)
        )
        if not self.files:
            raise FileNotFoundError(f"No '*_sequences.npz' under '{root_dir}'")

        self.subject_of_file, self.Xs, self.ys, self.index = [], [], [], []
        for fi, p in enumerate(self.files):
            sid = parse_subject_id(p)
            self.subject_of_file.append(sid)
            d = np.load(p)
            X = d["features"]  # [N, T, F]
            y = d["labels"]    # [N]
            self.Xs.append(X); self.ys.append(y)
            for si in range(X.shape[0]):
                self.index.append((fi, si))

        self.subject_of_file = np.array(self.subject_of_file)  # per-file
        self.n_feats = self.Xs[0].shape[2]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, si = self.index[idx]
        x = torch.tensor(self.Xs[fi][si], dtype=torch.float32)   # [T, F]
        y = torch.tensor(self.ys[fi][si], dtype=torch.float32)   # scalar
        return x, y, fi


def collate_pad(batch):
    # batch of (x:[T,F], y, fi)
    B = len(batch)
    lens = [b[0].shape[0] for b in batch]
    Tm = max(lens); F = batch[0][0].shape[1]
    X = torch.zeros(B, Tm, F)
    mask = torch.ones(B, Tm, dtype=torch.bool)  # True=PAD
    y = torch.zeros(B)
    fis = torch.zeros(B, dtype=torch.long)
    for i,(seq, tgt, fi) in enumerate(batch):
        T = seq.shape[0]
        X[i,:T] = seq
        mask[i,:T] = False
        y[i] = tgt
        fis[i] = fi
    return X, y, mask, fis


# --------- Tiny Transformer (DropPath + pre-norm + GLU) ----------
def sinusoid(B, T, D, device):
    pe = torch.zeros(T, D, device=device)
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0)/D))
    pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
    return pe.unsqueeze(0).expand(B, T, D)


class DropPath(nn.Module):
    """Stochastic depth per-sample (Keeps expected activation scale)."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


class TinyBlock(nn.Module):
    def __init__(self, d=48, h=2, drop=0.2, drop_path=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.do1 = nn.Dropout(drop)
        self.dp1 = DropPath(drop_path)

        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, int(1.5*d)*2), nn.GLU(),
            nn.Linear(int(1.5*d), d), nn.Dropout(drop)
        )
        self.dp2 = DropPath(drop_path)

    def forward(self, x, key_padding_mask):
        q = self.n1(x)
        z = self.attn(q, q, q, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + self.dp1(self.do1(z))
        x = x + self.dp2(self.ff(self.n2(x)))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, n_feats, d=48, h=2, L=2, drop=0.2, drop_path=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feats, d)
        self.blocks = nn.ModuleList([TinyBlock(d, h, drop, drop_path) for _ in range(L)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, 1))
        self.d = d

    def forward(self, x, key_padding_mask=None):
        B, T, _ = x.shape
        x = self.proj(x) + sinusoid(B, T, self.d, x.device)
        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).sum(1).clamp(min=1).unsqueeze(-1)
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0).sum(1) / valid
        else:
            x = x.mean(1)
        return self.head(x).squeeze(-1)


# ---------------- metrics / efficiency ---------------
def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(np.float32); y_pred = y_pred.astype(np.float32)
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    yt = y_true - y_true.mean(); yp = y_pred - y_pred.mean()
    r = float((yt*yp).sum() / (np.sqrt((yt**2).sum())*np.sqrt((yp**2).sum()) + 1e-8))
    acc1 = float(np.mean(np.abs(y_pred - y_true) <= 1.0))
    acc2 = float(np.mean(np.abs(y_pred - y_true) <= 2.0))
    return dict(MAE=mae, RMSE=rmse, R=r, Acc1=acc1, Acc2=acc2)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_size_mb(model, tmp="__tmp__.pt"):
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / (1024*1024)
    try: os.remove(tmp)
    except: pass
    return mb


def cpu_latency_ms(model, n_feats, T=150, iters=50):
    device = torch.device("cpu")
    m = model.to(device).eval()
    x = torch.randn(1, T, n_feats)
    mask = torch.zeros(1, T, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(10): m(x, mask)  # warmup
        t0 = time.time()
        for _ in range(iters): m(x, mask)
        t1 = time.time()
    return (t1 - t0)*1000/iters


# ----------- normalization from TRAIN only -----------
@torch.no_grad()
def compute_feature_norm(train_loader, device):
    num = 0.0
    mean = None
    m2   = None
    for X,_,mask,_ in train_loader:
        X = X.to(device); mask = mask.to(device)
        valid = (~mask).unsqueeze(-1)  # [B,T,1]
        Xv = X.masked_select(valid).view(-1, X.shape[-1])  # [N_valid, F]
        if Xv.numel() == 0:
            continue
        if mean is None:
            mean = Xv.mean(dim=0)
            m2   = ((Xv - mean)**2).sum(dim=0)
            num  = Xv.shape[0]
        else:
            num_new = num + Xv.shape[0]
            delta = Xv.mean(dim=0) - mean
            mean = mean + delta * (Xv.shape[0]/num_new)
            m2 = m2 + ((Xv - mean)**2).sum(dim=0) + (delta**2) * (num * Xv.shape[0] / num_new)
            num = num_new
    std = torch.sqrt(m2.clamp_min(1e-12) / max(num-1, 1))
    return mean, std.clamp_min(1e-6)


def normalize_batch(X, mean, std):
    return (X - mean.view(1,1,-1)) / std.view(1,1,-1)


# -------------------- augmentations --------------------
def augment_batch(X, mask,
                  time_mask_p=TIME_MASK_P, time_max=TIME_MASK_MAX,
                  feat_mask_p=FEAT_MASK_P, feat_max=FEAT_MASK_MAX,
                  jitter_std=JITTER_STD, shift_max=SHIFT_MAX):
    """
    X:[B,T,F], mask:[B,T] (True=PAD). Applies small, safe augmentations for sequence regression.
    """
    B,T,F = X.shape

    # (a) tiny Gaussian jitter
    if jitter_std and jitter_std > 0:
        X = X + torch.randn_like(X) * jitter_std

    # (b) small circular time shift (keep mask aligned)
    if shift_max and shift_max > 0:
        shifts = torch.randint(-shift_max, shift_max+1, (B,), device=X.device)
        for i, s in enumerate(shifts.tolist()):
            if s != 0:
                X[i]    = torch.roll(X[i], shifts=s, dims=0)
                mask[i] = torch.roll(mask[i], shifts=s, dims=0)

    # (c) time masking (zero out a short window)
    if time_mask_p and time_mask_p > 0:
        for i in range(B):
            if torch.rand(1, device=X.device) < time_mask_p:
                w = int(torch.randint(1, min(time_max, T)+1, (1,), device=X.device))
                t0 = int(torch.randint(0, max(T - w + 1, 1), (1,), device=X.device))
                X[i, t0:t0+w] = 0.0

    # (d) feature masking (zero out a small band of features)
    if feat_mask_p and feat_mask_p > 0:
        for i in range(B):
            if torch.rand(1, device=X.device) < feat_mask_p:
                w = int(torch.randint(1, min(feat_max, F)+1, (1,), device=X.device))
                f0 = int(torch.randint(0, max(F - w + 1, 1), (1,), device=X.device))
                X[i, :, f0:f0+w] = 0.0

    return X, mask


# ---------------- training one fold ------------------
def train_one_fold(model, train_loader, val_loader, device, y_mu=None, y_std=None, fold_id=0):
    crit = nn.L1Loss()
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.3, patience=4)
    best = 1e9; wait=0; best_metrics=None

    # Pre-compute feature normalization on TRAIN set
    feat_mu, feat_std = compute_feature_norm(train_loader, device)

    hist = {"epoch":[], "train_loss":[], "val_mae":[], "val_rmse":[], "val_r":[], "val_acc1":[], "val_acc2":[]}

    for ep in range(1, EPOCHS+1):
        # ---- train ----
        model.train(); tr_losses=[]
        for X,y,mask,_ in train_loader:
            X,y,mask = X.to(device), y.to(device), mask.to(device)
            X = normalize_batch(X, feat_mu, feat_std)
            # augment only during training
            X, mask = augment_batch(X, mask)

            y_n = (y - y_mu)/y_std if USE_Y_ZSCORE else y
            opt.zero_grad(set_to_none=True)
            pred = model(X, mask)
            loss = crit((pred - y_mu)/y_std, y_n) if USE_Y_ZSCORE else crit(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        # ---- validate ---- (report in REAL units)
        model.eval(); vt, vp = [], []
        with torch.no_grad():
            for X,y,mask,_ in val_loader:
                X,y,mask = X.to(device), y.to(device), mask.to(device)
                X = normalize_batch(X, feat_mu, feat_std)
                out = model(X, mask)
                vt.append(y.cpu().numpy()); vp.append(out.cpu().numpy())
        y_true = np.concatenate(vt); y_pred = np.concatenate(vp)
        m = compute_metrics(y_true, y_pred)
        sch.step(m["RMSE"]**2)

        hist["epoch"].append(ep)
        hist["train_loss"].append(float(np.mean(tr_losses)))
        hist["val_mae"].append(m["MAE"])
        hist["val_rmse"].append(m["RMSE"])
        hist["val_r"].append(m["R"])
        hist["val_acc1"].append(m["Acc1"])
        hist["val_acc2"].append(m["Acc2"])

        print(f"[F{fold_id}] Ep{ep:03d} TrainMAE={np.mean(tr_losses):.4f} | "
              f"Val MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} R={m['R']:.3f} "
              f"Acc±1={m['Acc1']*100:.1f}% Acc±2={m['Acc2']*100:.1f}%")

        if m["RMSE"] < best - 1e-6:
            best = m["RMSE"]; wait=0; best_metrics = m.copy()
            torch.save(model.state_dict(), f"{LOG_DIR}/best_fold{fold_id}.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # save history CSV and plot per-fold curves
    df = pd.DataFrame(hist)
    csv_path = os.path.join(LOG_DIR, f"history_F{fold_id}.csv")
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train L1 (z-scored)" if USE_Y_ZSCORE else "Train L1")
    plt.plot(df["epoch"], df["val_rmse"],  label="Val RMSE (real)")
    plt.xlabel("Epoch"); plt.ylabel("Loss / RMSE"); plt.title(f"Learning Curves - Fold {fold_id}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    fig_path = os.path.join(LOG_DIR, f"curves_F{fold_id}.png")
    plt.savefig(fig_path, dpi=160); plt.close()
    print(f"[plot] saved {fig_path}")

    return best_metrics



# ----------------------- 5-fold CV ------------------------
def run_cv5(root_dir=DATA_DIR, batch=BATCH, k_folds=K_FOLDS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NPZConcat(root_dir)

    # Map each sample to a subject via file index
    file_subjects = ds.subject_of_file                # per-file
    sample_subjects = np.array([file_subjects[fi] for (fi,_) in ds.index], dtype=int)

    # Print per-subject counts
    file_counts = Counter(file_subjects.tolist())
    seq_counts  = Counter(sample_subjects.tolist())
    present = sorted([int(s) for s in set(sample_subjects) if s != -1])

    print("Detected subjects:", present)
    print("\nPer-subject counts (files -> sequences):")
    # for s in sorted([x for x in file_subjects if x != -1]):
    #     print(f"  S{s:02d}: files={file_counts[s]}  sequences={seq_counts[s]}")

    for s in sorted({int(x) for x in file_subjects.tolist() if x != -1}):
        print(f"  S{s:02d}: files={file_counts[s]}  sequences={seq_counts[s]}")


    # Prepare CV splitter with groups=subjects (no leakage)
    gkf = GroupKFold(n_splits=k_folds)

    rows = []
    fold_id = 0
    for train_idx, val_idx in gkf.split(np.arange(len(ds)), groups=sample_subjects):
        fold_id += 1
        print(f"\n=== {k_folds}-fold CV: Fold {fold_id} ({len(train_idx)} train seq, {len(val_idx)} val seq) ===")

        # y stats on TRAIN ONLY (for optional z-scoring)
        y_train_all = []
        for idx in train_idx:
            fi, si = ds.index[idx]
            y_train_all.append(ds.ys[fi][si])
        y_train_all = np.array(y_train_all, dtype=np.float32)
        y_mu = float(y_train_all.mean()); y_std = float(y_train_all.std() + 1e-6)

        train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch, shuffle=True,  collate_fn=collate_pad)
        val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch, shuffle=False, collate_fn=collate_pad)

        # model (keep it tiny & deployable)
        model = TinyTransformer(n_feats=ds.n_feats, d=48, h=2, L=2, drop=DROP, drop_path=DROP_PATH).to(device)

        # efficiency (same architecture every fold)
        params = count_params(model)
        sizeMB = state_dict_size_mb(model)
        lat_ms = cpu_latency_ms(model, n_feats=ds.n_feats, T=150, iters=50)

        best = train_one_fold(model, train_loader, val_loader, device, y_mu=y_mu, y_std=y_std, fold_id=fold_id)

        row = {
            "Fold": fold_id,
            "MAE": best["MAE"], "RMSE": best["RMSE"], "R": best["R"],
            "Acc±1": best["Acc1"], "Acc±2": best["Acc2"],
            "Params": params, "ModelMB": sizeMB, "CPUms": lat_ms
        }
        rows.append(row)
        print(f"[Fold {fold_id}]  MAE={row['MAE']:.3f}  RMSE={row['RMSE']:.3f}  R={row['R']:.3f}  "
              f"Acc±1={row['Acc±1']*100:.1f}%  Acc±2={row['Acc±2']*100:.1f}%  | "
              f"Params={params}  Size={sizeMB:.2f}MB  CPU~{lat_ms:.1f}ms")

    # summary
    keys = ["MAE","RMSE","R","Acc±1","Acc±2","Params","ModelMB","CPUms"]
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    print("\n==== 5-fold Summary (mean ± std) ====")
    for k in ["MAE","RMSE","R","Acc±1","Acc±2"]:
        print(f"{k:6s}: {arr[k].mean():.4f} ± {arr[k].std():.4f}")
    print(f"Params (mean): {arr['Params'].mean():.0f}")
    print(f"Model size MB (mean): {arr['ModelMB'].mean():.2f}")
    print(f"CPU latency ms/seq (mean): {arr['CPUms'].mean():.1f}")

    # CSV
    out_csv = os.path.join(root_dir, "cv5_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Fold"]+keys)
        w.writeheader()
        for r in rows: w.writerow(r)
        w.writerow({"Fold":"SUMMARY_MEAN", **{k: float(arr[k].mean()) for k in keys}})
        w.writerow({"Fold":"SUMMARY_STD",  **{k: float(arr[k].std())  for k in keys}})
    print(f"\nSaved per-fold + summary to: {out_csv}")

    # plots: per-fold best RMSE curve and bars of metrics
    # (1) val RMSE bars
    plt.figure()
    folds = [r["Fold"] for r in rows]
    rmses = [r["RMSE"] for r in rows]
    plt.bar(folds, rmses)
    plt.xlabel("Fold"); plt.ylabel("Best Val RMSE"); plt.title("CV5: Best Val RMSE per Fold")
    plt.tight_layout()
    p1 = os.path.join(LOG_DIR, "curves_cv5_valrmse.png")
    plt.savefig(p1, dpi=160); plt.close()
    print(f"[plot] saved {p1}")

    # (2) summary metrics bars (MAE & RMSE)
    plt.figure()
    width = 0.35
    x = np.arange(len(folds))
    plt.bar(x - width/2, [r["MAE"] for r in rows], width, label="MAE")
    plt.bar(x + width/2, [r["RMSE"] for r in rows], width, label="RMSE")
    plt.xticks(x, folds)
    plt.legend(); plt.title("CV5: Fold Metrics")
    plt.xlabel("Fold"); plt.ylabel("Score")
    plt.tight_layout()
    p2 = os.path.join(LOG_DIR, "cv5_metrics.png")
    plt.savefig(p2, dpi=160); plt.close()
    print(f"[plot] saved {p2}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    run_cv5(DATA_DIR)
