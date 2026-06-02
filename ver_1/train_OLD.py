"""
Train the LOS classifier and save to model.pkl.

Run:  python train.py

For each (user, BS) pair we have:
  - feature vector x_i (built from RTT + warm-start residual geometry)
  - binary label y_i  =  1 if true bias < 2 m else 0  (i.e. LOS-like vs NLOS)

We train a small MLP with BCE (+pos_weight to handle class imbalance,
LOS ratio ~ 34 %). User-level 80:20 split prevents leakage between BS
rows of the same user. The best validation-loss weights are saved.
"""
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import (
    LOSClassifier,
    FEATURE_DIM,
    LOS_BIAS_THRESHOLD,
    linear_ls_init,
    extract_features,
    clip_to_room,
    room_box,
)


def build_dataset(d_hat, p_bs, p_gt):
    """For each (user, BS) build a feature row and a LOS/NLOS label."""
    num_user = d_hat.shape[1]
    box = room_box(p_bs)
    Xs, ys = [], []
    for u in range(num_user):
        d = d_hat[:, u]
        p0 = clip_to_room(linear_ls_init(d, p_bs), box)
        feats = extract_features(d, p_bs, p0)                       # (18, F)
        true_d = np.linalg.norm(p_gt[:, u:u + 1] - p_bs, axis=0)    # (18,)
        bias = d - true_d
        label = (bias < LOS_BIAS_THRESHOLD).astype(np.float32)
        Xs.append(feats)
        ys.append(label)
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def main():
    # ---- 1. load training data ----
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    p_bs  = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p_gt  = np.asarray(data['p'], dtype=float)
    num_user = d_hat.shape[1]
    print(f"Users: {num_user}  BSes: {p_bs.shape[1]}")

    # ---- 2. features + labels ----
    print("Building features...")
    X, y = build_dataset(d_hat, p_bs, p_gt)
    print(f"X={X.shape}  y={y.shape}  LOS ratio = {y.mean():.3f}")

    # ---- 3. user-level train/val split ----
    rng = np.random.default_rng(42)
    perm = rng.permutation(num_user)
    n_val = num_user // 5
    is_val = np.zeros(num_user, dtype=bool)
    is_val[perm[:n_val]] = True
    sample_user = np.repeat(np.arange(num_user), p_bs.shape[1])
    val_mask = is_val[sample_user]

    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xv,  yv  = X[val_mask],  y[val_mask]
    print(f"Train: {len(Xtr)}  Val: {len(Xv)}  "
          f"(train LOS ratio = {ytr.mean():.3f})")

    # ---- 4. tensors ----
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv_t,  yv_t  = torch.from_numpy(Xv),  torch.from_numpy(yv)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                        batch_size=512, shuffle=True)

    # ---- 5. model + optim ----
    torch.manual_seed(0)
    model = LOSClassifier(in_dim=FEATURE_DIM, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # class imbalance handling (LOS is the minority class)
    pos_weight = torch.tensor([(1 - ytr.mean()) / (ytr.mean() + 1e-6)],
                              dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    EPOCHS = 200 # 100 -> 200
    best_val = float('inf')
    patience, wait = 30, 0 # 20, 0 -> 30, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv_t), yv_t).item()
            # accuracy at p>0.5
            logits = model(Xv_t).numpy()
            pred = (logits > 0).astype(np.float32)
            acc = (pred == yv).mean()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'model.pkl')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep+1}")
                break

        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1:3d}  val_loss={val_loss:.4f}  acc={acc:.3f}  "
                  f"best={best_val:.4f}")

    print(f"\nDone. best val BCE = {best_val:.4f}. Saved model.pkl")


if __name__ == "__main__":
    main()
