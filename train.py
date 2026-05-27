"""
Training script. Run `python train.py` to produce model.pkl.

Procedure:
  - For each (user, BS) pair, build a feature vector and the TRUE NLOS bias label
    (true_bias = d_hat - ||p_gt - p_bs||).
  - Train BiasMLP with SmoothL1 loss to predict that bias.
  - Save best-validation weights to model.pkl.
"""
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import (
    BiasMLP, FEATURE_DIM,
    linear_ls_init, extract_features,
)


def build_dataset(d_hat, p_bs, p_gt):
    """
    For each (user, BS), build features and the true NLOS-bias label.
    Returns X: (N*18, F), y: (N*18,)  — both float32.
    """
    num_user = d_hat.shape[1]
    Xs, ys = [], []
    for u in range(num_user):
        p0 = linear_ls_init(d_hat[:, u], p_bs)
        feats = extract_features(d_hat[:, u], p_bs, p0)         # (18, F)
        true_d = np.linalg.norm(p_gt[:, u:u + 1] - p_bs, axis=0)  # (18,)
        true_bias = (d_hat[:, u] - true_d).astype(np.float32)
        Xs.append(feats)
        ys.append(true_bias)
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def main():
    # ---- 1. load training data ----
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    p_bs = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p_gt = np.asarray(data['p'], dtype=float)
    num_user = d_hat.shape[1]
    print(f"Users: {num_user}  BSes: {p_bs.shape[1]}")

    # ---- 2. build (X, y) ----
    print("Building features...")
    X, y = build_dataset(d_hat, p_bs, p_gt)
    print(f"X={X.shape}  y={y.shape}  bias mean={y.mean():.2f} std={y.std():.2f}")

    # ---- 3. user-level train/val split (avoid leakage between BS rows of same user) ----
    rng = np.random.default_rng(42)
    perm = rng.permutation(num_user)
    n_val = num_user // 5
    val_users = np.zeros(num_user, dtype=bool)
    val_users[perm[:n_val]] = True
    sample_user = np.repeat(np.arange(num_user), 18)
    val_mask = val_users[sample_user]
    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xv, yv = X[val_mask], y[val_mask]
    print(f"Train: {len(Xtr)}  Val: {len(Xv)}")

    # ---- 4. tensors ----
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv_t, yv_t = torch.from_numpy(Xv), torch.from_numpy(yv)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                        batch_size=256, shuffle=True)

    # ---- 5. model + optim ----
    model = BiasMLP(in_dim=FEATURE_DIM, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # SmoothL1 is robust to outliers in the bias label (some users have extreme NLOS)
    loss_fn = nn.SmoothL1Loss()

    EPOCHS = 100
    best_val = float('inf')
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv_t), yv_t).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'model.pkl')
        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1:3d}  val_loss={val_loss:.3f}  best={best_val:.3f}")

    print(f"\nDone. best val SmoothL1 = {best_val:.3f}. Saved model.pkl")


if __name__ == "__main__":
    main()