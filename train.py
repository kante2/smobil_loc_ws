"""
Model, shared helpers, AND training for HLOS-Rwgh-OWLS positioning.

This single file contains BOTH the algorithm (formerly model.py) and the
LOS-classifier training routine. main.py imports the algorithm symbols
(LOSClassifier, FEATURE_DIM, predict_position) from here.

Run training:  python train.py   -> saves model.pkl

Algorithm short name : HLOS-Rwgh-OWLS
                       (Hybrid LOS-classifier + Residual-Weighted-subset +
                        NLOS-aware One-sided robust WLS)

Pipeline (no absolute coordinates used as features, for generalization):
  step 1 : closed-form linear LS warm start, clipped to room box
  step 2 : per-BS feature vector -> small MLP -> P(LOS_i)
  step 3 : take top-K BSes by P(LOS_i), enumerate k-subsets, solve NLS in each
  step 4 : keep the single subset with the lowest normalized residual
           (Rwgh ranking) as the mid estimate
  step 5 : final NLOS-AWARE refinement over all 18 BSes, warm started from
           step 4. Each BS contributes:
             - LOS-like (high P(LOS)) : two-sided P(LOS)^gamma-weighted term
             - NLOS-like (low P(LOS)) : ONE-SIDED hinge that penalizes only an
               overshoot beyond the measured distance. Because an NLOS RTT is
               inflated, the true distance is <= the measurement, so a measured
               value is a valid UPPER BOUND. This recovers geometric information
               from NLOS BSes that simple down-weighting throws away -> it helps
               most for users with few LOS BSes (the error tail).
           A soft_l1 robust loss caps the leverage of any single residual so a
           misclassified BS cannot dominate the fit.

Reference papers:
  [Chen99]    P.-C. Chen, "A non-line-of-sight error mitigation algorithm
              in location estimation," IEEE WCNC 1999.  (Rwgh residual ranking)
  [Breg18]    K. Bregar & M. Mohorcic, "Improving Indoor Localization Using
              Convolutional Neural Networks on Computationally Restricted
              Devices," IEEE Access 2018.  (learned NLOS identification)
  [Guvenc09]  I. Guvenc & C.-C. Chong, "A Survey on TOA Based Wireless
              Localization and NLOS Mitigation Techniques," IEEE Comm. Surveys
              2009.  (NLOS measurement as a one-sided / inequality constraint)
"""
from itertools import combinations

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Geometry helpers
# ============================================================
ROOM_MARGIN = 30.0


def room_box(p_bs):
    return (p_bs[0].min() - ROOM_MARGIN, p_bs[0].max() + ROOM_MARGIN,
            p_bs[1].min() - ROOM_MARGIN, p_bs[1].max() + ROOM_MARGIN)


def clip_to_room(p, box):
    return np.array([np.clip(p[0], box[0], box[1]),
                     np.clip(p[1], box[2], box[3])])


def linear_ls_init(d, p_bs, ref=None):
    """
    Closed-form 2D linear LS by reference-BS differencing.
    Picks the BS with the smallest measurement as reference (most likely LOS).
    """
    M = p_bs.shape[1]
    if ref is None:
        ref = int(np.argmin(d))
    idx = [i for i in range(M) if i != ref]
    A = -2.0 * (p_bs[:, idx] - p_bs[:, ref:ref + 1]).T
    b = (d[idx] ** 2 - d[ref] ** 2
         - np.sum(p_bs[:, idx] ** 2, axis=0)
         + np.sum(p_bs[:, ref] ** 2))
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def nonlinear_wls(p0, d, p_bs, w=None, max_nfev=80):
    """Plain (two-sided) weighted NLS, used inside the subset solves (fast LM)."""
    if w is None:
        w = np.ones(p_bs.shape[1])
    sw = np.sqrt(np.maximum(w, 0.0))
    def res(p):
        return sw * (np.linalg.norm(p[:, None] - p_bs, axis=0) - d)
    return least_squares(res, p0, method='lm', max_nfev=max_nfev).x


def nonlinear_wls_onesided(p0, d, p_bs, p_los,
                           w_pow=12.0, nlos_w=1.0, f_scale=3.0, max_nfev=150):
    """
    Final NLOS-aware refinement over all BSes.

    Gap for BS i:  g_i = ||p - bs_i|| - d_i   (predicted minus measured)
      LOS-like  (high P(LOS)) : two-sided term  P(LOS)_i^w_pow * g_i
                                -> must match the measurement on both sides.
      NLOS-like (low  P(LOS)) : one-sided hinge (1-P(LOS))_i * nlos_w * max(g_i,0)
                                -> NLOS RTT is inflated, so true distance <= d_i;
                                   only an overshoot is penalized, an undershoot
                                   is consistent with NLOS bias and stays free.
    soft_l1 robust loss caps the influence of any single (possibly mislabeled)
    residual.
    """
    w_los = np.maximum(p_los ** w_pow, 1e-3)
    w_nlos = (1.0 - p_los) * nlos_w

    def res(p):
        g = np.linalg.norm(p[:, None] - p_bs, axis=0) - d
        return np.concatenate([w_los * g, w_nlos * np.maximum(g, 0.0)])

    sol = least_squares(res, p0, method='trf', loss='soft_l1',
                        f_scale=f_scale, max_nfev=max_nfev)
    return sol.x


# ============================================================
# Neural network: per-BS LOS classifier
# ============================================================
FEATURE_DIM = 9


class LOSClassifier(nn.Module):
    """Outputs logit for P(LOS) per BS."""
    def __init__(self, in_dim=FEATURE_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_features(d, p_bs, p0):
    """
    Per-BS feature vector.

    All features are RELATIVE — no absolute UE or BS coordinates.
    This is deliberate: hidden test users have unknown positions but the
    relative geometry of measurements vs. warm-start is still meaningful.

    Returns (M, FEATURE_DIM) float32.
    """
    M = p_bs.shape[1]
    d_pred = np.linalg.norm(p0[:, None] - p_bs, axis=0)
    resid = d - d_pred
    abs_r = np.abs(resid)

    med_r = np.median(resid)
    mad_r = np.median(np.abs(resid - med_r)) + 1e-6
    z = (resid - med_r) / (1.4826 * mad_r)  # robust z-score

    # rank of measurement (0 = smallest, 1 = largest)
    order = np.argsort(d)
    rank = np.empty(M)
    rank[order] = np.arange(M) / (M - 1)

    feats = np.stack([
        d,                          # 1. raw RTT
        d_pred,                     # 2. geometric prediction from warm start
        resid,                      # 3. signed residual
        abs_r,                      # 4. |residual|
        z,                          # 5. robust z-score
        rank,                       # 6. rank among BSes for this user
        np.full(M, med_r),          # 7. global median residual
        np.full(M, mad_r),          # 8. global MAD
        d - d.min(),                # 9. excess over smallest measurement
    ], axis=1).astype(np.float32)
    return feats


# ============================================================
# Final prediction function
# ============================================================
LOS_BIAS_THRESHOLD = 3.0   # used as label threshold during training


def predict_position(d, p_bs, model,
                     top_k=8, k_subset=4, gamma_r=2.0, gamma_w=12.0, nlos_w=1.0):
    """
    Locate one user.

    d     : (M,)   RTT measurements
    p_bs  : (2, M) BS coordinates
    model : trained LOSClassifier (eval mode)

    Returns (2,) estimated position.
    """
    M = p_bs.shape[1]
    box = room_box(p_bs)

    # ---- 1. warm start ----
    p0 = clip_to_room(linear_ls_init(d, p_bs), box)

    # ---- 2. LOS probabilities ----
    feats = extract_features(d, p_bs, p0)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats)).numpy()
    p_los = 1.0 / (1.0 + np.exp(-logits))

    # ---- 3. candidate pool from top-K LOS scores ----
    K = min(top_k, M)
    cand = np.argsort(-p_los)[:K]

    # ---- 4. Rwgh: enumerate k-subsets, keep the lowest-residual one ----
    best_p, best_score = None, -np.inf
    for combo in combinations(cand, k_subset):
        idx = np.array(combo)
        p0_loc = clip_to_room(linear_ls_init(d[idx], p_bs[:, idx]), box)
        p_sub = clip_to_room(nonlinear_wls(p0_loc, d[idx], p_bs[:, idx]), box)
        d_pred = np.linalg.norm(p_sub[:, None] - p_bs[:, idx], axis=0)
        r = np.linalg.norm(d_pred - d[idx]) / np.sqrt(k_subset - 2 + 1e-9)
        score = 1.0 / (r ** gamma_r + 1e-6)
        if score > best_score:
            best_score, best_p = score, p_sub
    p_mid = clip_to_room(best_p, box)

    # ---- 5. final NLOS-aware one-sided robust refinement over all BSes ----
    p_hat = nonlinear_wls_onesided(p_mid, d, p_bs, p_los,
                                   w_pow=gamma_w, nlos_w=nlos_w)
    return clip_to_room(p_hat, box)


# ============================================================
# Training: LOS classifier  (run `python train.py`)
# ============================================================
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    # class imbalance handling (LOS is the minority class)
    pos_weight = torch.tensor([(1 - ytr.mean()) / (ytr.mean() + 1e-6)],
                              dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    EPOCHS = 200
    best_val = float('inf')
    patience, wait = 30, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        scheduler.step()

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