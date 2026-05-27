"""
tune.py — ~/smobil_loc_ws/ 에서 실행
  python3 tune.py

3가지 튜닝:
  1) Feature 추가 (9 → 13)
  2) Model capacity 증가 + dropout
  3) predict_position 하이퍼파라미터 grid search
"""
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations, product
from scipy.optimize import least_squares
import time

# ============================================================
# 1. UPGRADED FEATURES (9 → 13)
# ============================================================
FEATURE_DIM_V2 = 13


def extract_features_v2(d, p_bs, p0):
    """
    기존 9개 + 4개 추가:
      10. d / median(d)          — 상대적 RTT 크기
      11. residual skewness      — 전체 residual 분포 비대칭도
      12. 해당 BS까지 거리 순위 기반 이웃 BS와의 residual 차이
      13. d^2 - d_pred^2         — 비선형 residual (NLOS bias 감지에 유리)
    """
    M = p_bs.shape[1]
    d_pred = np.linalg.norm(p0[:, None] - p_bs, axis=0)
    resid = d - d_pred
    abs_r = np.abs(resid)

    med_r = np.median(resid)
    mad_r = np.median(np.abs(resid - med_r)) + 1e-6
    z = (resid - med_r) / (1.4826 * mad_r)

    order = np.argsort(d)
    rank = np.empty(M)
    rank[order] = np.arange(M) / (M - 1)

    # --- new features ---
    d_ratio = d / (np.median(d) + 1e-6)                          # 10
    skew = np.full(M, ((resid - resid.mean())**3).mean()
                   / (resid.std()**3 + 1e-9))                    # 11

    # neighbor residual diff: for each BS, diff with closest BS by d
    sorted_idx = np.argsort(d)
    neighbor_diff = np.zeros(M)
    for i in range(M):
        pos = np.where(sorted_idx == i)[0][0]
        if pos > 0:
            neighbor_diff[i] = abs_r[i] - abs_r[sorted_idx[pos - 1]]
        else:
            neighbor_diff[i] = abs_r[i] - abs_r[sorted_idx[1]]   # 12

    sq_resid = d**2 - d_pred**2                                   # 13

    feats = np.stack([
        d, d_pred, resid, abs_r, z, rank,
        np.full(M, med_r), np.full(M, mad_r),
        d - d.min(),
        d_ratio, skew, neighbor_diff, sq_resid,
    ], axis=1).astype(np.float32)
    return feats


# ============================================================
# 2. UPGRADED MODEL
# ============================================================
class LOSClassifierV2(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM_V2, hidden=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Geometry helpers (from model.py, duplicated to keep standalone)
# ============================================================
ROOM_MARGIN = 30.0
LOS_BIAS_THRESHOLD = 2.0


def room_box(p_bs):
    return (p_bs[0].min() - ROOM_MARGIN, p_bs[0].max() + ROOM_MARGIN,
            p_bs[1].min() - ROOM_MARGIN, p_bs[1].max() + ROOM_MARGIN)


def clip_to_room(p, box):
    return np.array([np.clip(p[0], box[0], box[1]),
                     np.clip(p[1], box[2], box[3])])


def linear_ls_init(d, p_bs, ref=None):
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
    if w is None:
        w = np.ones(p_bs.shape[1])
    sw = np.sqrt(np.maximum(w, 0.0))
    def res(p):
        return sw * (np.linalg.norm(p[:, None] - p_bs, axis=0) - d)
    return least_squares(res, p0, method='lm', max_nfev=max_nfev).x


# ============================================================
# 3. TRAINING
# ============================================================
def build_dataset_v2(d_hat, p_bs, p_gt):
    num_user = d_hat.shape[1]
    box = room_box(p_bs)
    Xs, ys = [], []
    for u in range(num_user):
        d = d_hat[:, u]
        p0 = clip_to_room(linear_ls_init(d, p_bs), box)
        feats = extract_features_v2(d, p_bs, p0)
        true_d = np.linalg.norm(p_gt[:, u:u + 1] - p_bs, axis=0)
        bias = d - true_d
        label = (bias < LOS_BIAS_THRESHOLD).astype(np.float32)
        Xs.append(feats)
        ys.append(label)
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def train_model(X, y, num_user, M, hidden=128, dropout=0.15,
                epochs=200, lr=1e-3, patience=30):
    rng = np.random.default_rng(42)
    perm = rng.permutation(num_user)
    n_val = num_user // 5
    is_val = np.zeros(num_user, dtype=bool)
    is_val[perm[:n_val]] = True
    sample_user = np.repeat(np.arange(num_user), M)
    val_mask = is_val[sample_user]

    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xv,  yv  = X[val_mask],  y[val_mask]

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xv_t  = torch.from_numpy(Xv)
    yv_t  = torch.from_numpy(yv)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                        batch_size=512, shuffle=True)

    torch.manual_seed(0)
    model = LOSClassifierV2(in_dim=X.shape[1], hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_weight = torch.tensor([(1 - ytr.mean()) / (ytr.mean() + 1e-6)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float('inf')
    best_state = None
    wait = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv_t), yv_t).item()
            logits = model(Xv_t).numpy()
            pred = (logits > 0).astype(np.float32)
            acc = (pred == yv).mean()

            # precision / recall
            tp = ((pred == 1) & (yv == 1)).sum()
            fp = ((pred == 1) & (yv == 0)).sum()
            fn = ((pred == 0) & (yv == 1)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop ep {ep+1}")
                break

        if (ep + 1) % 20 == 0:
            print(f"  Ep {ep+1:3d}  val_loss={val_loss:.4f}  acc={acc:.3f}"
                  f"  prec={prec:.3f}  rec={rec:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val BCE = {best_val:.4f}")
    return model, best_val


# ============================================================
# 4. PREDICTION (with tunable hyperparams)
# ============================================================
def predict_position_v2(d, p_bs, model, feat_fn,
                        top_k=8, k_subset=4, gamma_r=2.0, gamma_w=4.0):
    M = p_bs.shape[1]
    box = room_box(p_bs)
    p0 = clip_to_room(linear_ls_init(d, p_bs), box)

    feats = feat_fn(d, p_bs, p0)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats)).numpy()
    p_los = 1.0 / (1.0 + np.exp(-logits))

    K = min(top_k, M)
    cand = np.argsort(-p_los)[:K]

    estimates, inv_res = [], []
    for combo in combinations(cand, k_subset):
        idx = np.array(combo)
        p0_loc = clip_to_room(linear_ls_init(d[idx], p_bs[:, idx]), box)
        p_sub = nonlinear_wls(p0_loc, d[idx], p_bs[:, idx])
        p_sub = clip_to_room(p_sub, box)
        d_pred = np.linalg.norm(p_sub[:, None] - p_bs[:, idx], axis=0)
        r = np.linalg.norm(d_pred - d[idx]) / np.sqrt(k_subset - 2 + 1e-9)
        estimates.append(p_sub)
        inv_res.append(1.0 / (r ** gamma_r + 1e-6))
    estimates = np.stack(estimates)
    inv_res = np.asarray(inv_res)

    cutoff = np.percentile(inv_res, 50)
    mask = inv_res >= cutoff
    w_sub = inv_res[mask] / inv_res[mask].sum()
    p_mid = (estimates[mask] * w_sub[:, None]).sum(axis=0)
    p_mid = clip_to_room(p_mid, box)

    w_final = np.maximum(p_los ** gamma_w, 1e-3)
    p_hat = nonlinear_wls(p_mid, d, p_bs, w_final)
    return clip_to_room(p_hat, box)


# ============================================================
# 5. EVALUATION
# ============================================================
def evaluate(d_hat, p_bs, p_gt, model, feat_fn,
             top_k=8, k_subset=4, gamma_r=2.0, gamma_w=4.0):
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = predict_position_v2(
            d_hat[:, u], p_bs, model, feat_fn,
            top_k=top_k, k_subset=k_subset,
            gamma_r=gamma_r, gamma_w=gamma_w)
    err = np.linalg.norm(p_hat - p_gt, axis=0)
    return err


def print_metrics(err, label):
    print(f"  {label:30s}  mean={err.mean():.3f}  med={np.median(err):.3f}"
          f"  90p={np.percentile(err, 90):.3f}  max={err.max():.1f}")


# ============================================================
# MAIN
# ============================================================
def main():
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    p_bs  = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p_gt  = np.asarray(data['p'], dtype=float)
    num_user = d_hat.shape[1]
    M = p_bs.shape[1]

    # =============================================
    # STEP A: Train upgraded classifier
    # =============================================
    print("=" * 60)
    print("STEP A: Training V2 classifier (13 features, bigger model)")
    print("=" * 60)
    X, y = build_dataset_v2(d_hat, p_bs, p_gt)
    print(f"X={X.shape}  LOS ratio={y.mean():.3f}")

    model, _ = train_model(X, y, num_user, M,
                           hidden=128, dropout=0.15,
                           epochs=200, lr=1e-3, patience=30)

    # =============================================
    # STEP B: Hyperparameter grid search
    # =============================================
    print("\n" + "=" * 60)
    print("STEP B: Grid search (top_k, k_subset, gamma_r, gamma_w)")
    print("=" * 60)

    param_grid = {
        'top_k':    [6, 8, 10],
        'k_subset': [4, 5, 6],
        'gamma_r':  [1.5, 2.0, 3.0],
        'gamma_w':  [3.0, 4.0, 5.0, 6.0],
    }

    # Use val users only for grid search (prevent overfitting to train users)
    rng = np.random.default_rng(42)
    perm = rng.permutation(num_user)
    n_val = num_user // 5
    val_users = perm[:n_val]

    d_val = d_hat[:, val_users]
    p_val = p_gt[:, val_users]

    best_mean = float('inf')
    best_params = {}
    results = []

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    print(f"Total combos: {len(combos)}")

    t0 = time.time()
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        err = evaluate(d_val, p_bs, p_val, model, extract_features_v2, **params)
        mean_err = err.mean()
        results.append((mean_err, params))

        if mean_err < best_mean:
            best_mean = mean_err
            best_params = params.copy()

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(combos)}]  best so far: {best_mean:.4f} m"
                  f"  ({elapsed:.0f}s)")

    print(f"\n  Best params: {best_params}")
    print(f"  Best val mean error: {best_mean:.4f} m")

    # =============================================
    # STEP C: Full evaluation with best params
    # =============================================
    print("\n" + "=" * 60)
    print("STEP C: Full evaluation (all 700 users)")
    print("=" * 60)

    err_full = evaluate(d_hat, p_bs, p_gt, model, extract_features_v2,
                        **best_params)
    print(f"\n  HLOS-Rwgh-WLS V2:")
    print(f"  Mean Error  : {err_full.mean():.4f} m")
    print(f"  Median Error: {np.median(err_full):.4f} m")
    print(f"  RMSE        : {np.sqrt((err_full**2).mean()):.4f} m")
    print(f"  90th %%ile   : {np.percentile(err_full, 90):.4f} m")
    print(f"  95th %%ile   : {np.percentile(err_full, 95):.4f} m")
    print(f"  Max          : {err_full.max():.4f} m")
    print(f"  < 1 m       : {(err_full < 1).mean()*100:.1f}%")
    print(f"  < 2 m       : {(err_full < 2).mean()*100:.1f}%")
    print(f"  < 5 m       : {(err_full < 5).mean()*100:.1f}%")

    # Compare to V1
    print(f"\n  vs V1 (mean 3.7711):  {(1 - err_full.mean()/3.7711)*100:+.1f}%")

    # =============================================
    # STEP D: Save best model + params
    # =============================================
    torch.save(model.state_dict(), 'model_v2.pkl')
    print(f"\n  Saved model_v2.pkl")
    print(f"  Best hyperparams to use in model.py:")
    for k, v in best_params.items():
        print(f"    {k} = {v}")

    # Top 5 combos
    results.sort(key=lambda x: x[0])
    print(f"\n  Top 5 param combos:")
    for mean_e, p in results[:5]:
        print(f"    mean={mean_e:.4f}  {p}")


if __name__ == "__main__":
    main()
