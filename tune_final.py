"""
tune_final.py — 남은 파라미터 전부 탐색
  python3 tune_final.py

Part 1: prediction params (retrain 불필요, 빠름)
  - gamma_w: 5,6,7,8,10
  - subset cutoff percentile: 30,40,50,60,70

Part 2: training params (retrain 필요, 좀 걸림)
  - LOS_BIAS_THRESHOLD: 1.0, 1.5, 2.0, 2.5, 3.0
  - epochs 200 + cosine LR
"""
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations

from model import (
    LOSClassifier, FEATURE_DIM,
    linear_ls_init, extract_features,
    clip_to_room, room_box, nonlinear_wls,
)


# ---- prediction with tunable cutoff ----
def predict_tuned(d, p_bs, model,
                  top_k=8, k_subset=4, gamma_r=2.0, gamma_w=6.0,
                  cutoff_pct=50):
    M = p_bs.shape[1]
    box = room_box(p_bs)
    p0 = clip_to_room(linear_ls_init(d, p_bs), box)

    feats = extract_features(d, p_bs, p0)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats)).numpy()
    p_los = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

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

    cutoff = np.percentile(inv_res, cutoff_pct)
    mask = inv_res >= cutoff
    w_sub = inv_res[mask] / inv_res[mask].sum()
    p_mid = (estimates[mask] * w_sub[:, None]).sum(axis=0)
    p_mid = clip_to_room(p_mid, box)

    w_final = np.maximum(p_los ** gamma_w, 1e-3)
    p_hat = nonlinear_wls(p_mid, d, p_bs, w_final)
    return clip_to_room(p_hat, box)


def eval_all(d_hat, p_bs, p_gt, model, **kwargs):
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = predict_tuned(d_hat[:, u], p_bs, model, **kwargs)
    err = np.linalg.norm(p_hat - p_gt, axis=0)
    return err


def fmt(err):
    return (f"mean={err.mean():.3f}  med={np.median(err):.3f}  "
            f"90p={np.percentile(err,90):.2f}  "
            f"<2m={100*(err<2).mean():.1f}%  <5m={100*(err<5).mean():.1f}%")


# ---- retrain with different threshold ----
def build_dataset_threshold(d_hat, p_bs, p_gt, threshold):
    num_user = d_hat.shape[1]
    box = room_box(p_bs)
    Xs, ys = [], []
    for u in range(num_user):
        d = d_hat[:, u]
        p0 = clip_to_room(linear_ls_init(d, p_bs), box)
        feats = extract_features(d, p_bs, p0)
        true_d = np.linalg.norm(p_gt[:, u:u+1] - p_bs, axis=0)
        bias = d - true_d
        label = (bias < threshold).astype(np.float32)
        Xs.append(feats)
        ys.append(label)
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def train_with_config(X, y, num_user, M, epochs=200, lr=1e-3, hidden=64,
                      use_cosine=True):
    rng = np.random.default_rng(42)
    perm = rng.permutation(num_user)
    n_val = num_user // 5
    is_val = np.zeros(num_user, dtype=bool)
    is_val[perm[:n_val]] = True
    sample_user = np.repeat(np.arange(num_user), M)
    val_mask = is_val[sample_user]

    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xv, yv = X[val_mask], y[val_mask]

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xv_t = torch.from_numpy(Xv)
    yv_t = torch.from_numpy(yv)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=512, shuffle=True)

    torch.manual_seed(0)
    model = LOSClassifier(in_dim=X.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_weight = torch.tensor([(1 - ytr.mean()) / (ytr.mean() + 1e-6)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float('inf')
    best_state = None
    patience, wait = 30, 0

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if use_cosine:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv_t), yv_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def main():
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    p_bs = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p_gt = np.asarray(data['p'], dtype=float)
    num_user = d_hat.shape[1]
    M = p_bs.shape[1]

    # Load current best model
    model = LOSClassifier(in_dim=FEATURE_DIM)
    state = torch.load('model.pkl', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ========================================
    # PART 1: gamma_w × cutoff_pct (no retrain)
    # ========================================
    print("=" * 65)
    print("PART 1: gamma_w × cutoff_pct (기존 model.pkl)")
    print("=" * 65)

    best_mean = 999
    best_cfg = {}
    for gw in [5.0, 6.0, 7.0, 8.0, 10.0]:
        for cp in [30, 40, 50, 60, 70]:
            err = eval_all(d_hat, p_bs, p_gt, model, gamma_w=gw, cutoff_pct=cp)
            tag = f"gw={gw:.0f} cut={cp}"
            m = err.mean()
            if m < best_mean:
                best_mean = m
                best_cfg = {'gamma_w': gw, 'cutoff_pct': cp}
            print(f"  {tag:15s}  {fmt(err)}")

    print(f"\n  >>> PART 1 best: {best_cfg}  mean={best_mean:.4f}")

    # ========================================
    # PART 2: LOS threshold retrain + best gamma_w
    # ========================================
    print("\n" + "=" * 65)
    print("PART 2: LOS_BIAS_THRESHOLD retrain (epochs=200, cosine LR)")
    print("=" * 65)

    best_gw = best_cfg['gamma_w']
    best_cp = best_cfg['cutoff_pct']

    best_overall = best_mean
    best_threshold = 2.0

    for thr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        print(f"\n  --- threshold={thr} ---")
        X, y = build_dataset_threshold(d_hat, p_bs, p_gt, thr)
        print(f"    LOS ratio={y.mean():.3f}")
        m, bce = train_with_config(X, y, num_user, M,
                                   epochs=200, hidden=64, use_cosine=True)
        print(f"    best BCE={bce:.4f}")

        err = eval_all(d_hat, p_bs, p_gt, m,
                       gamma_w=best_gw, cutoff_pct=best_cp)
        print(f"    {fmt(err)}")

        if err.mean() < best_overall:
            best_overall = err.mean()
            best_threshold = thr
            torch.save(m.state_dict(), f'model_thr{thr}.pkl')
            print(f"    >>> NEW BEST! saved model_thr{thr}.pkl")

    # ========================================
    # PART 3: hidden=128 with best threshold
    # ========================================
    print("\n" + "=" * 65)
    print(f"PART 3: hidden=128 with threshold={best_threshold}")
    print("=" * 65)

    X, y = build_dataset_threshold(d_hat, p_bs, p_gt, best_threshold)
    m128, bce = train_with_config(X, y, num_user, M,
                                  epochs=200, hidden=128, use_cosine=True)
    err128 = eval_all(d_hat, p_bs, p_gt, m128,
                      gamma_w=best_gw, cutoff_pct=best_cp)
    print(f"  hidden=128: {fmt(err128)}")

    if err128.mean() < best_overall:
        best_overall = err128.mean()
        torch.save(m128.state_dict(), 'model_best.pkl')
        print(f"  >>> NEW BEST with h=128! saved model_best.pkl")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"  Current V1 (gw=6, thr=2.0, h=64):  mean=3.2595")
    print(f"  Best found:  mean={best_overall:.4f}")
    print(f"    gamma_w      = {best_gw}")
    print(f"    cutoff_pct   = {best_cp}")
    print(f"    threshold    = {best_threshold}")


if __name__ == "__main__":
    main()
