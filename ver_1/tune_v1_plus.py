"""
tune_v1_plus.py — V1 모델 유지 + hyperparams 변경 + iterative refinement
  python3 tune_v1_plus.py
"""
import numpy as np
import scipy.io as sio
import torch
from itertools import combinations
from model import (
    LOSClassifier, FEATURE_DIM, linear_ls_init, extract_features,
    clip_to_room, room_box, nonlinear_wls,
)


def predict_position_tuned(d, p_bs, model,
                           top_k=6, k_subset=4, gamma_r=1.5, gamma_w=6.0,
                           n_iter=2):
    """
    V1 pipeline + iterative refinement.
    After step 5, re-extract features using the refined position,
    re-estimate LOS probs, and do another weighted NLS.
    """
    M = p_bs.shape[1]
    box = room_box(p_bs)

    # ---- 1. warm start ----
    p0 = clip_to_room(linear_ls_init(d, p_bs), box)

    for iteration in range(n_iter):
        # ---- 2. LOS probabilities ----
        feats = extract_features(d, p_bs, p0)
        with torch.no_grad():
            logits = model(torch.from_numpy(feats)).numpy()
        p_los = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

        # ---- 3. candidate pool ----
        K = min(top_k, M)
        cand = np.argsort(-p_los)[:K]

        # ---- 4. Rwgh subset combination ----
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

        # ---- 5. final P(LOS)-weighted NLS ----
        w_final = np.maximum(p_los ** gamma_w, 1e-3)
        p0 = clip_to_room(nonlinear_wls(p_mid, d, p_bs, w_final), box)

    return p0


def evaluate(d_hat, p_bs, p_gt, model, **kwargs):
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = predict_position_tuned(d_hat[:, u], p_bs, model, **kwargs)
    err = np.linalg.norm(p_hat - p_gt, axis=0)
    return err


def print_metrics(err, label):
    print(f"\n  === {label} ===")
    print(f"  Mean   : {err.mean():.4f} m")
    print(f"  Median : {np.median(err):.4f} m")
    print(f"  RMSE   : {np.sqrt((err**2).mean()):.4f} m")
    print(f"  90th   : {np.percentile(err, 90):.4f} m")
    print(f"  95th   : {np.percentile(err, 95):.4f} m")
    print(f"  Max    : {err.max():.4f} m")
    print(f"  <1m    : {(err<1).mean()*100:.1f}%")
    print(f"  <2m    : {(err<2).mean()*100:.1f}%")
    print(f"  <5m    : {(err<5).mean()*100:.1f}%")


def main():
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    p_bs  = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p_gt  = np.asarray(data['p'], dtype=float)

    model = LOSClassifier(in_dim=FEATURE_DIM)
    state = torch.load('model.pkl', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ---- Test configs ----
    configs = [
        ("V1 original (baseline)",
         dict(top_k=8, k_subset=4, gamma_r=2.0, gamma_w=4.0, n_iter=1)),

        ("V1 + gamma_w=6.0",
         dict(top_k=8, k_subset=4, gamma_r=2.0, gamma_w=6.0, n_iter=1)),

        ("V1 + top_k=6, gamma_w=6.0",
         dict(top_k=6, k_subset=4, gamma_r=1.5, gamma_w=6.0, n_iter=1)),

        ("V1 + gamma_w=6.0 + iter=2",
         dict(top_k=8, k_subset=4, gamma_r=2.0, gamma_w=6.0, n_iter=2)),

        ("V1 + top_k=6, gamma_w=6.0, iter=2",
         dict(top_k=6, k_subset=4, gamma_r=1.5, gamma_w=6.0, n_iter=2)),

        ("V1 + top_k=6, gamma_w=6.0, iter=3",
         dict(top_k=6, k_subset=4, gamma_r=1.5, gamma_w=6.0, n_iter=3)),

        ("V1 + gamma_w=5.0, iter=2",
         dict(top_k=8, k_subset=4, gamma_r=2.0, gamma_w=5.0, n_iter=2)),

        ("V1 + top_k=7, gamma_w=5.0, iter=2",
         dict(top_k=7, k_subset=4, gamma_r=2.0, gamma_w=5.0, n_iter=2)),
    ]

    print("=" * 60)
    print("  V1 model + hyperparams + iterative refinement")
    print("=" * 60)

    for label, kwargs in configs:
        err = evaluate(d_hat, p_bs, p_gt, model, **kwargs)
        print_metrics(err, label)


if __name__ == "__main__":
    main()
