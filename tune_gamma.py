"""
tune_gamma.py — gamma_w 추가 탐색 (10~30)
model_thr3.0.pkl 사용
  python3 tune_gamma.py
"""
import numpy as np
import scipy.io as sio
import torch
from itertools import combinations
from model import (
    LOSClassifier, FEATURE_DIM,
    linear_ls_init, extract_features,
    clip_to_room, room_box, nonlinear_wls,
)


def predict_gw(d, p_bs, model, gamma_w):
    M = p_bs.shape[1]
    box = room_box(p_bs)
    p0 = clip_to_room(linear_ls_init(d, p_bs), box)
    feats = extract_features(d, p_bs, p0)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats)).numpy()
    p_los = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    K = min(8, M)
    cand = np.argsort(-p_los)[:K]

    estimates, inv_res = [], []
    for combo in combinations(cand, 4):
        idx = np.array(combo)
        p0_loc = clip_to_room(linear_ls_init(d[idx], p_bs[:, idx]), box)
        p_sub = nonlinear_wls(p0_loc, d[idx], p_bs[:, idx])
        p_sub = clip_to_room(p_sub, box)
        d_pred = np.linalg.norm(p_sub[:, None] - p_bs[:, idx], axis=0)
        r = np.linalg.norm(d_pred - d[idx]) / np.sqrt(2 + 1e-9)
        estimates.append(p_sub)
        inv_res.append(1.0 / (r ** 2.0 + 1e-6))
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


def main():
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    p_bs = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p_gt = np.asarray(data['p'], dtype=float)
    num_user = d_hat.shape[1]

    # threshold=3.0 model
    model = LOSClassifier(in_dim=FEATURE_DIM)
    state = torch.load('model_thr3.0.pkl', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"{'gw':>5s}  {'mean':>7s}  {'median':>7s}  {'90p':>7s}  "
          f"{'95p':>7s}  {'max':>7s}  {'<2m':>6s}  {'<5m':>6s}")
    print("-" * 65)

    for gw in [6, 8, 10, 12, 14, 16, 18, 20, 25, 30]:
        p_hat = np.zeros((2, num_user))
        for u in range(num_user):
            p_hat[:, u] = predict_gw(d_hat[:, u], p_bs, model, gamma_w=gw)
        err = np.linalg.norm(p_hat - p_gt, axis=0)
        print(f"{gw:5.0f}  {err.mean():7.4f}  {np.median(err):7.4f}  "
              f"{np.percentile(err,90):7.3f}  {np.percentile(err,95):7.3f}  "
              f"{err.max():7.2f}  "
              f"{100*(err<2).mean():5.1f}%  {100*(err<5).mean():5.1f}%")


if __name__ == "__main__":
    main()
