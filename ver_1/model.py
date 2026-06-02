"""
Model and shared helpers for HLOS-Rwgh-WLS positioning.

Module is imported by both train.py and main.py.

Algorithm short name : HLOS-Rwgh-WLS
                       (Hybrid LOS-classifier + Residual-Weighted-Subset + WLS)

Pipeline (no absolute coordinates used as features, for generalization):
  step 1 : closed-form linear LS warm start, clipped to room box
  step 2 : per-BS feature vector -> small MLP -> P(LOS_i)
  step 3 : take top-K BSes by P(LOS_i), enumerate k-subsets, solve NLS in each
  step 4 : combine subsets weighted by inverse residual^gamma (Chen 1999 Rwgh)
  step 5 : final P(LOS_i)^gamma weighted NLS over all 18 BSes, warm started
           from step 4's combined estimate

Reference papers:
  [Chen99]    P.-C. Chen, "A non-line-of-sight error mitigation algorithm
              in location estimation," IEEE WCNC 1999.
  [Breg18]    K. Bregar & M. Mohorčič, "Improving Indoor Localization Using
              Convolutional Neural Networks on Computationally Restricted
              Devices," IEEE Access 2018.
  [Kend17]    A. Kendall & Y. Gal, "What Uncertainties Do We Need ...", NIPS 2017
              (heteroscedastic loss — explored as Approach C but not chosen).
"""
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares


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
    if w is None:
        w = np.ones(p_bs.shape[1])
    sw = np.sqrt(np.maximum(w, 0.0))
    def res(p):
        return sw * (np.linalg.norm(p[:, None] - p_bs, axis=0) - d)
    return least_squares(res, p0, method='lm', max_nfev=max_nfev).x


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
LOS_BIAS_THRESHOLD = 3.0   # used as label threshold during training 2.0 -> 3.0


def predict_position(d, p_bs, model,
                     top_k=8, k_subset=4, gamma_r=2.0, gamma_w=12.0): # 4.0 -> 6.0 -> 12.0
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

    # ---- 4. Rwgh: residual-weighted subset combination ----
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
    # drop the worst half of subsets, weight rest by inv_res
    cutoff = np.percentile(inv_res, 50)
    mask = inv_res >= cutoff
    w_sub = inv_res[mask] / inv_res[mask].sum()
    p_mid = (estimates[mask] * w_sub[:, None]).sum(axis=0)
    p_mid = clip_to_room(p_mid, box)

    # ---- 5. final refinement: P(LOS)^gamma-weighted NLS over all BSes ----
    w_final = np.maximum(p_los ** gamma_w, 1e-3)
    p_hat = nonlinear_wls(p_mid, d, p_bs, w_final)
    return clip_to_room(p_hat, box)
