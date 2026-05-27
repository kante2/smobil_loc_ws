cd ~/smobil_loc_ws
cat > model.py << 'EOF'
"""
Shared model + helpers for BC-WLS (Bias-Corrected Weighted Least Squares).
Imported by both main.py (inference) and train.py (training).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares


# ============================================================
# Neural network: per-BS NLOS bias predictor
# ============================================================
class BiasMLP(nn.Module):
    """
    Input  : per-BS feature vector  (FEATURE_DIM,)
    Output : predicted NLOS bias    scalar (positive = measurement overshot)
    """
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


FEATURE_DIM = 8  # keep in sync with extract_features()


# ============================================================
# Closed-form initial LS estimate (linearised by reference-BS differencing)
# ============================================================
def linear_ls_init(d, p_bs):
    """
    Quick warm-start estimate.
    d: (18,), p_bs: (2, 18)  ->  returns (2,)
    """
    ref = 0
    A = -2.0 * (p_bs[:, 1:] - p_bs[:, ref:ref + 1]).T  # (17, 2)
    b = (d[1:] ** 2 - d[ref] ** 2
         - np.sum(p_bs[:, 1:] ** 2, axis=0)
         + np.sum(p_bs[:, ref] ** 2))
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


# ============================================================
# Nonlinear weighted LS (Levenberg-Marquardt)
# ============================================================
def nonlinear_wls(p0, d, p_bs, w):
    """
    p0: (2,) initial guess,  d: (18,) distances,
    p_bs: (2, 18),           w:  (18,) per-BS weights
    """
    sw = np.sqrt(np.maximum(w, 0.0))

    def residuals(p):
        return sw * (np.linalg.norm(p[:, None] - p_bs, axis=0) - d)

    res = least_squares(residuals, p0, method='lm', max_nfev=50)
    return res.x


# ============================================================
# Per-BS feature extraction
# ============================================================
def extract_features(d_hat, p_bs, p0):
    """
    d_hat: (18,)    measurements
    p_bs : (2, 18)  BS positions
    p0   : (2,)     initial position estimate
    Returns (18, FEATURE_DIM) float32 feature matrix.
    """
    d_pred = np.linalg.norm(p0[:, None] - p_bs, axis=0)  # (18,)
    resid = d_hat - d_pred
    abs_resid = np.abs(resid)

    median_resid = np.median(resid)
    mad_resid = np.median(np.abs(resid - median_resid))

    feats = np.stack([
        d_hat,                       # 1. raw RTT measurement
        d_pred,                      # 2. geometric prediction from p0
        resid,                       # 3. signed residual
        abs_resid,                   # 4. |residual|
        np.full(18, median_resid),   # 5. global median residual
        np.full(18, mad_resid),      # 6. robust spread
        p_bs[0],                     # 7. BS x coordinate
        p_bs[1],                     # 8. BS y coordinate
    ], axis=1)  # (18, 8)
    return feats.astype(np.float32)
EOF