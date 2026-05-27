"""
Inference entry point. The grader executes `python main.py` and calls main().
Returns a (2, num_user) numpy array of predicted positions.
"""
import numpy as np
import scipy.io as sio
import torch

from model import (
    BiasMLP, FEATURE_DIM,
    linear_ls_init, nonlinear_wls, extract_features,
)


def your_algorithm(d, p_bs, model):
    """
    Locate one user with bias-corrected weighted LS.
    d   : (18,)   RTT measurements
    p_bs: (2, 18) BS coordinates
    model: trained BiasMLP
    Returns (2,) estimated position.
    """
    # 1. closed-form warm-start
    p0 = linear_ls_init(d, p_bs)

    # 2. predict per-BS NLOS bias
    feats = extract_features(d, p_bs, p0)
    with torch.no_grad():
        bias_hat = model(torch.from_numpy(feats)).numpy()  # (18,)

    # 3. correct the measurements (clip to small positive)
    d_corr = np.maximum(d - bias_hat, 0.1)

    # 4. confidence weights — downweight large residuals after correction
    d_pred = np.linalg.norm(p0[:, None] - p_bs, axis=0)
    new_resid = np.abs(d_corr - d_pred)
    w = 1.0 / (1.0 + new_resid)

    # 5. final weighted nonlinear LS
    p_hat = nonlinear_wls(p0, d_corr, p_bs, w)
    return p_hat


def main():
    # 1) load data — grader places file in cwd with this name
    mat_path = 'DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)
    # Accept either variable naming (spec says p_bs, sample file uses BS_positions)
    p_bs = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)

    # 2) load trained model
    model = BiasMLP(in_dim=FEATURE_DIM)
    state = torch.load('model.pkl', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 3) per-user prediction — keep num_user dynamic per spec
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs, model)

    return p_hat


if __name__ == "__main__":
    p_hat = main()
    print(f"p_hat: shape={p_hat.shape},  "
          f"mean=({p_hat[0].mean():.2f}, {p_hat[1].mean():.2f})")