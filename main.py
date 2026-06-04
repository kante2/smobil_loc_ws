"""
Inference entry point. The grader executes `python main.py` which calls
main(); main() returns a (2, num_user) numpy array of predicted positions.

Algorithm: HLOS-Rwgh-OWLS  (see train.py docstring).

The algorithm and model definitions now live in train.py; importing them
here does NOT trigger training (train.main() is guarded by __main__).
"""
import numpy as np
import scipy.io as sio
import torch

from train import (
    LOSClassifier,
    FEATURE_DIM,
    predict_position,
)


def your_algorithm(d, p_bs, model):
    """Locate one user. d:(18,), p_bs:(2,18) -> (2,)"""
    return predict_position(d, p_bs, model)


def main():
    # 1) load .mat (grader places it in cwd)
    mat_path = 'DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)

    # accept either variable naming (README spec says p_bs;
    # provided sample uses BS_positions — keep both)
    p_bs = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)

    # 2) load trained classifier
    model = LOSClassifier(in_dim=FEATURE_DIM)
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