"""
평가 스크립트 — smobil_loc_ws 폴더에서 실행
  python evaluate.py
"""
import numpy as np
import scipy.io as sio
import torch
from model import (
    LOSClassifier, FEATURE_DIM, predict_position,
    linear_ls_init, clip_to_room, room_box, nonlinear_wls,
)


def metrics(err, label):
    rmse = np.sqrt((err ** 2).mean())
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Mean Error  : {err.mean():.4f} m")
    print(f"  Median Error: {np.median(err):.4f} m")
    print(f"  RMSE        : {rmse:.4f} m")
    print(f"  Std          : {err.std():.4f} m")
    print(f"  90th %%ile   : {np.percentile(err, 90):.4f} m")
    print(f"  95th %%ile   : {np.percentile(err, 95):.4f} m")
    print(f"  Max          : {err.max():.4f} m")
    print(f"  < 1 m       : {(err < 1).mean()*100:.1f}%")
    print(f"  < 2 m       : {(err < 2).mean()*100:.1f}%")
    print(f"  < 5 m       : {(err < 5).mean()*100:.1f}%")
    return err


def main():
    data = sio.loadmat('DH_FR1.mat', squeeze_me=False)
    p_bs  = np.asarray(data.get('p_bs', data.get('BS_positions')), dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p_gt  = np.asarray(data['p'], dtype=float)
    num_user = d_hat.shape[1]
    box = room_box(p_bs)

    print(f"Users: {num_user}  |  BS range x:[{p_bs[0].min():.1f},{p_bs[0].max():.1f}]"
          f"  y:[{p_bs[1].min():.1f},{p_bs[1].max():.1f}]")

    # --- Baseline 1: Linear LS ---
    p_lin = np.zeros((2, num_user))
    for u in range(num_user):
        p_lin[:, u] = clip_to_room(linear_ls_init(d_hat[:, u], p_bs), box)
    err_lin = metrics(np.linalg.norm(p_lin - p_gt, axis=0), "Baseline: Linear LS")

    # --- Baseline 2: Unweighted NLS ---
    p_nls = np.zeros((2, num_user))
    for u in range(num_user):
        p0 = clip_to_room(linear_ls_init(d_hat[:, u], p_bs), box)
        p_nls[:, u] = clip_to_room(nonlinear_wls(p0, d_hat[:, u], p_bs), box)
    err_nls = metrics(np.linalg.norm(p_nls - p_gt, axis=0), "Baseline: NLS (uniform weight)")

    # --- HLOS-Rwgh-WLS ---
    model = LOSClassifier(in_dim=FEATURE_DIM)
    state = torch.load('model.pkl', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = predict_position(d_hat[:, u], p_bs, model)
    err_hat = metrics(np.linalg.norm(p_hat - p_gt, axis=0), "HLOS-Rwgh-WLS (yours)")

    # --- Comparison ---
    print(f"\n{'='*50}")
    print(f"  SUMMARY (Mean Error)")
    print(f"{'='*50}")
    print(f"  Linear LS       : {err_lin.mean():.4f} m")
    print(f"  NLS (uniform)   : {err_nls.mean():.4f} m")
    print(f"  HLOS-Rwgh-WLS   : {err_hat.mean():.4f} m")
    print(f"  vs Linear LS    : -{(1 - err_hat.mean()/err_lin.mean())*100:.1f}%")
    print(f"  vs NLS          : -{(1 - err_hat.mean()/err_nls.mean())*100:.1f}%")

    # --- LOS classifier precision/recall ---
    print(f"\n{'='*50}")
    print(f"  LOS Classifier (precision / recall)")
    print(f"{'='*50}")
    from model import extract_features, LOS_BIAS_THRESHOLD
    tp = fp = fn = tn = 0
    for u in range(num_user):
        d = d_hat[:, u]
        p0 = clip_to_room(linear_ls_init(d, p_bs), box)
        feats = extract_features(d, p_bs, p0)
        with torch.no_grad():
            logits = model(torch.from_numpy(feats)).numpy()
        pred_los = (logits > 0)
        true_d = np.linalg.norm(p_gt[:, u:u+1] - p_bs, axis=0)
        true_los = (d - true_d) < LOS_BIAS_THRESHOLD
        tp += (pred_los & true_los).sum()
        fp += (pred_los & ~true_los).sum()
        fn += (~pred_los & true_los).sum()
        tn += (~pred_los & ~true_los).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")


if __name__ == "__main__":
    main()
