import numpy as np
import pandas as pd
import os
from datetime import datetime

def softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def prediction_metrics(logits_or_probs):
    arr = np.asarray(logits_or_probs, dtype=np.float64)

    if arr.ndim > 1:
        arr = arr.squeeze()

    # if already probabilities, keep them; otherwise softmax
    if np.all(arr >= 0) and np.isclose(arr.sum(), 1.0, atol=1e-3):
        probs = arr
    else:
        probs = softmax_np(arr)

    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top1 - top2

    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(len(probs) + 1e-12)
    normalized_entropy = float(entropy / (max_entropy + 1e-12))

    return {
        "top1_confidence": top1,
        "top2_confidence": top2,
        "margin": float(margin),
        "entropy": float(entropy),
        "normalized_entropy": normalized_entropy,
    }

def cam_focus_score(cam_map):
    """
    cam_map: 2D array normalized or unnormalized
    Returns how concentrated the attention is.
    Higher = more focused.
    """
    cam = np.asarray(cam_map, dtype=np.float64)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-12)

    # proportion of total mass in top 20% hottest pixels
    flat = cam.flatten()
    k = max(1, int(0.2 * len(flat)))
    topk = np.sort(flat)[-k:]
    focus = float(topk.sum() / (flat.sum() + 1e-12))

    # center bias: how much attention is in center 50% region
    h, w = cam.shape
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4
    center_mass = float(cam[y1:y2, x1:x2].sum() / (cam.sum() + 1e-12))

    return {
        "focus_score": focus,
        "center_mass": center_mass,
    }

def feature_metrics(fmap_tensor):
    fmap = fmap_tensor[0].detach().cpu().numpy()   # [C,H,W]
    return {
        "mean_activation": float(fmap.mean()),
        "variance": float(fmap.var()),
        "sparsity_pct": float((fmap < 0.01).mean() * 100.0),
        "num_channels": int(fmap.shape[0]),
        "spatial_h": int(fmap.shape[1]),
        "spatial_w": int(fmap.shape[2]),
    }

def save_feedback_row(row, csv_path="feedback_log.csv"):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)

def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")