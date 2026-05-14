import json
import gdown

import os
import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers, shifted_window

if not os.path.exists("challenge.mat"):
    print("Dataset not found. Downloading...")
    url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
    downloaded_file = "challenge.mat"
    gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)

def extract_rank1_interference(band_matrix):
    cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = band_matrix @ vecs[:, -1]
    denom = np.vdot(shared, shared) + 1e-30
    
    return np.column_stack([
        (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
        for ch in range(band_matrix.shape[1])
    ])

def your_canceller(tx_n, rx):
    """
    Features:
    1. Extended set of basis functions: 3d, 5th, 7th orders of nonlinearity;
    2. Deep Memory model: evaluated taps [-20,20]
    3. Alternating Least Squares algorithm: Joint estimation of Conductive PIM and Rank-1 
       spatial interference
    """
    pairs =[(0, 1), (1, 0), (0, 3), (3, 0), (1, 2), (2, 1), (3, 2), (2, 3), (0, 5), (5, 0)]
    
    raw_model_terms =[]
    for a, b in pairs:
        base = tx_n[:, a] ** 2 * tx_n[:, b].conj()
        ma = np.abs(tx_n[:, a]) ** 2
        mb = np.abs(tx_n[:, b]) ** 2
        
        raw_model_terms.extend([
            base,                                        # 3rd
            base * ma,           base * mb,              # 5th
            base * (ma**2),      base * ma * mb,         base * (mb**2), # 7th
        ])
        
    raw_model_terms = tuple(raw_model_terms)
    lags = tuple(range(-25, 24))
    subset = slice(20_000, 220_000)
    
    print(f"Building basis matrix ({len(raw_model_terms)} terms x {len(lags)} lags = {len(raw_model_terms)*len(lags)} columns)...")

    raw_model_x = np.column_stack([
        shifted_window(t, l, subset.start, subset.stop)
        for t in raw_model_terms for l in lags
    ])

    print("Computing pseudo-inverse Gram matrix...")
    raw_gram = raw_model_x.conj().T @ raw_model_x + 1e-6 * np.eye(raw_model_x.shape[1])
    inv_gram = np.linalg.inv(raw_gram)
    
    X_H = raw_model_x.conj().T

    def fit_deep_tx_fast(target_rx):
        Y = target_rx[subset, :]
        
        coefs_all = inv_gram @ (X_H @ Y) 
        
        coefs_reshaped = coefs_all.reshape(len(raw_model_terms), len(lags), rx.shape[1])
    
        pred = np.zeros_like(target_rx)
        
        for t_idx, t in enumerate(raw_model_terms):
            t_shifts = np.zeros((target_rx.shape[0], len(lags)), dtype=np.complex128)
            for l_idx, lag in enumerate(lags):
                if lag >= 0:
                    t_shifts[lag:, l_idx] = t[:len(t)-lag]
                else:
                    kk = -lag
                    t_shifts[:len(t)-kk, l_idx] = t[kk:]
            
            coefs_for_term = coefs_reshaped[t_idx, :, :]
            pred += t_shifts @ coefs_for_term
            
        return pred

    print("Running ALS iterations...")
    e_raw = np.zeros_like(rx)
    tx_raw_pred = np.zeros_like(rx)
    
    for i in range(3):
        print(f"  Iteration {i+1}/3...")
        tx_raw_pred = fit_deep_tx_fast(rx - e_raw)
        
        r_raw = rx - tx_raw_pred
        r_band = np.column_stack([helpers["score_filter"](r_raw[:, ch]) for ch in range(r_raw.shape[1])])
        
        cov = r_band.conj().T @ r_band
        _, vecs = np.linalg.eigh(cov)
        v = vecs[:, -1]
        
        e_raw = np.outer(r_raw @ v, v.conj())
    
    print("Done. Returning clean signal.")
    return rx - tx_raw_pred - e_raw

print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
