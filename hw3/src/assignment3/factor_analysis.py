import numpy as np


def _beta_and_r2(asset, market):
    x = np.asarray(market, dtype=float)
    y = np.asarray(asset, dtype=float)
    x_center = x - np.mean(x)
    y_center = y - np.mean(y)
    beta = float(np.dot(x_center, y_center) / np.dot(x_center, x_center)) if np.dot(x_center, x_center) != 0 else 0.0
    alpha = float(np.mean(y) - beta * np.mean(x))
    fitted = alpha + beta * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return beta, r2


def rolling_beta(asset, market, window=50):
    return [ _beta_and_r2(asset[i:i+window], market[i:i+window])[0] for i in range(len(asset) - window + 1) ]


def rolling_r2_current_beta(asset, market, window=50):
    return [ _beta_and_r2(asset[i:i+window], market[i:i+window])[1] for i in range(len(asset) - window + 1) ]


def rolling_r2_previous_beta(asset, market, window=50):
    betas = rolling_beta(asset, market, window)
    result = []
    x = np.asarray(market, dtype=float)
    y = np.asarray(asset, dtype=float)
    for i in range(len(betas)):
        if i == 0:
            result.append(rolling_r2_current_beta(asset, market, window)[0])
            continue
        beta = betas[i - 1]
        x_win = x[i:i+window]
        y_win = y[i:i+window]
        alpha = float(np.mean(y_win) - beta * np.mean(x_win))
        fitted = alpha + beta * x_win
        ss_res = float(np.sum((y_win - fitted) ** 2))
        ss_tot = float(np.sum((y_win - np.mean(y_win)) ** 2))
        result.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
    return result


def beta_rank_rows(labels, beta_map):
    rows = []
    names = list(beta_map)
    for idx, label in enumerate(labels):
        ordered = sorted(names, key=lambda name: beta_map[name][idx], reverse=True)
        rows.append([label] + ordered)
    return rows
