import math

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic_regression(x, y, steps=300, lr=0.1):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_aug = np.column_stack([np.ones(len(x_arr)), x_arr])
    w = np.zeros(x_aug.shape[1], dtype=float)
    for _ in range(steps):
        preds = sigmoid(x_aug @ w)
        grad = x_aug.T @ (preds - y_arr) / len(x_arr)
        w -= lr * grad
    return w.tolist()


def predict_proba(weights, x):
    x_arr = np.asarray(x, dtype=float)
    x_aug = np.column_stack([np.ones(len(x_arr)), x_arr])
    w = np.asarray(weights, dtype=float)
    return sigmoid(x_aug @ w).tolist()


def build_feature_frame(price_map, idx):
    rows = []
    names = []
    for name, prices in price_map.items():
        p5 = prices[idx] / prices[idx - 5] - 1
        p10 = prices[idx] / prices[idx - 10] - 1
        p20 = prices[idx] / prices[idx - 20] - 1
        recent = np.asarray(prices[idx - 5:idx], dtype=float)
        ma5 = float(np.mean(prices[idx - 5:idx]))
        ma20 = float(np.mean(prices[idx - 20:idx]))
        rets = (recent[1:] / recent[:-1] - 1) if len(recent) > 1 else np.asarray([])
        vol5 = float(np.std(rets)) if len(rets) > 0 else 0.0
        rows.append([p5, p10, p20, vol5, ma5 / ma20 - 1])
        names.append(name)
    return names, rows


def build_training_set(price_map, end_idx):
    x = []
    y = []
    for idx in range(20, end_idx - 1):
        names, rows = build_feature_frame(price_map, idx)
        for name, feats in zip(names, rows):
            next_ret = price_map[name][idx + 1] / price_map[name][idx] - 1
            x.append(feats)
            y.append(1.0 if next_ret > 0 else 0.0)
    return x, y


def rank_assets(weights, price_map, idx):
    names, rows = build_feature_frame(price_map, idx)
    probs = predict_proba(weights, rows)
    return {name: prob for name, prob in zip(names, probs)}


def run_model_strategy(price_map, weights, top_n, initial_capital):
    names = list(price_map)
    n = len(next(iter(price_map.values())))
    values = [initial_capital]
    holdings = []
    for idx in range(1, n):
        if idx <= 20:
            selected = names
        else:
            scores = rank_assets(weights, price_map, idx - 1)
            selected = [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]
        holdings.append({name: (1 / len(selected) if name in selected else 0.0) for name in names})
        gross = sum(price_map[name][idx] / price_map[name][idx - 1] for name in selected) / len(selected)
        values.append(values[-1] * gross)
    return values, holdings
