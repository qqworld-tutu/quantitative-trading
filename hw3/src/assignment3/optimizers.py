import numpy as np


def mean_vector(returns):
    return np.mean(returns, axis=0)


def covariance_matrix(returns):
    return np.cov(returns, rowvar=False)


def min_eigenvalue(cov):
    return float(np.min(np.linalg.eigvalsh(cov)))


def _inv_cov(cov):
    return np.linalg.pinv(np.asarray(cov, dtype=float))


def global_min_variance_weights(cov):
    n = cov.shape[0]
    ones = np.ones(n)
    inv_cov = _inv_cov(cov)
    w = inv_cov @ ones
    return w / (ones @ inv_cov @ ones)


def mean_variance_target_weights(mu, cov, target_return):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    ones = np.ones(n)
    inv_cov = _inv_cov(cov)
    a = float(ones @ inv_cov @ ones)
    b = float(ones @ inv_cov @ mu)
    c = float(mu @ inv_cov @ mu)
    d = a * c - b * b
    if abs(d) < 1e-12:
        return global_min_variance_weights(cov)
    w = inv_cov @ (((c - target_return * b) / d) * ones + ((target_return * a - b) / d) * mu)
    return w


def quadratic_utility_weights(mu, cov, risk_aversion):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    ones = np.ones(n)
    inv_cov = _inv_cov(cov)
    gamma = float((ones @ inv_cov @ mu - risk_aversion) / (ones @ inv_cov @ ones))
    w = (1.0 / risk_aversion) * (inv_cov @ (mu - gamma * ones))
    return w
