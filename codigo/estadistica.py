"""ACF y Ljung-Box escritos con numpy; simulación con semillas explícitas."""
import numpy as np
import pandas as pd
from scipy.stats import chi2


def acf_manual(y, max_lag=21):
    y = np.asarray(y, dtype=float)
    z = y - y.mean()
    denominador = z @ z
    if not np.isfinite(y).all() or denominador == 0:
        raise ValueError('La ACF requiere valores finitos y varianza positiva.')
    return np.array([z[k:] @ z[:-k] / denominador
                     for k in range(1, max_lag + 1)])


def ljung_box(y, m, parametros=0):
    n = len(y)
    r = acf_manual(y, m)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, m + 1)))
    return {'Q': float(q), 'gl': m - parametros,
            'p': float(chi2.sf(q, m - parametros))}


def simular_ar1(phi, n=300, replicas=1, semilla=202600, burn=200):
    rng = np.random.default_rng(semilla)
    eps = rng.normal(size=(replicas, n + burn))
    y = np.zeros_like(eps)
    y[:, 0] = eps[:, 0]
    for t in range(1, n + burn):
        y[:, t] = phi * y[:, t - 1] + eps[:, t]
    return y[:, burn:]


def resumir_replicas(y, nombre, semilla, phi=None):
    replicas, n = y.shape
    z = y - y.mean(axis=1, keepdims=True)
    den = (z * z).sum(axis=1)
    r = np.column_stack([(z[:, k:] * z[:, :-k]).sum(axis=1) / den
                         for k in range(1, 11)])
    x = y[:, :-1] - y[:, :-1].mean(axis=1, keepdims=True)
    v = y[:, 1:] - y[:, 1:].mean(axis=1, keepdims=True)
    phi_ols = (x * v).sum(axis=1) / (x * x).sum(axis=1)
    rechazo = np.abs(r[:, 0]) > 1.96 / np.sqrt(n)
    q = n * (n + 2) * (r * r / (n - np.arange(1, 11))).sum(axis=1)
    lb = q > chi2.ppf(.95, 10)
    fila = {'escenario': nombre, 'semilla': semilla,
            'phi5_referencia': np.nan if phi is None else phi ** 5}
    for clave, valores in [('r1', r[:, 0]), ('r5', r[:, 4]), ('r7', r[:, 6]),
                          ('phi_ols', phi_ols), ('rechazo_r1', rechazo),
                          ('rechazo_lb10', lb)]:
        fila[clave] = float(np.mean(valores))
        fila[clave + '_se'] = float(np.std(valores, ddof=1) / np.sqrt(replicas))
    return fila
