"""Standalone runnable version of the Colab notebook — for local verification."""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist


def load_libsvm_1d(path):
    xs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            xs.append(float(parts[1].split(':')[1]))
    return np.array(xs).reshape(-1, 1)


def platt_fit(f, y):
    Np = (y == 1).sum(); Nn = (y == -1).sum()
    t = np.where(y == 1, (Np + 1) / (Np + 2), 1.0 / (Nn + 2))
    def nll(ab):
        A, B = ab
        z = A * f + B
        return np.sum(t * z + np.logaddexp(0, -z))
    res = minimize(nll, x0=[0.0, np.log((Nn + 1) / (Np + 1))], method='Nelder-Mead')
    return res.x


def platt_predict(f, AB):
    A, B = AB
    return 1.0 / (1.0 + np.exp(A * f + B))


def binning_equidistant(f_train, f_test, n=5, eps=0.001):
    fmin, fmax = f_train.min(), f_train.max()
    marks = np.concatenate([np.linspace(fmin, 0, n + 1)[:-1], [0.0], np.linspace(0, fmax, n + 1)[1:]])
    probs = np.linspace(0, 1, 2 * n + 1); probs[0], probs[-1] = eps, 1 - eps
    idx = np.argmin(np.abs(f_test[:, None] - marks[None, :]), axis=1)
    return probs[idx]


def binning_density(f_train, f_test, n=5, eps=0.001):
    neg = np.sort(f_train[f_train < 0]); pos = np.sort(f_train[f_train >= 0])
    def centers(arr, n):
        if len(arr) == 0: return np.array([])
        return np.quantile(arr, (np.arange(n) + 0.5) / n)
    marks = np.concatenate([centers(neg, n), [0.0], centers(pos, n)])
    probs = np.linspace(0, 1, 2 * n + 1); probs[0], probs[-1] = eps, 1 - eps
    idx = np.argmin(np.abs(f_test[:, None] - marks[None, :]), axis=1)
    return probs[idx]


def new_gamma_scaling(f_train, f_test):
    fmax = f_train.max()
    S_train = np.maximum(fmax - f_train, 0)
    mu, var = S_train.mean(), S_train.var()
    k, theta = mu * mu / var, var / mu
    S_test = np.maximum(fmax - f_test, 0)
    cdf = gamma_dist.cdf(S_test, a=k, scale=theta)
    cdf_max = gamma_dist.cdf(fmax, a=k, scale=theta)
    p_normal = 1 - cdf; p_dec0 = 1 - cdf_max
    return np.where(p_normal >= p_dec0,
                    0.5 + 0.5 * (p_normal - p_dec0) / (1 - p_dec0),
                    0.5 * p_normal / p_dec0)


if __name__ == '__main__':
    DATA = r'D:/advanced ml/one-class-svm-output/data'
    X_train = load_libsvm_1d(f'{DATA}/art1')
    X_test  = load_libsvm_1d(f'{DATA}/art1.t')
    p_true  = np.loadtxt(f'{DATA}/art1_prob.t')

    clf = OneClassSVM(kernel='rbf', nu=0.25, gamma=0.0001).fit(X_train)
    f_train = clf.decision_function(X_train).ravel()
    f_test  = clf.decision_function(X_test).ravel()

    y_pred = np.where(f_train >= 0, 1, -1)
    AB = platt_fit(f_train, y_pred)
    p_platt = platt_predict(f_test, AB)
    p_eq  = binning_equidistant(f_train, f_test)
    p_den = binning_density(f_train, f_test)
    p_gam = new_gamma_scaling(f_train, f_test)

    print(f'{"Method":<22} {"MSE":>10}')
    print('-' * 36)
    for name, p in [('Platt scaling', p_platt), ('Binning equidistant', p_eq),
                    ('Binning by density', p_den), ('New Gamma scaling', p_gam)]:
        print(f'{name:<22} {mean_squared_error(p_true, p):10.6f}')
    print('\nPaper Table I (ART1):')
    print('  Platt 0.077723 | eq 0.026122 | density 0.001056 | new Gamma 0.000003')
