#!/usr/bin/env python3
"""
RAINet-inspired prototype for rotational-spectrum analysis (v4).

Scope:
- peak-list input (CSV/TXT, one frequency in MHz per row)
- synthetic training data for linear rotors, prolate symmetric tops and
  asymmetric tops (rigid rotor, a-type R-branch, parameters A, B, C)
- neural-network classification followed by type-specific regression
- frequency-only input (no intensities), as in Zaleski & Prozument (2018)
- optional physical refinement of the ML estimate (linear least squares
  on lines assigned to the nearest predicted transition) - the paper
  itself says the ANN output should be "tightened up with standard fitting"
"""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


# === CONFIGURATION ===

N_PEAKS = 60                 # number of lowest peaks used as ML input
N_BINS = N_PEAKS             # kept for dashboard compatibility
FREQ_MIN_MHZ = 0.0
FREQ_MAX_MHZ = 300_000.0
MERGE_TOL_MHZ = 0.15         # lines closer than this are one peak
MIN_PEAKS = 2
CLASSES = ("linear", "symmetric", "asymmetric")
PARAM_NAMES = {
    "linear": ["B_MHz", "DJ_MHz"],
    "symmetric": ["B_MHz", "DJ_MHz", "DJK_MHz"],
    "asymmetric": ["A_MHz", "B_MHz", "C_MHz"],
}


# === SPECTRUM GENERATION ===

@lru_cache(maxsize=None)    # the matrix is computed once for each (kind, j_max) pair
def _design_matrix(kind: str, j_max: int) -> np.ndarray:
    """
    Matrix M: nu = M @ params, for all transitions J -> J+1
    (J = 0,...,j_max-1) in MHz.
    The model is linear in the parameters; the coefficients depend only on J and K.
        linear:    nu = 2*B*(J+1) - 4*DJ*(J+1)^3
        symmetric: nu = 2*B*(J+1) - 4*DJ*(J+1)^3 - 2*DJK*(J+1)*K^2,  K = 0,...,J
    """
    if kind == "linear":
        jj = np.arange(1, j_max + 1, dtype=float)   # jj = J+1
        M = np.column_stack([2.0 * jj, -4.0 * jj**3])           # columns: B, DJ
    else:
        # "symmetric_h" adds the sextic terms HJ, HJK, HKJ:
        #   E += HJ * (J*(J+1))^3 + HJK * (J*(J+1))^2 * K^2 + HKJ J*(J+1) * K^4
        # with x = (J+1)*(J+2), y = J*(J+1), jj = J+1:
        #   nu_HJ = 2 * jj * (x^2 + x * y + y^2),  nu_HJK = 4 * jj^3 * K^2,  nu_HKJ = 2 * jj * K^4
        rows = []
        for j in range(j_max):
            jj = j + 1.0
            x, y = (j + 1.0) * (j + 2.0), j * (j + 1.0)
            for k in range(j + 1):
                row = (2.0 * jj, -4.0 * jj**3, -2.0 * jj * k**2)
                if kind == "symmetric_h":
                    row = row + (2.0 * jj * (x * x + x * y + y * y),    # HJ
                                 4.0 * jj**3 * k**2,                    # HJK
                                 2.0 * jj * k**4)                       # HKJ
                rows.append(row)
        M = np.asarray(rows, dtype=float)
    M.setflags(write=False)
    return M


def linear_frequencies(B: float, DJ: float, j_max: int,
                       f_max: float = FREQ_MAX_MHZ) -> np.ndarray:
    """
    Linear molecule, nu(J -> J+1) in MHz.
    """
    nu = _design_matrix("linear", int(j_max)) @ np.array([B, DJ], dtype=float)
    return nu[(nu > 0) & (nu <= f_max)]


SYM_H_NAMES = ["HJ_MHz", "HJK_MHz", "HKJ_MHz"]


def symmetric_top_frequencies(B: float, DJ: float, DJK: float, j_max: int,
                              f_max: float = FREQ_MAX_MHZ, sextic=None) -> np.ndarray:
    """
    Prolate symmetric molecule (DeltaJ=+1, DeltaK=0).
    A cancels in these transitions, so it is not a parameter.
    E(J, K) = B * J*(J+1) - DJ * [J*(J+1)]^2 - DJK J*(J+1) * K^2 + (A-B) * K^2
    """
    if sextic is None:
        nu = _design_matrix("symmetric", int(j_max)) @ np.array([B, DJ, DJK], dtype=float)
    else:
        nu = _design_matrix("symmetric_h", int(j_max)) @ np.array(
            [B, DJ, DJK, *sextic], dtype=float)
    return nu[(nu > 0) & (nu <= f_max)]


# === ASYMMETRIC (A-TYPE) ===

KB_MHZ_PER_K = 20836.619      # k_B / h in MHz per kelvin

ASYM_DIST_NAMES = ["DJ_MHz", "DJK_MHz", "DK_MHz", "dJ_MHz", "dK_MHz"]

@lru_cache(maxsize=None)
def _asym_basis(J: int):
    """
    K, K^2 and the |<K+2|J+^2|K>| factors for one J
    """
    K = np.arange(-J, J + 1, dtype=float)   # 2J + 1 states
    jj = J * (J + 1.0)
    k = K[:-2]      # all K that have a partner K+2
    # matrix element <K+2|J+^2|K>
    off = np.sqrt(np.clip((jj - k * (k + 1)) * (jj - (k + 1) * (k + 2)), 0.0, None))
    K2 = K * K
    for arr in (K, K2, off, k):
        arr.setflags(write=False)
    return jj, K, K2, k, off


def _asym_levels(A: float, B: float, C: float, J: int, dist=None) -> np.ndarray:
    """
    Asymmetric molecuule energy levels of one J.
    Ascending index i corresponds to (Ka, Kc) with
        Ka = ceil(i/2), Kc = J - Ka (i even) or J + 1 - Ka (i odd).
    """
    if J == 0:
        return np.zeros(1)      # fro J = 0 we have one energy level = 0
    jj, K, K2, k, off = _asym_basis(int(J))
    bc = 0.5 * (B + C)
    diag = bc * jj + (A - bc) * K2  # energy on the diagonal E = (B+C)/2 * J*(J+1) + (A-(B+C)/2) * K^2
    g = np.full(len(off), 0.25 * (B - C))
    if dist is not None:
        DJ, DJK, DK, dJ, dK = dist
        diag = diag - DJ * jj * jj - DJK * jj * K2 - DK * K2 * K2
        g = g - dJ * jj - 0.5 * dK * (k * k + (k + 2.0) ** 2)
    H = np.zeros((2 * J + 1, 2 * J + 1))
    idx = np.arange(2 * J - 1)
    if J >= 1:
        H[idx, idx + 2] = off * g
        H[idx + 2, idx] = off * g
    H[np.diag_indices_from(H)] = diag
    return np.linalg.eigvalsh(H)


def asym_a_type_lines(A: float, B: float, C: float, j_max: int,
                      f_max: float = FREQ_MAX_MHZ, e_lower_max: float = np.inf,
                      return_labels: bool = False, dist=None):
    """
    a-type R-branch lines (J, Ka, Kc) -> (J+1, Ka, Kc+1), J = 0,...,j_max-1.
    In ascending-energy ordering these connect level i of J with level i of J+1,
    so nu = E_{J+1}(i) - E_J(i).
    """
    freqs, labels = [], []
    prev = _asym_levels(A, B, C, 0, dist)       # levels of J = 0
    for J in range(int(j_max)):
        nxt = _asym_levels(A, B, C, J + 1, dist)    # levels of J + 1
        # level i of J -> level i of J+1 (a-type); the last two levels of J+1
        # have no partner in J, so only first 2J+1 are used
        nu = nxt[:2 * J + 1] - prev
        keep = (nu > 0) & (nu <= f_max) & (prev <= e_lower_max)
        freqs.append(nu[keep])
        if return_labels:
            for i in np.flatnonzero(keep):
                ka = (i + 1) // 2
                kc = J - ka if i % 2 == 0 else J + 1 - ka
                labels.append((J, ka, kc))
        prev = nxt
    out = np.concatenate(freqs) if freqs else np.zeros(0)
    return (out, labels) if return_labels else out


def _merge_close(freqs: np.ndarray, tol: float = MERGE_TOL_MHZ) -> np.ndarray:
    """
    Sort and merge lines that are close
    """
    f = np.sort(np.asarray(freqs, dtype=float))
    if len(f) < 2:
        return f
    starts = np.r_[True, np.diff(f) > tol]
    return f[starts]


def make_peak_list(lines: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Mimic an experimental peak pick: missing lines, noise, etc.
    """
    if len(lines) == 0:
        return lines
    # missing lines
    keep = rng.random(len(lines)) > rng.uniform(0.0, 0.25)
    observed = lines[keep]
    # noise
    if len(observed):
        observed = observed + rng.normal(0.0, rng.uniform(0.005, 0.08), len(observed))
    # Noise can push a line below 0 or above FREQ_MAX_MHZ
    # such peaks are removed
    observed = observed[(observed > 0) & (observed <= FREQ_MAX_MHZ)]
    return _merge_close(observed)


# === ML FEATURES ===

N_RATIO = 12   # number of leading peaks and gaps given to the network

def spectrum_to_features(
    frequencies: np.ndarray,
    n_bins: int = N_PEAKS,
    f_min: float = FREQ_MIN_MHZ,
    f_max: float = FREQ_MAX_MHZ,
) -> np.ndarray:
    """
    Fixed-length feature vector from a variable-length peak list.

    Layout (2 + 7 + 2*N_RATIO values):
      log10 f1                      for linear moelcule f1 is approx 2B
      N_found / n_bins              how many lines are present
      percentiles of log10(1+gap)   0,10,25,50,75,90,100 - summary of the
                                    line spacing (J spacing and K splittings)
                                    (uniform for linear molecule), (small K splitting and huge J spacing for symmetric molecule)
      f_i / f1,  i = 2..N_RATIO+1   scale-free J pattern
      log10(1+gap_i), i = 1..N_RATIO  small K-splittings stay visible in MHz
    Intensities are not used (as in the paper).
    """
    n = int(n_bins)
    x = np.zeros(2 + 7 + 2 * N_RATIO, dtype=float)
    f = np.asarray(frequencies, dtype=float).ravel()
    f = f[np.isfinite(f) & (f > max(f_min, 0.0)) & (f <= f_max)]
    f = _merge_close(f)[:n]
    if len(f) == 0:
        return x
    x[0] = np.log10(f[0])       # fisrt peak in logarithmic scale. Information about the scale
    x[1] = len(f) / n           # N_found / n_bins
    if len(f) > 1:
        gaps = np.log10(1.0 + np.diff(f))
        x[2:9] = np.percentile(gaps, [0, 10, 25, 50, 75, 90, 100])
        m = min(N_RATIO, len(f) - 1)
        x[9:9 + m] = f[1:m + 1] / f[0]
        x[9 + N_RATIO:9 + N_RATIO + m] = gaps[:m]
    return x


# === TRAINING-SET GENERATION ===

def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """
    random number from [lo,hi] uniform in log scale, so every order of magnitude is equally likely
    """
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def _sample_linear(rng):
    """
    Make one synthetic training sample for linear molecule
    """
    while True:
        B = _log_uniform(rng, 500.0, 60000.0)  # 500 MHz - heavy molecules, 60 GHz - light molecules
        DJ = B * 10 ** rng.uniform(-7.5, -5.5)      # HCN: 2e-6*B, OCS: 2e-7*B
        j_max = int(rng.integers(6, 36))
        peaks = make_peak_list(linear_frequencies(B, DJ, j_max), rng)
        if len(peaks) >= MIN_PEAKS:
            return peaks, [B, DJ]

def _sample_symmetric(rng):
    """
    Make one synthetic training sample for symmetric molecule (prolate)
    """
    while True:
        B = _log_uniform(rng, 500.0, 30000.0)
        DJ = B * 10 ** rng.uniform(-7.5, -5.5)
        DJK = rng.choice([-1.0, 1.0]) * 10 ** rng.uniform(-2.5, -0.5)  # 3 kHz,...,0.3 MHz
        j_max = int(rng.integers(6, 26))
        peaks = make_peak_list(symmetric_top_frequencies(B, DJ, DJK, j_max), rng)
        if len(peaks) >= MIN_PEAKS:
            return peaks, [B, DJ, DJK]


def _sample_asymmetric(rng):
    """
    Make training sample for asymmetric molecule

    Centrifugal distortion is added to 75% of the spectra
    (real molecules always have it) and only
    lines whose lower state lies below a few kT are kept.
    """
    while True:
        B = _log_uniform(rng, 1500.0, 60000.0)
        C = B * rng.uniform(0.45, 0.985)                # always C < B
        A = B * (1.0 + 10 ** rng.uniform(-1.5, 1.0))    # A/B ratio is approx 1.03 to 11 (always A > B)
        dist = None
        if rng.random() < 0.75:
            DJ = B * 10 ** rng.uniform(-7.5, -5.5)
            DJK = rng.choice([-1.0, 1.0]) * DJ * 10 ** rng.uniform(0.0, 2.3)
            DK = rng.choice([-1.0, 1.0]) * A * 10 ** rng.uniform(-7.0, -4.7)
            dJ = DJ * rng.uniform(0.05, 0.4)
            dK = rng.choice([-1.0, 1.0]) * abs(DJK) * rng.uniform(0.02, 0.5)
            dist = (DJ, DJK, DK, dJ, dK)
        temp = rng.uniform(5.0, 150.0)
        e_max = 3.5 * KB_MHZ_PER_K * temp
        j_max = int(rng.integers(5, 15))
        lines = asym_a_type_lines(A, B, C, j_max, e_lower_max=e_max, dist=dist)
        peaks = make_peak_list(lines, rng)
        if len(peaks) >= 4:
            return peaks, [A, B, C]


def generate_training_set(n_per_class: int, seed: int = 7):
    """
    Generates synthetic training set
    Returns {kind: (peak_lists, features, targets)}.
    """
    rng = np.random.default_rng(seed)
    samplers = {"linear": _sample_linear, "symmetric": _sample_symmetric,
                "asymmetric": _sample_asymmetric}
    data = {}
    for kind in CLASSES:
        peak_lists, X, Y = [], [], []
        for _ in range(n_per_class):
            peaks, target = samplers[kind](rng)
            peak_lists.append(peaks)
            X.append(spectrum_to_features(peaks))
            Y.append(target)
        data[kind] = (peak_lists, np.asarray(X), np.asarray(Y, dtype=float))
    return data


# === REFINEMENT ===

def _refine_from(kind, p_start, p_prior, peaks, f_max, n_iter, sigma_obs):
    """
    Returns (params, n_matched, rms) or None. For linear or symmetric molecule
    """
    n_par = len(p_start)
    lo = np.array([1.0, 0.0, -1.0][:n_par])
    hi = np.array([1e6, 1.0, 1.0][:n_par])
    scale = np.array([1000.0, 0.01, 0.01][:n_par])          # column scaling
    prior_sigma = np.array([np.inf, 0.02, 0.05][:n_par])
    j_cap = 400 if kind == "linear" else 60

    p = np.clip(p_start, lo, hi)

    for it in range(n_iter):
        # low J lines first, more lines in later iterations.
        pk = peaks[:max(6, int(round(6 * 1.7**it)))] if it < n_iter - 1 else peaks
        spacing = 2.0 * p[0]
        tol = max(1.0, 0.4 * spacing * 0.6**it)
        j_max = int(min(max(pk[-1] / spacing + 3, 3), j_cap))
        M = _design_matrix(kind, j_max)
        nu = M @ p
        valid = nu > 0
        if not valid.any():
            return None
        Mv, nuv = M[valid], nu[valid]
        idx = np.abs(pk[:, None] - nuv[None, :]).argmin(axis=1)
        sel = np.abs(pk - nuv[idx]) <= tol
        if sel.sum() < max(MIN_PEAKS, n_par - 1):
            return None

        A = Mv[idx[sel]] * scale / sigma_obs
        b = pk[sel] / sigma_obs
        rows, rhs = [], []
        for i in range(1, n_par):
            row = np.zeros(n_par)
            row[i] = scale[i] / prior_sigma[i]
            rows.append(row)
            rhs.append(p_prior[i] / prior_sigma[i])
        sol = lsq_linear(
            np.vstack([A] + rows), np.concatenate([b, np.array(rhs)]),
            bounds=(lo / scale, hi / scale),
        )
        p = sol.x * scale

    # final assessment with a 1 MHz tolerance
    j_max = int(min(max(peaks[-1] / (2.0 * p[0]) + 3, 3), j_cap))
    nu = _design_matrix(kind, j_max) @ p
    nu = nu[nu > 0]
    res = peaks - nu[np.abs(peaks[:, None] - nu[None, :]).argmin(axis=1)]
    ok = np.abs(res) <= 1.0
    if not ok.any():
        return None
    return p, int(ok.sum()), float(np.sqrt(np.mean(res[ok] ** 2)))


def refine_parameters(kind: str, p0, peaks, f_max: float = FREQ_MAX_MHZ,
                      n_iter: int = 8, max_peaks: int = 200,
                      sigma_obs: float = 0.05, scan: float = 0.30,
                      n_starts: int = 41):
    """
    Each peak is assigned to the nearest predicted transition (the tolerance
    shrinks every iteration);
    Returns (parameters, info). If too few peaks can be assigned the ML
    estimate is returned unchanged (info["refined"] is False).
    """
    p0 = np.asarray(p0, dtype=float)
    info = {"refined": False, "n_peaks_used": 0, "n_matched": 0, "rms_mhz": None}

    peaks = np.asarray(peaks, dtype=float)
    peaks_all = _merge_close(peaks[np.isfinite(peaks) & (peaks > 0) & (peaks <= f_max)])
    peaks = peaks_all[:max_peaks]
    info["n_peaks_used"] = int(len(peaks))
    if len(peaks) < MIN_PEAKS:
        return p0, info

    if kind == "asymmetric":
        best = _refine_asym(p0, peaks, f_max)
        if best is None or best[2] < max(MIN_PEAKS, 0.5 * len(peaks)):
            return p0, info
        info.update(refined=True, n_matched=best[2], rms_mhz=best[3],
                    distortion_fitted=bool(np.any(best[1][3:] != 0)))
        return best[1], info

    best = None
    for factor in np.linspace(1.0 - scan, 1.0 + scan, n_starts):
        start = p0.copy()
        start[0] = p0[0] * factor
        out = _refine_from(kind, start, p0, peaks, f_max, n_iter, sigma_obs)
        if out is None:
            continue
        p, n_match, rms = out
        # prefer more matched peaks, then lower rms, then closeness to the ML guess
        key = (n_match, -rms - 1e-3 * abs(np.log(p[0] / p0[0])))
        if best is None or key > best[0]:
            best = (key, p, n_match, rms)

    if best is None or best[2] < max(MIN_PEAKS, 0.5 * len(peaks)):
        return p0, info
    info.update(refined=True, n_matched=best[2], rms_mhz=best[3])
    params = best[1]

    if kind == "symmetric" and len(peaks_all) > len(peaks):
        params, info = _try_sextic(params, peaks_all, f_max, info)

    return params, info


def _try_sextic(params, peaks_all, f_max, info):
    """
    Adopt the sextic fit only if it explains more of all peaks.
    """

    try:
        j_max = int(min(max(peaks_all[-1] / (2.0 * params[0]) + 3, 3), 150))
        nu = np.sort(symmetric_top_frequencies(*params, j_max, f_max * 1.2))
        pos = np.searchsorted(nu, peaks_all).clip(1, len(nu) - 1)
        d = np.minimum(np.abs(peaks_all - nu[pos - 1]), np.abs(peaks_all - nu[pos]))
        n_base = int(np.sum(d <= 1.0))
        out = _refine_sym_extended(params, peaks_all, f_max)
    except Exception:
        return params, info
    base = dict(info, n_all_peaks=int(len(peaks_all)), n_matched_all=n_base)
    if out is None or out[1] <= n_base:
        return params, dict(base, sextic_fitted=False)
    p6, n6, rms6 = out
    return p6, dict(base, n_peaks_used=int(len(peaks_all)), n_matched=n6, rms_mhz=rms6,
                    sextic_fitted=True, n_matched_all=n6)


def _asym_params(q):
    """
    q = (C, B-C, A-B) with all >= 0  ->  (A, B, C)
    keeps A >= B >= C.
    """
    C = q[0]
    B = C + q[1]
    return B + q[2], B, C

def _asym_assign(peaks, nu_sorted, tol):
    idx = np.abs(peaks[:, None] - nu_sorted[None, :]).argmin(axis=1)
    sel = np.abs(peaks - nu_sorted[idx]) <= tol
    return idx, sel

def _asym_stage(q, d, peaks, f_max, j_max, tol, fit_dist, sigma_obs, d_sigma):
    """
    One assignment + least-squares step.
    q: (C, B-C, A-B); d: 5 distortion constants.
    Returns (q, d) or None.
    """
    A, B, C = _asym_params(q)
    nu = np.sort(asym_a_type_lines(A, B, C, j_max, f_max=f_max * 1.2, dist=tuple(d)))
    if len(nu) < 3:
        return None
    idx, sel = _asym_assign(peaks, nu, tol)
    n_free = 8 if fit_dist else 3
    if sel.sum() < 3:
        return None
    pk, ranks = peaks[sel], idx[sel]

    def unpack(x):
        qq = x[:3]
        dd = x[3:] if fit_dist else d
        return qq, dd

    def resid(x):
        qq, dd = unpack(x)
        a_, b_, c_ = _asym_params(qq)
        nu_ = np.sort(asym_a_type_lines(a_, b_, c_, j_max, f_max=f_max * 1.2,
                                        dist=tuple(dd)))
        r = (pk - nu_[ranks.clip(0, len(nu_) - 1)]) / sigma_obs
        if fit_dist:
            r = np.concatenate([r, x[3:] / d_sigma])      # weak priors on distortion
        return r

    x0 = np.concatenate([q, d]) if fit_dist else q.copy()
    lo = np.r_[1.0, 0.0, 0.0, [-np.inf] * 5][:n_free]
    hi = np.r_[[1e6] * 3, [np.inf] * 5][:n_free]
    x_scale = np.r_[[1000.0] * 3, d_sigma][:n_free]
    try:
        sol = least_squares(resid, np.clip(x0, lo + 1e-9, hi), bounds=(lo, hi),
                            x_scale=x_scale, max_nfev=25)
    except Exception:
        return None
    qq, dd = unpack(sol.x)
    return np.array(qq, dtype=float), np.array(dd, dtype=float)


def _asym_score(q, d, peaks, f_max, tol_final=1.0):
    """
    Score how well the parameters explain the peaks
    """
    A, B, C = _asym_params(q)
    j_max = int(min(max(peaks[-1] / (B + C) + 3, 3), 24))
    nu = np.sort(asym_a_type_lines(A, B, C, j_max, f_max=f_max * 1.2, dist=tuple(d)))
    if len(nu) == 0:
        return None
    res = peaks - nu[np.abs(peaks[:, None] - nu[None, :]).argmin(axis=1)]
    ok = np.abs(res) <= tol_final
    loose = np.abs(res) <= 50.0
    rms = float(np.sqrt(np.mean(res[ok] ** 2))) if ok.any() else float("inf")
    rms_l = float(np.sqrt(np.mean(res[loose] ** 2))) if loose.any() else float("inf")
    return int(ok.sum()), rms, int(loose.sum()), rms_l


def _refine_asym_from(p_start, peaks, f_max, n_iter, sigma_obs, with_dist=True):
    """
    One start
    """
    A0, B0, C0 = p_start
    q = np.array([C0, max(B0 - C0, 0.0), max(A0 - B0, 0.0)], dtype=float)
    d = np.zeros(5)
    for it in range(n_iter):
        pk = peaks[:max(6, int(round(6 * 1.6**it)))] if it < n_iter - 1 else peaks
        A, B, C = _asym_params(q)
        j_max = int(min(max(pk[-1] / (B + C) + 3, 3), 24))
        tol = max(1.0, 0.5 * (B + C) * 0.55**it)
        out = _asym_stage(q, d, pk, f_max, j_max, tol, False, sigma_obs, None)
        if out is None:
            return None
        q = out[0]
    rigid = _asym_score(q, d, peaks, f_max)
    if rigid is None:
        return None
    if with_dist and rigid[1] > 0.3:
        A, B, C = _asym_params(q)
        j_max = int(min(max(peaks[-1] / (B + C) + 3, 3), 24))
        d_sigma = np.array([B * 1e-5, B * 1e-4, A * 1e-4, B * 2e-6, B * 2e-5])
        for tol in (60.0, 25.0, 10.0, 4.0, 1.5, 1.0):
            out = _asym_stage(q, d, peaks, f_max, j_max, tol, True, sigma_obs, d_sigma)
            if out is None:
                break
            q, d = out
    sc = _asym_score(q, d, peaks, f_max)
    if sc is None:
        return None
    A, B, C = _asym_params(q)
    return np.concatenate([[A, B, C], d]), sc


def _refine_asym(p0, peaks, f_max, n_iter=7, sigma_obs=0.05, scan=0.25,
                 n_starts=5):
    """
    Non-linear fit of A, B, C (+  distortion).
    """
    A0, B0, C0 = [float(x) for x in p0]
    f1 = float(peaks[0])
    scales = np.linspace(1.0 - scan, 1.0 + scan, n_starts)
    cands = []
    for s_sum in (f1, B0 + C0):
        for a_f in scales:
            for r_f in scales:
                ratio = np.clip((C0 / B0) * r_f, 0.3, 0.995)
                B = s_sum / (1.0 + ratio)
                C = ratio * B
                A = max(A0 * a_f, B * 1.001)
                out = _refine_asym_from((A, B, C), peaks, f_max, n_iter,
                                        sigma_obs, with_dist=False)
                if out is None:
                    continue
                p, sc = out
                cands.append(((sc[2], -sc[3]), p))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)
    best = None
    seen = []
    for _, p in cands:
        if any(np.allclose(p[:3], q, rtol=1e-3) for q in seen):
            continue
        seen.append(p[:3])
        if len(seen) > 4:
            break
        out = _refine_asym_from(p[:3], peaks, f_max, 4, sigma_obs, with_dist=True)
        if out is None:
            continue
        p2, sc = out
        key = (sc[0], -sc[1]) if sc[0] else (sc[2] * 0, -sc[3])
        # prefer a clean (<=1 MHz) description
        # otherwise the one with most lines within 50 MHz
        full_key = (sc[0], sc[2], -sc[1] if sc[0] else -sc[3])
        if best is None or full_key > best[0]:
            best = (full_key, p2, sc[0], sc[1])
    return best


class TargetScaler:
    """
    Target transform + standardisation for the regression:
      B   -> log(B)               (B spans 500,...,60000 MHz)
      DJ  -> asinh(DJ / 1e-3)     (tiny, non-negative)
      DJK -> asinh(DJK / 1e-2)    (tiny, signed)
    Interface like StandardScaler (fit / transform / inverse_transform)
    """
    _C = (None, 1e-3, 1e-2)

    def __init__(self):
        self._ss = StandardScaler()

    def _fwd(self, Y):
        Y = np.asarray(Y, dtype=float)
        Z = np.empty_like(Y)
        Z[:, 0] = np.log(Y[:, 0])
        for i in range(1, Y.shape[1]):
            Z[:, i] = np.arcsinh(Y[:, i] / self._C[i])
        return Z

    def fit(self, Y):
        self._ss.fit(self._fwd(Y))
        return self

    def fit_transform(self, Y):
        return self._ss.fit_transform(self._fwd(Y))

    def transform(self, Y):
        return self._ss.transform(self._fwd(Y))

    def inverse_transform(self, Z):
        Z = self._ss.inverse_transform(np.asarray(Z, dtype=float))
        Y = np.empty_like(Z)
        Y[:, 0] = np.exp(Z[:, 0])
        for i in range(1, Z.shape[1]):
            Y[:, i] = self._C[i] * np.sinh(Z[:, i])
        Y[:, 1] = np.maximum(Y[:, 1], 0.0)     # DJ >= 0
        return Y


class LogTargetScaler:
    """
    log of every target + standardisation (used for A, B, C; asymmetric molecule).
    Same as TargetScaler but simpler
    """

    def __init__(self):
        self._ss = StandardScaler()

    def fit(self, Y):
        self._ss.fit(np.log(np.asarray(Y, dtype=float)))
        return self

    def fit_transform(self, Y):
        return self._ss.fit_transform(np.log(np.asarray(Y, dtype=float)))

    def transform(self, Y):
        return self._ss.transform(np.log(np.asarray(Y, dtype=float)))

    def inverse_transform(self, Z):
        return np.exp(self._ss.inverse_transform(np.asarray(Z, dtype=float)))


def _new_target_scaler(kind: str = "symmetric"):
    """
    For the dashboard
    """
    try:
        import importlib
        mod = importlib.import_module("rainet_rotational_ml")
        return mod.LogTargetScaler() if kind == "asymmetric" else mod.TargetScaler()
    except Exception:
        print("WARNING: could not import rainet_rotational_ml.")
        return LogTargetScaler() if kind == "asymmetric" else TargetScaler()


# === TRAINING ===

def _make_classifier(seed):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=300,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
        random_state=seed,
    )


def _make_regressor(seed):
    return MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        alpha=1e-5, batch_size=64, learning_rate_init=1e-3, max_iter=800,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
        tol=1e-7, random_state=seed,
    )


def train(output: Path, n_per_class: int, seed: int):
    print(f"Generating {n_per_class} synthetic spectra per class...")
    data = generate_training_set(n_per_class, seed)

    rng = np.random.default_rng(seed)
    split = {}
    for kind in CLASSES:
        n = len(data[kind][1])
        perm = rng.permutation(n)
        cut = int(0.8 * n)
        split[kind] = (perm[:cut], perm[cut:])

    # classifier
    X_tr = np.vstack([data[k][1][split[k][0]] for k in CLASSES])
    y_tr = np.concatenate([[k] * len(split[k][0]) for k in CLASSES])
    X_va = np.vstack([data[k][1][split[k][1]] for k in CLASSES])
    y_va = np.concatenate([[k] * len(split[k][1]) for k in CLASSES])

    classifier_scaler = StandardScaler()
    classifier = _make_classifier(seed)
    classifier.fit(classifier_scaler.fit_transform(X_tr), y_tr)
    pred = classifier.predict(classifier_scaler.transform(X_va))
    print(f"Validation classification accuracy: {accuracy_score(y_va, pred):.3f}"
          "  (synthetic data)")

    # regressors
    regressors = {}
    for kind in CLASSES:
        peaks_all, X, Y = data[kind]
        tr, va = split[kind]
        x_scaler, y_scaler = StandardScaler(), _new_target_scaler(kind)
        model = _make_regressor(seed)
        model.fit(x_scaler.fit_transform(X[tr]), y_scaler.fit_transform(Y[tr]))
        regressors[kind] = {"x_scaler": x_scaler, "y_scaler": y_scaler, "model": model}

        y_pred = y_scaler.inverse_transform(model.predict(x_scaler.transform(X[va])))
        err = np.abs(y_pred - Y[va])
        rel = np.median(err / np.abs(Y[va]), axis=0) * 100
        print(f"[{kind}] ML only      MAE {dict(zip(PARAM_NAMES[kind], np.round(err.mean(0), 5)))}"
              f" | median rel. error {dict(zip(PARAM_NAMES[kind][:1] if kind != 'asymmetric' else PARAM_NAMES[kind], np.round(rel[:1] if kind != 'asymmetric' else rel, 2)))} %")

        n_chk = min(40 if kind == "asymmetric" else 300, len(va))
        ref_err = []
        for i in range(n_chk):
            p_ref, _ = refine_parameters(kind, y_pred[i], peaks_all[va[i]])
            ref_err.append(np.abs(p_ref[:Y.shape[1]] - Y[va][i]))
        ref_err = np.asarray(ref_err)
        relr = np.median(ref_err / np.abs(Y[va][:n_chk]), axis=0) * 100
        print(f"[{kind}] ML + refine  MAE {dict(zip(PARAM_NAMES[kind], np.round(ref_err.mean(0), 5)))}"
              f" | median rel. error {dict(zip(PARAM_NAMES[kind][:1] if kind != 'asymmetric' else PARAM_NAMES[kind], np.round(relr[:1] if kind != 'asymmetric' else relr, 4)))} %"
              f"  (first {n_chk} validation spectra)")

    bundle = {
        "classifier": classifier,
        "classifier_scaler": classifier_scaler,
        "regressors": regressors,
        "n_bins": N_PEAKS,
        "n_peaks": N_PEAKS,
        "f_min": FREQ_MIN_MHZ,
        "f_max": FREQ_MAX_MHZ,
        "classes": CLASSES,
        "param_names": PARAM_NAMES,
        "version": 4,
        "description": "RAINet-inspired linear / symmetric / asymmetric (a-type) prototype (v4)",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"Saved model: {output}")


# === ANALISYS ===

def load_peak_list(path: Path) -> np.ndarray:
    """
    Read one frequency per line or a CSV with a frequency column.
    """
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("The input file is empty.")

    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]

    values = []
    for line_number, line in enumerate(lines, start=1):
        if re.fullmatch(r"[+-]?\d+,\d+", line):
            line = line.replace(",", ".")
        if ";" in line:
            cells = [c.strip() for c in line.split(";")]
        elif "\t" in line:
            cells = [c.strip() for c in line.split("\t")]
        elif "," in line:
            cells = [c.strip() for c in line.split(",")]
        else:
            cells = line.split()

        if not cells or any("frequency" in c.lower() for c in cells):
            continue

        raw_value = cells[0].strip().strip('"')
        if not raw_value:
            continue
        if raw_value.count(",") == 1 and "." not in raw_value:
            raw_value = raw_value.replace(",", ".")
        try:
            values.append(float(raw_value))
        except ValueError as exc:
            raise ValueError(f"Cannot parse frequency on line {line_number}: {line!r}") from exc

    if not values:
        raise ValueError("No frequencies found. Expected one frequency per line.")
    return np.asarray(values, dtype=float)


def _refine_sym_extended(p3, peaks, f_max, sigma_obs=0.05):
    """
    Add the sextic constants HJ, HJK, HKJ to a symmetric fit and use all
    peaks.
    Returns (params6, n_matched_1MHz, rms) or None.
    """
    p3 = np.asarray(p3, dtype=float)
    j_max = int(min(max(peaks[-1] / (2.0 * p3[0]) + 3, 3), 150))
    M = _design_matrix("symmetric_h", j_max)
    sc = 1.0 / np.maximum(np.max(np.abs(M), axis=0), 1e-30)          # column scaling
    prior_sigma = np.array([np.inf, 0.02, 0.05, 1e-5, 1e-4, 1e-3])    # MHz
    p = np.r_[p3, 0.0, 0.0, 0.0]

    def match(p_, pk, tol):
        nu = M @ p_
        v = nu > 0
        order = np.argsort(nu[v])
        nus, Ms = nu[v][order], M[v][order]
        pos = np.searchsorted(nus, pk).clip(1, len(nus) - 1)
        idx = np.where(np.abs(pk - nus[pos - 1]) <= np.abs(pk - nus[pos]), pos - 1, pos)
        res = pk - nus[idx]
        return Ms[idx], res, np.abs(res) <= tol

    n_all = len(peaks)
    sizes = sorted({min(n_all, int(n)) for n in (250, 400, 650, 1000, 1600, n_all)})
    tols = dict(zip(sizes, np.linspace(3.0, 1.0, len(sizes))))
    lo = np.array([1.0, 0.0, -1.0, -np.inf, -np.inf, -np.inf]) / sc
    hi = np.array([1e6, 1.0, 1.0, np.inf, np.inf, np.inf]) / sc
    for n in sizes:
        pk = peaks[:n]
        for _ in range(2):
            Mk, res, ok = match(p, pk, tols[n])
            if ok.sum() < 8:
                return None
            A = Mk[ok] * sc / sigma_obs
            b = pk[ok] / sigma_obs
            rows, rhs = [], []
            for i in range(1, 6):
                r = np.zeros(6)
                r[i] = sc[i] / prior_sigma[i]
                rows.append(r)
                rhs.append(p3[i] / prior_sigma[i] if i < 3 else 0.0)
            sol = lsq_linear(np.vstack([A] + rows), np.r_[b, rhs], bounds=(lo, hi))
            p = sol.x * sc
    _, res, ok = match(p, peaks, 1.0)
    if not ok.any():
        return None
    return p, int(ok.sum()), float(np.sqrt(np.mean(res[ok] ** 2)))


def parameter_uncertainties(kind: str, params, peaks, f_max: float = FREQ_MAX_MHZ,
                            sigma_obs: float = 0.05, tol: float = 1.0):
    """
    Uncertainties of the fitted parameters.

    Returns {"sigma": {name: value}, "poorly_determined": [names]} or None.
    A parameter is flagged when the data barely constrain it
    """
    params = np.asarray(params, dtype=float)
    peaks = _merge_close(np.asarray(peaks, dtype=float))
    peaks = peaks[(peaks > 0) & (peaks <= f_max)]
    n = len(params)
    if len(peaks) < 2:
        return None

    if kind == "asymmetric":
        names = PARAM_NAMES[kind] + (ASYM_DIST_NAMES if n > 3 else [])
        bc = params[1] + params[2]
        j_max = int(min(max(peaks[-1] / bc + 3, 3), 24))
        if n > 3:
            prior = np.r_[[np.inf] * 3, params[1] * 1e-5, params[1] * 1e-4,
                          params[0] * 1e-4, params[1] * 2e-6, params[1] * 2e-5]
        else:
            prior = np.full(3, np.inf)
        steps = np.r_[[0.5] * 3, np.full(max(n - 3, 0), 1.0)] * np.where(
            np.isfinite(prior), np.where(np.arange(n) >= 3, prior * 1e-3, 1.0), 1.0)

        def nu_fn(x):
            return np.sort(asym_a_type_lines(x[0], x[1], x[2], j_max, f_max=f_max * 1.2,
                                             dist=tuple(x[3:]) if n > 3 else None))
    else:
        names = PARAM_NAMES[kind] + (SYM_H_NAMES if kind == "symmetric" and n > 3 else [])
        spacing = 2.0 * params[0]
        j_max = int(min(max(peaks[-1] / spacing + 3, 3), 400 if kind == "linear" else 150))
        dkind = "symmetric_h" if (kind == "symmetric" and n > 3) else kind
        prior = np.array([np.inf, 0.02, 0.05, 1e-5, 1e-4, 1e-3][:n])
        steps = np.array([1.0, 1e-5, 1e-5, 1e-9, 1e-8, 1e-7][:n])

        def nu_fn(x):
            return np.sort(_design_matrix(dkind, j_max) @ x)

    nu0 = nu_fn(params)
    nu0 = nu0[nu0 > 0]
    if len(nu0) == 0:
        return None
    idx = np.abs(peaks[:, None] - nu0[None, :]).argmin(axis=1)
    sel = np.abs(peaks - nu0[idx]) <= tol
    if sel.sum() < 2:
        return None
    ranks = idx[sel]
    n_obs = int(sel.sum())
    r0 = peaks[sel] - nu0[ranks]

    J = np.zeros((n_obs, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = steps[i]
        up, dn = nu_fn(params + dx), nu_fn(params - dx)
        up, dn = up[up > 0], dn[dn > 0]
        k = min(len(up), len(dn))
        rr = np.clip(ranks, 0, k - 1)
        J[:, i] = (up[rr] - dn[rr]) / (2.0 * steps[i])
    J = J / sigma_obs
    info_mat = J.T @ J
    fin = np.isfinite(prior)
    info_mat[fin, fin] += 1.0 / prior[fin] ** 2 if fin.any() else 0.0
    info_mat = info_mat + np.eye(n) * 1e-12 * np.trace(info_mat)
    try:
        cov = np.linalg.inv(info_mat)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info_mat)
    dof = n_obs - n
    chi2_red = float(np.sum((r0 / sigma_obs) ** 2) / dof) if dof > 0 else 1.0
    sig = np.sqrt(np.clip(np.diag(cov), 0, None)) * np.sqrt(max(1.0, chi2_red))

    bad = []
    for i, name in enumerate(names):
        v = abs(params[i])
        if np.isfinite(prior[i]):
            if sig[i] > 0.5 * prior[i] or sig[i] > 0.5 * max(v, 1e-12):
                bad.append(name)
        elif sig[i] > 1.0 or sig[i] > 0.01 * max(v, 1e-12):
            bad.append(name)
    return {"sigma": {nm: float(x) for nm, x in zip(names, sig)},
            "poorly_determined": bad, "n_lines_used": n_obs, "n_parameters": n}


def labelled_lines(kind: str, params: dict, f_max: float = FREQ_MAX_MHZ):
    """
    Lines together with their quantum-number labels, sorted by frequency.
    Returns (frequencies, labels).
    """
    if kind == "linear":
        B = params["B_MHz"]
        j_max = int(min(max(f_max / (2.0 * B) + 3, 3), 400))
        nu = _design_matrix("linear", j_max) @ np.array([B, params["DJ_MHz"]])
        lab = [f"{j} -> {j + 1}" for j in range(j_max)]
    elif kind == "symmetric":
        B = params["B_MHz"]
        j_max = int(min(max(f_max / (2.0 * B) + 3, 3), 150))
        p = [B, params["DJ_MHz"], params["DJK_MHz"]]
        dk = "symmetric"
        if all(k in params for k in SYM_H_NAMES):
            p += [params[k] for k in SYM_H_NAMES]
            dk = "symmetric_h"
        nu = _design_matrix(dk, j_max) @ np.array(p)
        lab = [f"J  {j} -> {j + 1}, K={k}" for j in range(j_max) for k in range(j + 1)]
    elif kind == "asymmetric":
        bc = params["B_MHz"] + params["C_MHz"]
        j_max = int(min(max(f_max / bc + 3, 3), 130))
        dist = None
        if all(k in params for k in ASYM_DIST_NAMES):
            dist = tuple(params[k] for k in ASYM_DIST_NAMES)
        nu, labels = asym_a_type_lines(params["A_MHz"], params["B_MHz"], params["C_MHz"],
                                       j_max, f_max, return_labels=True, dist=dist)
        order = np.argsort(nu)
        return nu[order], [f"{J}({ka},{kc}) -> {J + 1}({ka},{kc + 1})"
                           for J, ka, kc in (labels[i] for i in order)]
    else:
        raise ValueError(f"Unknown molecule type: {kind!r}")
    keep = (nu > 0) & (nu <= f_max)
    nu, lab = nu[keep], [l for l, k in zip(lab, keep) if k]
    order = np.argsort(nu)
    return nu[order], [lab[i] for i in order]


def _fit_kind(bundle, x, peaks, kind, refine):
    reg = bundle["regressors"][kind]
    scaled = reg["model"].predict(reg["x_scaler"].transform(x))
    ml_params = reg["y_scaler"].inverse_transform(scaled)[0]
    params, info = ml_params, {"refined": False}
    if refine:
        params, info = refine_parameters(kind, ml_params, peaks, bundle["f_max"])
    frac = (info["n_matched"] / info["n_peaks_used"]
            if info.get("refined") and info.get("n_peaks_used") else 0.0)
    return ml_params, params, info, frac


def predict_from_peaks(bundle: dict, peaks, refine: bool = True,
                       min_confidence: float = 0.0,
                       min_match_fraction: float = 0.5,
                       force_kind: str | None = None) -> dict:
    """
    Classification -> regression -> (optional) physical refinement.

    The classifier only says which class the lowest peaks look most like.
    If the refined fit of the winning class explains fewer than
    `min_match_fraction` of the peaks, the other classes are tried too (most
    probable first) and the best explained fit wins.
    """
    peaks = np.asarray(peaks, dtype=float)
    x = spectrum_to_features(peaks, bundle["n_bins"], bundle["f_min"],
                             bundle["f_max"]).reshape(1, -1)
    ci = bundle["classifier_scaler"].transform(x)
    kind = str(bundle["classifier"].predict(ci)[0])
    probs = bundle["classifier"].predict_proba(ci)[0]
    confidence = {str(c): float(p) for c, p in zip(bundle["classifier"].classes_, probs)}
    if force_kind is not None:
        if force_kind not in bundle["regressors"]:
            raise ValueError(f"Unknown rotor type: {force_kind!r}")
        kind = force_kind
    first_kind = kind

    warnings = []
    n_valid = int(np.sum(np.isfinite(peaks) & (peaks > 0)))
    if n_valid < 3:
        warnings.append("Fewer than 3 peaks: the assignment is ambiguous.")
    if confidence[kind] < max(min_confidence, 0.0):
        warnings.append("Low classification confidence.")

    ml_params, params, info, frac = _fit_kind(bundle, x, peaks, kind, refine)
    if refine and frac < min_match_fraction and force_kind is None:
        best = (frac, kind, ml_params, params, info)
        for other in sorted(confidence, key=confidence.get, reverse=True):
            if other == kind or other not in bundle["regressors"]:
                continue
            res = _fit_kind(bundle, x, peaks, other, True)
            if res[3] > best[0]:
                best = (res[3], other, res[0], res[1], res[2])
        if best[1] != kind and best[0] >= min_match_fraction:
            warnings.append(f"The classifier chose '{kind}' but the fit of '{best[1]}' "
                            f"explains {best[0]*100:.0f}% of the peaks; using '{best[1]}'.")
        if best[1] != kind or best[0] > frac:
            _, kind, ml_params, params, info = best
    names = PARAM_NAMES[kind]
    if refine and not info["refined"]:
        warnings.append("Refinement failed (too few peaks matched the ML model): "
                        "showing the raw ML estimate.")

    ext = (ASYM_DIST_NAMES if kind == "asymmetric" and len(params) > 3 else
           SYM_H_NAMES if kind == "symmetric" and len(params) > 3 else [])
    uncertainty = None
    if refine and info.get("refined"):
        try:
            uncertainty = parameter_uncertainties(kind, params, peaks, bundle["f_max"])
        except Exception:
            uncertainty = None
        if uncertainty and uncertainty["n_lines_used"] <= uncertainty["n_parameters"] + 1:
            warnings.append(
                f"Only {uncertainty['n_lines_used']} lines for "
                f"{uncertainty['n_parameters']} fitted parameters: the fit is "
                "(nearly) exactly determined, so a tiny rms is not a quality measure.")
    return {
        "spectrum_type": kind,
        "classifier_choice": first_kind,
        "forced_kind": force_kind is not None,
        "classification_probabilities": confidence,
        "number_of_input_peaks": int(n_valid),
        "ml_parameters": {n: float(v) for n, v in zip(names, ml_params)},
        "parameters": {n: float(v) for n, v in zip(names + ext, params)},
        "refinement": info,
        "uncertainties": uncertainty,
        "warnings": warnings,
    }


def analyze(model_path: Path, peak_path: Path, output_json: Path | None = None,
            refine: bool = True):
    bundle = joblib.load(model_path)
    peaks = load_peak_list(peak_path)
    result = predict_from_peaks(bundle, peaks, refine=refine)
    result["note"] = ("Preliminary estimate only.")
    print(json.dumps(result, indent=2))
    if output_json:
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


# === BENCHMARK ===

# Literature constants quoted in Zaleski & Prozument (2018),
# converted to MHz.
BENCHMARK = [
    ("OCS", "linear", [6081.492106, 0.001301367]),
    ("HCN", "linear", [44315.9757, 0.08724]),
    ("HCCCN", "linear", [4549.058588, 0.0005442223]),
    ("CH3CCH", "symmetric", [8545.876883, 0.002939344, 0.16341306]),
    ("CH3CN", "symmetric", [9198.899134, 0.003807528, 0.17740796]),
    ("CF3I", "symmetric", [1523.28226, 0.00016462, 0.0009925]),
    # Asymmetric: approximate ground-state A, B, C
    # (MHz, quoted from memory, rigid-rotor test only - distortion ignored).
    ("C2H3CN", "asymmetric", [49850.7, 4971.2, 4513.8]),
    ("C2H5CN", "asymmetric", [27663.6, 4714.2, 4235.4]),
    ("NH2CHO", "asymmetric", [72717.7, 11373.5, 9833.2]),
    ("HCOOH", "asymmetric", [77512.2, 12055.1, 10416.1]),
]


def benchmark(model_path: Path, noise_mhz: float = 0.02, seed: int = 1):
    bundle = joblib.load(model_path)
    rng = np.random.default_rng(seed)
    print(f"Idealised literature test, line noise {noise_mhz} MHz\n")
    print(f"{'molecule':8s} {'class':10s} {'B_exp':>12s} {'B_ML':>12s} {'B_refined':>12s}  note")
    for name, kind, p in BENCHMARK:
        if kind == "linear":
            lines = linear_frequencies(p[0], p[1], 40)
        elif kind == "symmetric":
            lines = symmetric_top_frequencies(p[0], p[1], p[2], 22)
        else:
            lines = asym_a_type_lines(p[0], p[1], p[2], 12,
                                      e_lower_max=3.5 * KB_MHZ_PER_K * 30.0)
        peaks = _merge_close(lines + rng.normal(0.0, noise_mhz, len(lines)))
        r = predict_from_peaks(bundle, peaks, refine=True)
        note = "" if r["spectrum_type"] == kind else f"MISCLASSIFIED as {r['spectrum_type']}"
        i_b = 1 if kind == "asymmetric" else 0
        print(f"{name:8s} {r['spectrum_type']:10s} {p[i_b]:12.4f} "
              f"{r['ml_parameters']['B_MHz']:12.4f} {r['parameters']['B_MHz']:12.4f}  {note}")


def main():
    parser = argparse.ArgumentParser(
        description="RAINet-inspired ML analysis of rotational peak lists")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Generate synthetic spectra and train networks")
    p_train.add_argument("--samples", type=int, default=20000,
                         help="Synthetic examples per class (default: 20000)")
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--output", type=Path, default=Path("rainet_model.joblib"))

    p_an = sub.add_parser("analyze", help="Analyze an experimental peak list")
    p_an.add_argument("peak_file", type=Path)
    p_an.add_argument("--model", type=Path, default=Path("rainet_model.joblib"))
    p_an.add_argument("--json", type=Path, default=None)
    p_an.add_argument("--no-refine", action="store_true",
                      help="Show the raw ML estimate without least-squares refinement")

    p_b = sub.add_parser("benchmark", help="Test on literature constants from the paper")
    p_b.add_argument("--model", type=Path, default=Path("rainet_model.joblib"))

    args = parser.parse_args()
    if args.command == "train":
        train(args.output, args.samples, args.seed)
    elif args.command == "analyze":
        analyze(args.model, args.peak_file, args.json, refine=not args.no_refine)
    elif args.command == "benchmark":
        benchmark(args.model)


if __name__ == "__main__":
    main()
