# main_real.py
import numpy as np
import matplotlib.pyplot as plt

from sgld_gibbs_fixed_cov import run  # we only need run, not generate_data

from svi import svi_gmm_diag, svi_gmm_diag_flat, predict_labels_from_svi, mu_marginal_student_t_params, sample_posterior_means

from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, t as student_t, gaussian_kde

# ---------------------------------------------------------------------
# Posterior comparison: SVI vs SGLD (mu only)
# ---------------------------------------------------------------------

def get_param_color_map(K, d):
    """Return a consistent color mapping for parameters (k,j).

    This matches the GMM_syn plotting style: fixed, repeatable colors across figures.

    Parameters
    ----------
    K : int
        Number of mixture components.
    d : int
        Dimension of each component mean.

    Returns
    -------
    colors : dict
        Dictionary mapping (k, j) -> matplotlib color.
    """
    # Use Matplotlib's default color cycle for consistency across environments.
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        # Fallback cycle if Matplotlib returns an empty list (rare).
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    colors = {}
    idx = 0
    for k in range(int(K)):
        for j in range(int(d)):
            colors[(k, j)] = cycle[idx % len(cycle)]
            idx += 1
    return colors

def plot_compare_mu_posteriors_svi_vs_sgld(
    svi_params,
    samples_means_sgld,
    true_means,
    *,
    tail=3000,
    n_cols=1,
    max_params=5,
    show_hist=False,
    hist_bins=30,
    hist_alpha=0.25,
    title_prefix="",
):
    """Compatibility wrapper.

    Main calls `plot_compare_mu_posteriors_svi_vs_sgld(...)`. This wrapper extracts
    the SVI Gaussian approximation inputs and forwards to
    `plot_posterior_mu_sgld_vs_svi`, which matches the GMM_syn plotting style
    (solid = SGLD Gaussian fit, dashed = SVI Gaussian approx).
    """
    import numpy as np

    # SVI mean for mu
    mu_svi = np.asarray(svi_params["m"], dtype=float)

    # Try to obtain a per-(k,j) scale for q(mu_{k,j}) and an effective Nk.
    # We keep this robust to different SVI parameterizations.
    K, d = mu_svi.shape

    sigma_svi = None
    Nk_svi = None

    # Preferred: use helper if available (imported at top)
    try:
        out = mu_marginal_student_t_params(svi_params)
        # Common patterns we support:
        #   (loc, scale, dof, Nk)
        #   {"loc":..., "scale":..., "dof":..., "Nk":...}
        if isinstance(out, dict):
            scale = out.get("scale", None)
            nk = out.get("Nk", None)
        else:
            # tuple/list
            scale = out[1] if len(out) >= 2 else None
            nk = out[3] if len(out) >= 4 else None

        if scale is not None:
            scale = np.asarray(scale, dtype=float)
            # Accept either (K,d) or broadcastable
            if scale.shape == (K, d):
                sigma_svi = scale
            else:
                sigma_svi = np.broadcast_to(scale, (K, d)).copy()

        if nk is not None:
            nk = np.asarray(nk, dtype=float)
            if nk.shape == (K,):
                Nk_svi = nk
            else:
                Nk_svi = np.broadcast_to(nk, (K,)).copy()

    except Exception:
        pass

    # Fallbacks if helper is unavailable or returns unexpected shapes
    if sigma_svi is None:
        # Use a conservative constant scale so the plot still renders.
        sigma_svi = np.ones((K, d), dtype=float)

    if Nk_svi is None:
        Nk_svi = np.ones((K,), dtype=float)

    return plot_posterior_mu_sgld_vs_svi(
        samples_means_sgld,
        mu_svi,
        sigma_svi,
        Nk_svi,
        true_means,
        tail_T=tail,
        max_params=max_params,
        n_cols=n_cols,
        title_prefix=title_prefix,
        show_hist=bool(locals().get("show_hist", False)),
        hist_bins=int(locals().get("hist_bins", 30)),
        hist_alpha=float(locals().get("hist_alpha", 0.25)),
    )

def _permute_svi_params(svi_params, perm):
    """Return a shallow-copied svi_params with component axis permuted."""
    out = dict(svi_params)
    for key in ["m", "beta", "a", "b"]:
        if key in out:
            out[key] = np.asarray(out[key])[perm]
    if "alpha" in out:
        out["alpha"] = np.asarray(out["alpha"])[perm]
    if "pi_fixed" in out:
        out["pi_fixed"] = np.asarray(out["pi_fixed"])[perm]
    return out


def plot_posterior_mu_sgld_vs_svi(
    samples_means_sgld,
    mu_svi,
    sigma_svi,
    Nk_svi,
    true_means,
    *,
    tail_T=3000,
    max_params=5,
    n_cols=1,
    grid_std=4.0,
    show_hist=False,
    hist_bins=30,
    hist_alpha=0.25,
    title_prefix="",
):
    """Posterior comparison for mu: Gaussian approx (SGLD tail) vs Gaussian approx (SVI).

    Style (match GMM_syn):
      - grid subplots
      - consistent color per (k,j)
      - solid = SGLD Gaussian fit, dashed = SVI Gaussian approx
      - legend only once
      - optional histogram of SGLD tail samples (density)
      - no tail-mean markers
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    samples_means_sgld = np.asarray(samples_means_sgld, dtype=float)
    mu_svi = np.asarray(mu_svi, dtype=float)
    sigma_svi = np.asarray(sigma_svi, dtype=float)
    Nk_svi = np.asarray(Nk_svi, dtype=float)
    true_means = np.asarray(true_means, dtype=float)

    T = samples_means_sgld.shape[0]
    tail_len = min(int(tail_T), int(T))
    tail = samples_means_sgld[-tail_len:]

    K, d = mu_svi.shape

    # Prefer a small set of representative entries; then fill row-major.
    requested = [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1)]
    exclude = {(2, 0), (0, 0), (0, 2), (1, 2)}

    idxs = []
    used = set()
    for (k, j) in requested:
        if 0 <= k < K and 0 <= j < d and (k, j) not in used and (k, j) not in exclude:
            idxs.append((k, j))
            used.add((k, j))

    max_params = min(int(max_params), K * d)
    for (k, j) in [(kk, jj) for kk in range(K) for jj in range(d)]:
        if len(idxs) >= max_params:
            break
        if (k, j) not in used and (k, j) not in exclude:
            idxs.append((k, j))
            used.add((k, j))

    # consistent colors (you already have this helper)
    colors = get_param_color_map(K, d)

    n = len(idxs)

    # Grid layout (user-requested default: 4x5 when max_params=20 and n_cols=5)
    n_cols = int(max(1, n_cols))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax_idx, (k, j) in enumerate(idxs):
        ax = axes_flat[ax_idx]
        c = colors[(k, j)]

        series = np.asarray(tail[:, k, j], dtype=float)
        # Optional: histogram of SGLD tail samples (density)
        if show_hist:
            ax.hist(
                series,
                bins=int(hist_bins),
                density=True,
                color=c,
                alpha=float(hist_alpha),
                edgecolor="none",
                label="SGLD tail hist" if ax_idx == 0 else None,
            )

        # SGLD Gaussian fit
        m_sgld = float(np.mean(series))
        s_sgld = float(np.std(series, ddof=1)) if tail_len > 1 else 0.0
        if (not np.isfinite(s_sgld)) or s_sgld <= 0:
            s_sgld = 1e-6

        # SVI Gaussian approx: mu_{k,j} ~ N(m, sigma^2 / Nk)
        m_svi = float(mu_svi[k, j])
        nk = max(float(Nk_svi[k]), 1.0)
        s2_svi = float((sigma_svi[k, j] ** 2) / nk)
        s_svi = float(np.sqrt(max(s2_svi, 1e-12)))

        # shared x-range
        std_ref = max(s_sgld, s_svi, 1e-6)
        x_min = min(m_sgld, m_svi) - grid_std * std_ref
        x_max = max(m_sgld, m_svi) + grid_std * std_ref
        xs = np.linspace(x_min, x_max, 400)

        # curves
        ax.plot(
            xs, norm.pdf(xs, loc=m_sgld, scale=s_sgld),
            color=c, linewidth=2.0,
            label="SGLD Gaussian fit" if ax_idx == 0 else None,
        )
        ax.plot(
            xs, norm.pdf(xs, loc=m_svi, scale=s_svi),
            color=c, linewidth=2.0, linestyle="--",
            label="SVI posterior (Gaussian approx)" if ax_idx == 0 else None,
        )

        ax.set_title(f"{title_prefix}mu[{k},{j}]  tail={tail_len}")
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[n:]:
        ax.axis("off")

    if n > 0:
        axes_flat[0].legend(frameon=True, fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig

# ---------------------------------------------------------------------
# Rank-uniformity (PIT) calibration helpers
# ---------------------------------------------------------------------

def _rank_uniformity_pvalues(samples, truth):
    """Compute p-values p_j = mean_t [ samples_tj > truth_j ] for vector truth.

    samples: (T, P)
    truth:   (P,)
    returns: (P,)
    """
    samples = np.asarray(samples)
    truth = np.asarray(truth)
    return np.mean(samples > truth[None, :], axis=0)


def plot_rank_uniformity(pvals, title="Rank-uniformity calibration"):
    """Plot sorted p-values against Uniform(0,1) line."""
    pvals = np.asarray(pvals)
    pvals = pvals[np.isfinite(pvals)]
    p_sorted = np.sort(pvals)
    n = p_sorted.size
    if n == 0:
        print("[rank-uniformity] No finite p-values to plot.")
        return

    u = (np.arange(1, n + 1) - 0.5) / n
    plt.figure(figsize=(5, 5))
    plt.plot(u, p_sorted, marker="o", markersize=3, linewidth=1)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Uniform quantiles")
    plt.ylabel("Sorted p-values")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# --- Helper: plot two rank-uniformity curves for SVI vs SGLD in one plot ---
def plot_rank_uniformity_compare(pvals_a, pvals_b, label_a="SVI", label_b="SGLD", title="Rank-uniformity comparison"):
    """Plot two rank-uniformity curves on the same axes."""
    pvals_a = np.asarray(pvals_a)
    pvals_b = np.asarray(pvals_b)

    pvals_a = pvals_a[np.isfinite(pvals_a)]
    pvals_b = pvals_b[np.isfinite(pvals_b)]

    plt.figure(figsize=(5, 5))

    if pvals_a.size > 0:
        pa = np.sort(pvals_a)
        na = pa.size
        ua = (np.arange(1, na + 1) - 0.5) / na
        plt.plot(ua, pa, marker="o", markersize=3, linewidth=1, label=label_a)

    if pvals_b.size > 0:
        pb = np.sort(pvals_b)
        nb = pb.size
        ub = (np.arange(1, nb + 1) - 0.5) / nb
        plt.plot(ub, pb, marker="s", markersize=3, linewidth=1, label=label_b)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black", label="Ideal")
    plt.xlabel("Uniform quantiles")
    plt.ylabel("Sorted p-values")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def rank_uniformity_mu_sgld(samples_means, true_means, tail=3000):
    """Rank-uniformity p-values for SGLD mu using tail samples, matched to true_means."""
    samples_means = np.asarray(samples_means)  # (T,K,d)
    T, K, d = samples_means.shape
    tail_len = min(int(tail), T)
    tail_samps = samples_means[-tail_len:, :, :]            # (tail_len,K,d)
    tail_mean = tail_samps.mean(axis=0)                     # (K,d)

    # Match to true_means
    perm = match_clusters(true_means, tail_mean)
    tail_samps = tail_samps[:, perm, :]

    # Flatten parameters (K*d)
    S = tail_samps.reshape(tail_len, K * d)
    truth = true_means.reshape(K * d)

    return _rank_uniformity_pvalues(S, truth)


def rank_uniformity_mu_svi(svi_params, true_means, n_samples=3000, seed=0):
    """Rank-uniformity p-values for SVI mu by sampling from q(mu), matched to true_means."""
    K, d = svi_params["m"].shape

    # Match SVI components to true_means using variational mean
    perm = match_clusters(true_means, np.asarray(svi_params["m"], dtype=float))
    svi_perm = _permute_svi_params(svi_params, perm)

    # Sample mu ~ q(mu)
    mu_samps = sample_posterior_means(svi_perm, n_samples=n_samples, seed=seed)  # (S,K,d)
    S = mu_samps.reshape(n_samples, K * d)
    truth = true_means.reshape(K * d)

    return _rank_uniformity_pvalues(S, truth)


import pandas as pd

# ---------------------------------------------------------------------
# Helper: compute GMM parameters directly from provided true labels
# ---------------------------------------------------------------------

def gmm_params_from_true_labels(X, y, K, *, ridge=1e-6):
    """Compute (means, diag covs, weights) from provided labels.

    X: (N,d) standardized feature space
    y: (N,) labels in {0,...,K-1}

    Returns
    -------
    means : (K,d)
    covs  : (K,d,d) diagonal cov matrices
    weights : (K,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    N, d = X.shape
    means = np.zeros((K, d), dtype=float)
    covs = np.zeros((K, d, d), dtype=float)
    weights = np.zeros(K, dtype=float)

    for k in range(K):
        Xk = X[y == k]
        nk = Xk.shape[0]
        if nk == 0:
            means[k] = 0.0
            covs[k] = np.eye(d)
            weights[k] = 1e-12
        else:
            means[k] = Xk.mean(axis=0)
            var = Xk.var(axis=0) + ridge
            covs[k] = np.diag(var)
            weights[k] = nk / float(N)

    # renormalize weights
    s = weights.sum()
    if s <= 0:
        weights = np.ones(K) / K
    else:
        weights = weights / s

    return means, covs, weights

np.random.seed(20)

#
# Choose which real dataset to run:
# one of: "wine", "seeds", "faithful", "digits", "pca4type", "flowcytometry"
DATASET = "flowcytometry"  # change back to "digits" or others when desired

SVI_FLAT_PRIOR = True  # True -> match SGLD (fixed pi=1/K, weak/flat priors)
RUN_BASELINE_GIBBS = True
# Shared initialization for BOTH SVI and SGLD
# Options: "gmm_em" (recommended), "kmeans", "random"
INIT_METHOD = "gmm_em"

# If True and dataset provides true labels, use label-derived init; otherwise use INIT_METHOD.
USE_TRUE_LABEL_INIT = False

# How to build pseudo-true parameters for matching/calibration
# Options: "gmm_em" or "kmeans"
PSEUDO_TRUE_METHOD = "gmm_em"

# Seeds
KMEANS_SEED = 42
EM_SEED = 42
def load_pca4type_dataset(path="pca_4type50.csv"):
    """Custom dataset where the cluster label is in the last column.

    For this dataset, we:
      - read the CSV from `path`,
      - take the first 8 columns as input features X,
      - take the last column as the cluster label y,
      - standardize X,
      - DO NOT perform PCA (we stay in this 8D feature space),
      - build pseudo-true GMM parameters via k-means.
    """
    df = pd.read_csv(path)

    # Features: first 8 columns
    X = df.iloc[:, 1:9].values

    # Labels: last column (cluster index)
    y_raw = df.iloc[:, -1].values
    unique_labels = np.unique(y_raw)
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[v] for v in y_raw])

    # Standardize features
    X = StandardScaler().fit_transform(X)

    K = len(np.unique(y))
    N, d = X.shape

    # Pseudo-true parameters: follow PSEUDO_TRUE_METHOD
    if PSEUDO_TRUE_METHOD == "gmm_em":
        mu0_pt, sigma0_pt, pi0_pt, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
        true_means = mu0_pt
        true_covs = np.array([np.diag(s**2) for s in sigma0_pt])
        true_weights = pi0_pt
    else:
        true_means, true_covs, true_weights = gmm_params_from_true_labels(X, y, K)
        # still compute an EM label baseline for reference
        _, _, _, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)

    return {
        "name": "pca4type",
        "X": X,
        "z_true": y,
        "z_km": z_km,
        "K": K,
        "d": d,
        "N": N,
        "true_means": true_means,
        "true_covs": true_covs,
        "true_weights": true_weights,
    }


# ---------------------------------------------------------------------
# Flow cytometry dataset loader
# ---------------------------------------------------------------------
def load_flowcytometry_dataset(path="flowcytometry.csv"):
    """
    Custom flow cytometry dataset.

    We:
      - read the CSV from `path`,
      - use only the fluorescence channels FL1.H, FL2.H, FL3.H, FL4.H as features X,
      - take the last column (or 'label' column if present) as the cluster label y,
      - standardize X,
      - DO NOT perform PCA (we stay in this 4D feature space),
      - build pseudo-true GMM parameters via k-means.
    """
    df = pd.read_csv(path)

    # Features: only the four fluorescence channels
    if all(col in df.columns for col in ["FL1.H", "FL2.H", "FL3.H", "FL4.H"]):
        X = df[["FL1.H", "FL2.H", "FL3.H", "FL4.H"]].values
    else:
        # Fallback: assume FL1.H–FL4.H are the 3rd–6th columns as in the original sample
        # (FSC.H, SSC.H, FL1.H, FL2.H, FL3.H, FL4.H, ...)
        X = df.iloc[:, 2:6].values

    # Labels: use 'label' column if present, otherwise last column
    if "label" in df.columns:
        y_raw = df["label"].values
    else:
        y_raw = df.iloc[:, -1].values

    unique_labels = np.unique(y_raw)
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[v] for v in y_raw])

    # Standardize features
    X = StandardScaler().fit_transform(X)

    K = len(np.unique(y))
    N, d = X.shape

    # Pseudo-true parameters: follow PSEUDO_TRUE_METHOD
    if PSEUDO_TRUE_METHOD == "gmm_em":
        mu0_pt, sigma0_pt, pi0_pt, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
        true_means = mu0_pt
        true_covs = np.array([np.diag(s**2) for s in sigma0_pt])
        true_weights = pi0_pt
    else:
        true_means, true_covs, true_weights = gmm_params_from_true_labels(X, y, K)
        # still compute an EM label baseline for reference
        _, _, _, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)

    return {
        "name": "flowcytometry",
        "X": X,
        "z_true": y,
        "z_km": z_km,
        "K": K,
        "d": d,
        "N": N,
        "true_means": true_means,
        "true_covs": true_covs,
        "true_weights": true_weights,
    }


# ---------------------------------------------------------------------
# Helpers: build pseudo-true GMM params from data via k-means
# ---------------------------------------------------------------------
def build_pseudo_true_gmm(X, K):
    """
    Given data X (N, d) and number of clusters K, run k-means and compute
    'pseudo-true' means, diagonal covariances and weights.
    """
    N, d = X.shape

    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    z = km.fit_predict(X)

    means = np.zeros((K, d))
    covs = np.zeros((K, d, d))
    weights = np.zeros(K)

    for k in range(K):
        Xk = X[z == k]
        if Xk.shape[0] == 0:
            # Empty cluster – just set something small
            means[k] = km.cluster_centers_[k]
            covs[k] = np.eye(d) * 1e-2
            weights[k] = 1e-3
        else:
            means[k] = Xk.mean(axis=0)
            # diagonal covariance
            var = Xk.var(axis=0) + 1e-6  # small ridge
            covs[k] = np.diag(var)
            weights[k] = Xk.shape[0] / N

    # renormalize weights (in case of tiny adjustments)
    weights = weights / weights.sum()

    return means, covs, weights, z


# ---------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------
def load_wine_dataset():
    data = load_wine()
    X = data.data
    y = data.target  # 0,1,2
    K = len(np.unique(y))

    # standardize
    X = StandardScaler().fit_transform(X)

    N, d = X.shape
    if d > 8:
        pca = PCA(n_components=8, random_state=42)
        X = pca.fit_transform(X)
        N, d = X.shape
    # Pseudo-true parameters: follow PSEUDO_TRUE_METHOD
    if PSEUDO_TRUE_METHOD == "gmm_em":
        mu0_pt, sigma0_pt, pi0_pt, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
        true_means = mu0_pt
        true_covs = np.array([np.diag(s**2) for s in sigma0_pt])
        true_weights = pi0_pt
    else:
        true_means, true_covs, true_weights = gmm_params_from_true_labels(X, y, K)
        _, _, _, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
    return {
        "name": "wine",
        "X": X,
        "z_true": y,
        "z_km": z_km,
        "K": K,
        "d": d,
        "N": N,
        "true_means": true_means,
        "true_covs": true_covs,
        "true_weights": true_weights,
    }


def load_seeds_dataset(path="Seed_Data.csv"):
    """ 
    Expects Seeds dataset saved as a CSV (e.g. 'Seed_Data.csv')
    in the current directory (features in all but last column, label in last column).
    """
    df = pd.read_csv(path)
    # Assume last column is the class label
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    # Map labels to 0,1,2 if they are not already
    unique_labels = np.unique(y_raw)
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[v] for v in y_raw])

    # standardize
    X = StandardScaler().fit_transform(X)

    K = len(np.unique(y))
    N, d = X.shape
    if d > 8:
        pca = PCA(n_components=8, random_state=42)
        X = pca.fit_transform(X)
        N, d = X.shape

    # Pseudo-true parameters: follow PSEUDO_TRUE_METHOD
    if PSEUDO_TRUE_METHOD == "gmm_em":
        mu0_pt, sigma0_pt, pi0_pt, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
        true_means = mu0_pt
        true_covs = np.array([np.diag(s**2) for s in sigma0_pt])
        true_weights = pi0_pt
    else:
        true_means, true_covs, true_weights = gmm_params_from_true_labels(X, y, K)
        _, _, _, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
    return {
        "name": "seeds",
        "X": X,
        "z_true": y,
        "z_km": z_km,
        "K": K,
        "d": d,
        "N": N,
        "true_means": true_means,
        "true_covs": true_covs,
        "true_weights": true_weights,
    }


def load_faithful_dataset(path="faithful.csv"):
    """
    Expects Old Faithful dataset (eruption duration, waiting time) saved as
    'faithful.csv' in the current directory with columns like:
    'eruptions', 'waiting' or similar.

    If your column names differ, just adjust the column selection below.
    """
    df = pd.read_csv(path)
    # Try reasonable column names; change if your CSV uses others.
    if "eruptions" in df.columns and "waiting" in df.columns:
        X = df[["eruptions", "waiting"]].values
    else:
        # fallback: just take first two numeric columns
        X = df.select_dtypes(include="number").iloc[:, :2].values

    # standardize
    X = StandardScaler().fit_transform(X)

    # 2 clusters is standard for Old Faithful
    K = 2
    N, d = X.shape
    if PSEUDO_TRUE_METHOD == "gmm_em":
        mu0_pt, sigma0_pt, pi0_pt, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
        true_means = mu0_pt
        true_covs = np.array([np.diag(s**2) for s in sigma0_pt])
        true_weights = pi0_pt
    else:
        true_means, true_covs, true_weights, z_km = build_pseudo_true_gmm(X, K)

    # no true labels; we just use k-means/EM labels as a reference
    z_true = z_km.copy()

    return {
        "name": "faithful",
        "X": X,
        "z_true": z_true,
        "z_km": z_km,
        "K": K,
        "d": d,
        "N": N,
        "true_means": true_means,
        "true_covs": true_covs,
        "true_weights": true_weights,
    }


def load_digits_dataset(digits=(0, 1, 2), max_per_digit=1000):
    """
    Handwritten digits from sklearn (8x8 images -> 64D features).

    Restrict to a subset of digits, e.g. (0,1,2), to keep K small and
    size manageable for SGLD.
    """
    data = load_digits()
    X_all = data.data  # (1797, 64)
    y_all = data.target

    mask = np.isin(y_all, digits)
    X = X_all[mask]
    y = y_all[mask]
    # Relabel to 0...(K-1) based on the chosen digits
    unique_digits = np.sort(np.unique(y))
    remap = {d: i for i, d in enumerate(unique_digits)}
    y = np.array([remap[v] for v in y])

    # optional subsample per digit
    X_sub = []
    y_sub = []
    for k in range(len(unique_digits)):
        idx = np.where(y == k)[0]
        if max_per_digit is not None and len(idx) > max_per_digit:
            idx = np.random.choice(idx, size=max_per_digit, replace=False)
        X_sub.append(X[idx])
        y_sub.append(y[idx])
    X = np.vstack(X_sub)
    y = np.concatenate(y_sub)

    # standardize
    X = StandardScaler().fit_transform(X)

    K = len(np.unique(y))
    N, d = X.shape
    if d > 8:
        pca = PCA(n_components=8, random_state=42)
        X = pca.fit_transform(X)
        N, d = X.shape

    # Pseudo-true parameters: follow PSEUDO_TRUE_METHOD
    if PSEUDO_TRUE_METHOD == "gmm_em":
        mu0_pt, sigma0_pt, pi0_pt, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
        true_means = mu0_pt
        true_covs = np.array([np.diag(s**2) for s in sigma0_pt])
        true_weights = pi0_pt
    else:
        true_means, true_covs, true_weights = gmm_params_from_true_labels(X, y, K)
        _, _, _, z_km = fit_em_gmm_init(X, K, seed=EM_SEED)
    return {
        "name": "digits",
        "X": X,
        "z_true": y,
        "z_km": z_km,
        "K": K,
        "d": d,
        "N": N,
        "true_means": true_means,
        "true_covs": true_covs,
        "true_weights": true_weights,
    }
# ---------------------------------------------------------------------
# Helpers: build pseudo-true GMM params from data via k-means
# ---------------------------------------------------------------------
def fit_em_gmm_init(X, K, *, seed=42, max_iter=200, n_init=10, reg_covar=1e-6):
    gm = GaussianMixture(
        n_components=K,
        covariance_type="diag",
        max_iter=max_iter,
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=seed,
        init_params="kmeans",
    )
    gm.fit(X)
    mu0 = np.asarray(gm.means_, dtype=float)
    sigma0 = np.sqrt(np.asarray(gm.covariances_, dtype=float))  # diag -> (K,d)
    pi0 = np.asarray(gm.weights_, dtype=float)
    z0 = gm.predict(X)
    return mu0, sigma0, pi0, z0


def load_real_dataset(which):
    if which == "wine":
        return load_wine_dataset()
    elif which == "seeds":
        return load_seeds_dataset()
    elif which == "faithful":
        return load_faithful_dataset()
    elif which == "digits":
        return load_digits_dataset(digits=(0, 1, 2), max_per_digit=800)
    elif which == "pca4type":
        return load_pca4type_dataset()
    elif which == "flowcytometry":
        return load_flowcytometry_dataset()
    else:
        raise ValueError(f"Unknown dataset: {which}")


# ---------------------------------------------------------------------
# Trajectory plots for mu and sigma, with optional cluster matching
# ---------------------------------------------------------------------
def match_clusters(true_means, est_means, true_sigma=None, est_sigma=None, lambda_sigma=1.0):
    """
    Match clusters between true and estimated parameters using Hungarian algorithm.
    Uses L2 distance between means, plus (optionally) L2 distance between sigmas.

    Returns perm of shape (K,) such that perm[k_true] = k_est, i.e.
    est_means[perm[k_true]] is matched to true_means[k_true].
    """
    K = true_means.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            c = np.linalg.norm(true_means[i] - est_means[j])
            if true_sigma is not None and est_sigma is not None:
                c += lambda_sigma * np.linalg.norm(true_sigma[i] - est_sigma[j])
            cost[i, j] = c
    row_ind, col_ind = linear_sum_assignment(cost)
    # Build perm so that perm[k_true] = k_est
    perm = np.zeros(K, dtype=int)
    for i, j in zip(row_ind, col_ind):
        perm[i] = j
    return perm

# ---------------------------------------------------------------------
# Baseline full-data Gibbs sampler (diag-cov GMM) + 3-way mu overlays
# ---------------------------------------------------------------------

def _sample_inv_gamma(rng, a, b):
    """Sample Inv-Gamma(a,b). If Y ~ Gamma(a, scale=1/b), then 1/Y ~ Inv-Gamma(a,b)."""
    a = float(a); b = float(b)
    y = rng.gamma(shape=max(a, 1e-12), scale=1.0 / max(b, 1e-12))
    return 1.0 / max(y, 1e-300)

def gibbs_gmm_diag_fullposterior(
    X, K, *,
    iters=3000, burnin=1000, thin=5, seed=0,
    ref_means=None,
    init_mu0=None, init_sigma0=None, init_pi0=None,
    # weak/flat-ish conjugate prior (Normal-InvGamma per dim)
    alpha0=1.0,      # Dirichlet concentration (symmetric)
    m0=0.0,
    kappa0=1e-3,
    a0=1e-3,
    b0=1e-3,
):
    """
    Full-data Gibbs baseline for diagonal-covariance GMM.

    pi ~ Dir(alpha0 * 1)
    z_i | pi ~ Cat(pi)
    x_i | z_i=k ~ N(mu_k, diag(sigma_k^2))

    Prior per k,j:
      sigma_{k,j}^2 ~ Inv-Gamma(a0,b0)
      mu_{k,j} | sigma_{k,j}^2 ~ N(m0, sigma_{k,j}^2 / kappa0)

    If ref_means is provided, each stored sample is permuted to match ref_means.
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    rng = np.random.default_rng(int(seed))

    # init
    if init_mu0 is None:
        idx = rng.choice(N, size=K, replace=False)
        mu = X[idx].copy()
    else:
        mu = np.asarray(init_mu0, dtype=float).copy()

    if init_sigma0 is None:
        sigma = np.ones((K, d), dtype=float)
    else:
        sigma = np.asarray(init_sigma0, dtype=float).copy()
        if sigma.ndim == 1:
            sigma = np.broadcast_to(sigma[None, :], (K, d)).copy()

    if init_pi0 is None:
        pi = np.ones(K, dtype=float) / float(K)
    else:
        pi = np.asarray(init_pi0, dtype=float).copy()
        pi = pi / max(pi.sum(), 1e-300)

    z = hard_assign_gmm(X, mu, sigma, pi)

    mu_samps, sigma_samps, pi_samps = [], [], []
    log2pi = np.log(2.0 * np.pi)

    for t in range(1, int(iters) + 1):
        # ---- sample z | pi, mu, sigma ----
        log_pi = np.log(np.maximum(pi, 1e-300))
        logp = np.zeros((N, K), dtype=float)
        for k in range(K):
            var = np.maximum(sigma[k] ** 2, 1e-12)
            diff = X - mu[k]
            loglik = -0.5 * (np.sum(log2pi + np.log(var)) + np.sum((diff ** 2) / var, axis=1))
            logp[:, k] = log_pi[k] + loglik

        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
        u = rng.random(N)
        cdf = np.cumsum(p, axis=1)
        z = (cdf < u[:, None]).sum(axis=1)
        z = np.clip(z, 0, K - 1)

        # ---- sample pi | z ----
        Nk = np.bincount(z, minlength=K).astype(float)
        pi = rng.dirichlet(alpha0 * np.ones(K, dtype=float) + Nk)

        # ---- sample (mu, sigma^2) | z ----
        for k in range(K):
            idxk = np.where(z == k)[0]
            nk = int(idxk.size)
            if nk == 0:
                for j in range(d):
                    s2 = _sample_inv_gamma(rng, a0, b0)
                    sigma[k, j] = np.sqrt(max(s2, 1e-12))
                    mu[k, j] = rng.normal(loc=m0, scale=np.sqrt(s2 / max(kappa0, 1e-12)))
                continue

            Xk = X[idxk]
            xbar = Xk.mean(axis=0)
            ss = ((Xk - xbar[None, :]) ** 2).sum(axis=0)

            kappa_n = kappa0 + nk
            m_n = (kappa0 * m0 + nk * xbar) / max(kappa_n, 1e-12)
            a_n = a0 + 0.5 * nk
            b_n = b0 + 0.5 * ss + 0.5 * (kappa0 * nk / max(kappa_n, 1e-12)) * ((xbar - m0) ** 2)

            for j in range(d):
                s2 = _sample_inv_gamma(rng, a_n, float(b_n[j]))
                sigma[k, j] = np.sqrt(max(s2, 1e-12))
                mu[k, j] = rng.normal(loc=float(m_n[j]), scale=np.sqrt(s2 / max(kappa_n, 1e-12)))

        # ---- store ----
        if t > int(burnin) and ((t - int(burnin)) % int(thin) == 0):
            mu_store = mu.copy()
            sigma_store = sigma.copy()
            pi_store = pi.copy()

            if ref_means is not None:
                perm = match_clusters(np.asarray(ref_means, dtype=float), mu_store)
                mu_store = mu_store[perm]
                sigma_store = sigma_store[perm]
                pi_store = pi_store[perm]

            mu_samps.append(mu_store)
            sigma_samps.append(sigma_store)
            pi_samps.append(pi_store)

    return {
        "mu_samps": np.asarray(mu_samps, dtype=float),
        "sigma_samps": np.asarray(sigma_samps, dtype=float),
        "pi_samps": np.asarray(pi_samps, dtype=float),
        "_kind": "gibbs_fullposterior_diag",
    }

def plot_posterior_mu_threeway_overlay(
    *, title_prefix, true_means,
    samples_means_sgld, sgld_tail_T,
    svi_params, svi_n_samples=3000, svi_seed=0,
    gibbs_mu_samps,
    max_params=10, n_cols=5, grid_std=4.0,
):
    """
    Three-way posterior overlays for mu: Gibbs baseline vs SGLD vs SVI.
    Curves only (Normal fits; no histogram).
    """
    true_means = np.asarray(true_means, dtype=float)
    K, d = true_means.shape

    samples_means_sgld = np.asarray(samples_means_sgld, dtype=float)
    gibbs_mu_samps = np.asarray(gibbs_mu_samps, dtype=float)

    assert samples_means_sgld.ndim == 3 and samples_means_sgld.shape[1:] == (K, d), \
        "[assert] SGLD samples not matched/reshaped to (T,K,d)."
    assert gibbs_mu_samps.ndim == 3 and gibbs_mu_samps.shape[1:] == (K, d), \
        "[assert] Gibbs samples must be (S,K,d) and already matched via ref_means."
    assert "m" in svi_params and np.asarray(svi_params["m"]).shape == (K, d), \
        "[assert] SVI params must be permuted/matched to (K,d) before overlay."

    # choose params: keep your original manual favorites then fill row-major
    requested = [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1)]
    exclude = {(2, 0), (0, 0), (0, 2), (1, 2)}
    idxs, used = [], set()
    for (k, j) in requested:
        if 0 <= k < K and 0 <= j < d and (k, j) not in used and (k, j) not in exclude:
            idxs.append((k, j)); used.add((k, j))

    max_params = min(int(max_params), K * d)
    for (k, j) in [(kk, jj) for kk in range(K) for jj in range(d)]:
        if len(idxs) >= max_params:
            break
        if (k, j) not in used and (k, j) not in exclude:
            idxs.append((k, j)); used.add((k, j))

    # SVI draws (factorized q(mu))
    mu_svi_samps = sample_posterior_means(svi_params, n_samples=int(svi_n_samples), seed=int(svi_seed))
    mu_svi_samps = np.asarray(mu_svi_samps, dtype=float)
    assert mu_svi_samps.shape[1:] == (K, d)

    # SGLD tail
    T = samples_means_sgld.shape[0]
    tail_len = min(int(sgld_tail_T), int(T))
    sgld_tail = samples_means_sgld[-tail_len:]

    # (removed per-parameter color map; use fixed method colors below)

    n = len(idxs)
    n_cols = int(max(1, n_cols))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.6 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax_idx, (k, j) in enumerate(idxs):
        ax = axes_flat[ax_idx]

        # Canonical method colors:
        #   Gibbs   -> dark gray / black
        #   SGLD    -> keep existing (orange-like, matches calibration)
        #   SVI     -> reuse previous Gibbs green
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

        c_gibbs = "0.2"        # dark gray / near-black
        c_sgld  = cycle[1] if len(cycle) > 1 else "C1"
        c_svi   = cycle[2] if len(cycle) > 2 else "C2"

        s_gibbs = np.asarray(gibbs_mu_samps[:, k, j], dtype=float)
        s_sgld = np.asarray(sgld_tail[:, k, j], dtype=float)
        s_svi = np.asarray(mu_svi_samps[:, k, j], dtype=float)

        # Fit Normal(mean, std) to each sample set
        def _fit_normal(series):
            series = series[np.isfinite(series)]
            m = float(np.mean(series)) if series.size > 0 else 0.0
            sd = float(np.std(series, ddof=1)) if series.size > 1 else 0.0
            if (not np.isfinite(sd)) or sd <= 0:
                sd = 1e-6
            return m, sd

        m_g, sd_g = _fit_normal(s_gibbs)
        m_sg, sd_sg = _fit_normal(s_sgld)
        m_v, sd_v = _fit_normal(s_svi)

        # Quantile-based shared x-range (robust when one method has much larger variance)
        all_samples = np.concatenate([s_gibbs, s_sgld, s_svi])
        all_samples = all_samples[np.isfinite(all_samples)]
        if all_samples.size >= 10:
            lo, hi = np.quantile(all_samples, [0.005, 0.995])
        else:
            # Fallback: use the three Normal fits
            m_all = np.array([m_g, m_sg, m_v], dtype=float)
            sd_all = np.array([sd_g, sd_sg, sd_v], dtype=float)
            m0 = float(np.mean(m_all))
            sd0 = float(np.max(sd_all))
            sd0 = max(sd0, 1e-6)
            lo, hi = (m0 - grid_std * sd0), (m0 + grid_std * sd0)
        # Guard against degenerate ranges
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-12:
            lo, hi = -1.0, 1.0
        xs = np.linspace(lo, hi, 500)

        y_gibbs = norm.pdf(xs, loc=m_g, scale=sd_g)
        y_sgld = norm.pdf(xs, loc=m_sg, scale=sd_sg)
        y_svi = norm.pdf(xs, loc=m_v, scale=sd_v)

        ax.plot(xs, y_gibbs, color=c_gibbs, linewidth=2.0, label="Gibbs baseline" if ax_idx == 0 else None)
        ax.plot(xs, y_sgld, color=c_sgld, linewidth=2.0, linestyle="-.", label="SGLD (tail)" if ax_idx == 0 else None)
        ax.plot(xs, y_svi, color=c_svi, linewidth=2.0, linestyle="--", label="SVI" if ax_idx == 0 else None)

        ax.set_title(f"{title_prefix}mu[{k},{j}]")
        ax.set_xlabel("value"); ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[n:]:
        ax.axis("off")

    if n > 0:
        axes_flat[0].legend(frameon=True, fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig
def plot_mu_trajectories(samples_means, samples_sigma, true_means, true_covs):
    """
    samples_means: (iters, K, d)
    samples_sigma: (iters, K, d)
    true_means: (K, d)
    true_covs: (K, d, d)
    """
    iters = samples_means.shape[0]
    K, d = samples_means.shape[1], samples_means.shape[2]
    t = np.arange(iters)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Estimate posterior mean params
    est_means = samples_means.mean(axis=0)         # (K, d)
    est_sigma = samples_sigma.mean(axis=0)         # (K, d)
    true_sigma = np.sqrt(np.diagonal(true_covs, axis1=1, axis2=2))  # (K, d)

    # Match clusters: perm[k_true] = k_est
    perm = match_clusters(true_means, est_means, true_sigma, est_sigma)
    samples_means = samples_means[:, perm, :]
    true_means_perm = true_means

    for k in range(K):
        plt.figure(figsize=(8, 3))
        for j in range(d):
            col = colors[j % len(colors)]
            plt.plot(t, samples_means[:, k, j], color=col, label=f"mu[{k},{j}]")
            plt.hlines(true_means_perm[k, j], 0, iters - 1,
                       linestyles="dashed", color=col, label=f"true mu[{k},{j}]")
        plt.title(f"Trajectory of mu for cluster {k}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_sigma_trajectories(samples_sigma, true_covs, samples_means, true_means):
    """
    samples_sigma: (iters, K, d)
    true_covs: (K, d, d)
    samples_means: (iters, K, d)
    true_means: (K, d)
    """
    iters = samples_sigma.shape[0]
    K, d = samples_sigma.shape[1], samples_sigma.shape[2]
    t = np.arange(iters)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    est_means = samples_means.mean(axis=0)
    est_sigma = samples_sigma.mean(axis=0)  # (K, d)
    true_sigma = np.sqrt(np.diagonal(true_covs, axis1=1, axis2=2))  # (K, d)

    # Match clusters: perm[k_true] = k_est
    perm = match_clusters(true_means, est_means, true_sigma, est_sigma)
    samples_sigma = samples_sigma[:, perm, :]
    # true_sigma is already indexed by true cluster, no permutation needed

    for k in range(K):
        plt.figure(figsize=(8, 3))
        for j in range(d):
            col = colors[j % len(colors)]
            plt.plot(t, samples_sigma[:, k, j], color=col, label=f"sigma[{k},{j}]")
            plt.hlines(true_sigma[k, j], 0, iters - 1,
                       linestyles="dashed", color=col, label=f"true sigma[{k},{j}]")
        plt.title(f"Trajectory of sigma for cluster {k}")
        plt.xlabel("Iteration")
        plt.ylabel("Std dev")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_density_contours_real(X, true_means, true_covs, true_weights,
                               samples_means, samples_sigma, z_last, tail=1000):
    """
    For real datasets, plot density contours of:
      - the pseudo-true GMM (from k-means)
      - the predicted GMM (from posterior mean params + empirical weights)
    overlaid on the true data.
    Uses the first two coordinates of the current feature space for visualization.
    """
    K, d = true_means.shape
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # True variances and sigmas in original d-dim space
    true_var_full = np.diagonal(true_covs, axis1=1, axis2=2)   # (K, d)
    true_sigma_full = np.sqrt(true_var_full)

    # Use tail of the chain to estimate parameters
    if samples_means.shape[0] >= tail:
        means_tail = samples_means[-tail:, :, :]
        sigma_tail = samples_sigma[-tail:, :, :]
    else:
        means_tail = samples_means
        sigma_tail = samples_sigma

    est_means_full = means_tail.mean(axis=0)       # (K, d)
    est_sigma_full = sigma_tail.mean(axis=0)       # (K, d)

    # Cluster matching (means + sigma) in original space
    # perm[k_true] = k_est
    perm = match_clusters(true_means, est_means_full, true_sigma_full, est_sigma_full)
    true_means_perm_full = true_means
    true_var_full_perm = true_var_full

    est_means_full = est_means_full[perm]
    est_sigma_full = est_sigma_full[perm]

    # Empirical weights from final hard assignments, permuted to align with true cluster order
    w_hat = np.array([np.mean(z_last == k) for k in range(K)])
    w_hat = w_hat[perm]

    # Use first two coordinates of the current feature space (already PCA-reduced if d > 8)
    if d < 2:
        print("Skipping density contour plot (need at least 2 dimensions).")
        return

    Z = X[:, :2]  # (N, 2)

    # Project means by taking their first two coordinates
    true_means_2 = true_means_perm_full[:, :2]  # (K, 2)
    est_means_2 = est_means_full[:, :2]         # (K, 2)

    # Variances along the first two coordinates
    true_var_2 = true_var_full_perm[:, :2]      # (K, 2)
    est_var_full = est_sigma_full ** 2
    est_var_2 = est_var_full[:, :2]             # (K, 2)

    # Build grid over data range in these two coordinates
    x_min, x_max = Z[:, 0].min() - 3.0, Z[:, 0].max() + 3.0
    y_min, y_max = Z[:, 1].min() - 3.0, Z[:, 1].max() + 3.0

    nx, ny = 150, 150
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (nx*ny, 2)

    def gmm_density(grid_points, means, variances, weights):
        densities = np.zeros(grid_points.shape[0])
        for k in range(K):
            diff = grid_points - means[k]
            v = variances[k]
            inv_v = 1.0 / v
            norm_const = 1.0 / (2.0 * np.pi * np.sqrt(v[0] * v[1]))
            expo = -0.5 * np.sum(diff * diff * inv_v, axis=1)
            densities += weights[k] * norm_const * np.exp(expo)
        return densities

    # True and predicted densities in PC space
    true_dens = gmm_density(grid_points, true_means_2, true_var_2, true_weights).reshape(nx, ny)
    est_dens = gmm_density(grid_points, est_means_2, est_var_2, w_hat).reshape(nx, ny)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True density
    ax = axes[0]
    cf = ax.contourf(xx, yy, true_dens, levels=20, alpha=0.8)
    ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.1)
    for k in range(K):
        ax.scatter(true_means_2[k, 0], true_means_2[k, 1],
                   marker="x", s=80, linewidths=2,
                   color=colors[k % len(colors)],
                   label="Pseudo-true mean" if k == 0 else None)
    ax.set_title("Pseudo-true GMM density")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Predicted density
    ax = axes[1]
    cf2 = ax.contourf(xx, yy, est_dens, levels=20, alpha=0.8)
    ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.1)
    for k in range(K):
        ax.scatter(est_means_2[k, 0], est_means_2[k, 1],
                   marker="o", s=80,
                   color=colors[k % len(colors)],
                   label="Estimated mean" if k == 0 else None)
    ax.set_title("Predicted GMM density")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.colorbar(cf2, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Label matching via confusion matrix and Hungarian algorithm
# ---------------------------------------------------------------------
def match_labels_confusion(z_true, z_pred, K):
    """
    Match predicted cluster labels to true labels via Hungarian algorithm
    on the confusion matrix.
    Returns:
      mapping_pred_to_true: array of length K such that new_pred = mapping[z_pred]
    """
    # Confusion matrix C[true, pred]
    C = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            C[i, j] = np.sum((z_true == i) & (z_pred == j))
    # We want to maximize trace(C), so minimize -C
    cost = -C
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping_pred_to_true = np.zeros(K, dtype=int)
    for i, j in zip(row_ind, col_ind):
        mapping_pred_to_true[j] = i
    return mapping_pred_to_true


# ---------------------------------------------------------------------
# Hard assignments from GMM parameters
# ---------------------------------------------------------------------
def hard_assign_gmm(X, means, sigma, weights=None):
    """Hard-assign each point in X to the most likely Gaussian component.

    X: (N, d)
    means: (K, d)
    sigma: (K, d)  # standard deviations for diagonal covariances
    weights: (K,) or None. If None, use uniform weights.
    Returns: z_pred of shape (N,) with cluster indices in {0,...,K-1}.
    """
    N, d = X.shape
    K = means.shape[0]
    if weights is None:
        weights = np.ones(K) / K
    log_w = np.log(weights + 1e-16)

    # Precompute per-component constants
    var = sigma ** 2  # (K, d)
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * var), axis=1)  # (K,)

    # Compute log p(x_n | k) + log w_k for all n,k
    log_probs = np.zeros((N, K))
    for k in range(K):
        diff = X - means[k]
        inv_var = 1.0 / var[k]
        quad = -0.5 * np.sum(diff * diff * inv_var, axis=1)  # (N,)
        log_probs[:, k] = log_w[k] + log_norm[k] + quad

    # Hard assignments
    z_pred = np.argmax(log_probs, axis=1)
    return z_pred


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------
def run_sgld_on_real(which=DATASET, *, init_mu0=None, init_sigma0=None, init_pi0=None):
    data = load_real_dataset(which)
    X = data["X"]
    z_true = data["z_true"]
    K = data["K"]
    d = data["d"]
    N = data["N"]
    true_means = data["true_means"]
    true_covs = data["true_covs"]
    true_weights = data["true_weights"]

    print(f"Dataset: {data['name']}")
    print(f"N = {N}, d = {d}, K = {K}")

    # Hyperparams – you can tune per dataset
    batch_size =25 if N > 500 else 1
    stepsize =  batch_size / N
    iters = 20000
    inverse_temperature = N  # same scaling you used

    print("Running SGLD+Gibbs with:")
    print(f"  batch_size = {batch_size}, stepsize = {stepsize}, iters = {iters}")

    try:
        mu_final, samples_means, sigma_final, samples_sigma, z0_history = run(
            K=K,
            d=d,
            X=X,
            stepsize=stepsize,
            batch_size=batch_size,
            iters=iters,
            true_means=true_means,
            true_covs=true_covs,
            true_weights=true_weights,
            inverse_temperature=inverse_temperature,
            use_precond=True,
            init_mu0=init_mu0,
            init_sigma0=init_sigma0,
            init_pi0=init_pi0,
        )
    except TypeError:
        print("[warn] SGLD run() does not accept init_* yet; using internal init.")
        mu_final, samples_means, sigma_final, samples_sigma, z0_history = run(
            K=K,
            d=d,
            X=X,
            stepsize=stepsize,
            batch_size=batch_size,
            iters=iters,
            true_means=true_means,
            true_covs=true_covs,
            true_weights=true_weights,
            inverse_temperature=inverse_temperature,
            use_precond=True,
        )

    # -----------------------------------------------------------------
    # Canonical matching: permute SGLD outputs to pseudo-true component order
    # (needed for any 3-way overlays: Gibbs baseline vs SGLD vs SVI)
    # -----------------------------------------------------------------
    samples_means = np.asarray(samples_means, dtype=float)
    samples_sigma = np.asarray(samples_sigma, dtype=float)
    mu_final = np.asarray(mu_final, dtype=float)
    sigma_final = np.asarray(sigma_final, dtype=float)

    # Use a stable estimate of means for matching (tail-mean by default)
    T_match = samples_means.shape[0]
    tail_match = min(5000, int(T_match))
    est_means_match = samples_means[-tail_match:].mean(axis=0)  # (K,d)

    perm_sgld = match_clusters(true_means, est_means_match)

    # Permute ALL per-component outputs so that index k matches pseudo-true component k
    samples_means = samples_means[:, perm_sgld, :]
    samples_sigma = samples_sigma[:, perm_sgld, :]
    mu_final = mu_final[perm_sgld, :]
    sigma_final = sigma_final[perm_sgld, :]

    sgld_matched_to_pseudotrue = True

    # Basic clustering accuracy using final assignments vs true labels
    # Use hard assignments from the final GMM parameters instead of relying on z0_history shape
    z_last = hard_assign_gmm(X, mu_final, sigma_final, true_weights)
    labels_for_plot = z_last
    if z_true is not None:
        # Report permutation-invariant clustering metrics (preferred)
        ari = adjusted_rand_score(z_true, z_last)
        ami = adjusted_mutual_info_score(z_true, z_last)
        print(f"ARI = {ari:.4f}")
        print(f"AMI = {ami:.4f}")

        # Optional: label matching (for downstream plots), but do not report accuracy
        K_true = len(np.unique(z_true))
        K_pred = len(np.unique(z_last))
        if K_true == K_pred == K:
            mapping = match_labels_confusion(z_true, z_last, K)
            z_last_matched = mapping[z_last]
            labels_for_plot = z_last_matched
        else:
            labels_for_plot = z_last

    # Plot trajectories of mu and sigma using pseudo-true parameters from k-means
    print("Plotting trajectories of mu and sigma...")
    plot_mu_trajectories(samples_means, samples_sigma, true_means, true_covs)
    plot_sigma_trajectories(samples_sigma, true_covs, samples_means, true_means)

    # Density contour plots on the true data (for 2D datasets like Old Faithful)
    print("Plotting density contours on the true data (pseudo-true vs predicted)...")
    plot_density_contours_real(X, true_means, true_covs, true_weights,
                               samples_means, samples_sigma, z_last)

    return {
        **data,
        "mu_final": mu_final,
        "sigma_final": sigma_final,
        "samples_means": samples_means,
        "samples_sigma": samples_sigma,
        "z0_history": z0_history,
        "perm_sgld": perm_sgld,
        "sgld_matched_to_pseudotrue": sgld_matched_to_pseudotrue,
    }


def main():
    which = DATASET
    print("\n" + "=" * 80)
    print(f"Selected dataset: {which}")

    # Load selected dataset
    data = load_real_dataset(which)
    X = data["X"]
    z_true = data["z_true"]
    K = data["K"]
    N = data["N"]

    # --- Shared init for BOTH SVI and SGLD ---
    init_mu0 = None
    init_sigma0 = None
    init_pi0 = None

    if USE_TRUE_LABEL_INIT and ("init_mu0" in data and "init_sigma0" in data and "init_pi0" in data):
        init_mu0 = np.asarray(data["init_mu0"], dtype=float)
        init_sigma0 = np.asarray(data["init_sigma0"], dtype=float)
        init_pi0 = np.asarray(data["init_pi0"], dtype=float)
        print("[init] Using TRUE-LABEL init (mu/sigma/pi) from dataset loader.")
    else:
        if INIT_METHOD == "gmm_em":
            init_mu0, init_sigma0, init_pi0, _ = fit_em_gmm_init(X, K, seed=EM_SEED)
            print("[init] Using EM-GMM initialization.")
        elif INIT_METHOD == "kmeans":
            mu0, covs0, w0, _ = build_pseudo_true_gmm(X, K)
            init_mu0 = mu0
            init_sigma0 = np.sqrt(np.diagonal(covs0, axis1=1, axis2=2))
            init_pi0 = w0
            print("[init] Using KMeans initialization.")
        elif INIT_METHOD == "random":
            rng = np.random.default_rng(0)
            init_mu0 = rng.standard_normal((K, X.shape[1]))
            init_sigma0 = rng.uniform(0.5, 2.0, size=(K, X.shape[1]))
            init_pi0 = np.ones(K) / K
            print("[init] Using random initialization.")
        else:
            raise ValueError(INIT_METHOD)

    # --- Run SVI on the selected dataset ---
    print("Running SVI..." + (" (flat-prior, fixed pi=1/K)" if SVI_FLAT_PRIOR else " (Dirichlet/Normal-Gamma prior)"))

    if SVI_FLAT_PRIOR:
        svi_params = svi_gmm_diag_flat(
            X,
            K,
            iters=3000,
            batch_size=min(512, N),
            seed=0,
            # keep eps values modest for stability; you can tune smaller if desired
            eps_beta0=1e-8,
            eps_a0=1e-8,
            eps_b0=1e-8,
            tau0=10.0,
            kappa=0.7,
            # IMPORTANT: share init with SGLD
            init_method=INIT_METHOD,
            init_mu0=init_mu0,
            init_sigma0=init_sigma0,
            init_pi0=init_pi0,
        )
    else:
        svi_params = svi_gmm_diag(
            X,
            K,
            iters=5000,
            batch_size=min(128, N),
            seed=0,
            alpha0=1.0,
            beta0=1.0,
            a0=2.0,
            b0=2.0,
            tau0=10.0,
            kappa=0.7,
            # IMPORTANT: share init with SGLD
            init_method=INIT_METHOD,
            init_mu0=init_mu0,
            init_sigma0=init_sigma0,
            init_pi0=init_pi0,
        )

    z_pred = predict_labels_from_svi(X, svi_params)
    # --- Match SVI components to pseudo-true order ---
    perm_svi = match_clusters(data["true_means"], np.asarray(svi_params["m"], dtype=float))
    svi_params = _permute_svi_params(svi_params, perm_svi)
    # If ground truth labels exist, report permutation-invariant metrics
    if z_true is not None:
        ari = adjusted_rand_score(z_true, z_pred)
        ami = adjusted_mutual_info_score(z_true, z_pred)
        print(f"ARI = {ari:.4f}")
        print(f"AMI = {ami:.4f}")

    # Also print mixture weights
    if "alpha" in svi_params:
        alpha = svi_params["alpha"]
        pi_mean = alpha / alpha.sum()
        print("Variational mean weights pi (Dirichlet mean):", pi_mean)
    elif "pi_fixed" in svi_params:
        print("Mixture weights pi (fixed):", svi_params["pi_fixed"])

    # Also print posterior mean of component means (m)
    print("Variational mean component means m:\n", svi_params["m"])

    # --- Run SGLD+Gibbs on the selected dataset ---
    print("\nRunning SGLD+Gibbs...")
    sgld_result = run_sgld_on_real(which, init_mu0=init_mu0, init_sigma0=init_sigma0, init_pi0=init_pi0)
    gibbs_out = None
    if RUN_BASELINE_GIBBS:
        print("\nRunning full-data Gibbs baseline (full posterior over (pi,mu,sigma,z))...")
        gibbs_out = gibbs_gmm_diag_fullposterior(
            X, K,
            iters=3000, burnin=1000, thin=5,
            seed=123,
            ref_means=data["true_means"],        # 关键：存样本时就 matching
            init_mu0=init_mu0,
            init_sigma0=init_sigma0,
            init_pi0=init_pi0,
            # weak/flat-ish priors
            alpha0=1.0, m0=0.0, kappa0=1e-3, a0=1e-3, b0=1e-3,
        )
        print(f"[Gibbs baseline] stored samples: {gibbs_out['mu_samps'].shape[0]}")
    # Optional: compare SVI vs SGLD final hard assignments with permutation-invariant metrics
    if z_true is not None:
        z_sgld = hard_assign_gmm(sgld_result["X"], sgld_result["mu_final"], sgld_result["sigma_final"], sgld_result["true_weights"])
        ari_vs_sgld = adjusted_rand_score(z_true, z_sgld)
        ami_vs_sgld = adjusted_mutual_info_score(z_true, z_sgld)
        print(f"SGLD ARI = {ari_vs_sgld:.4f}")
        print(f"SGLD AMI = {ami_vs_sgld:.4f}")

    # --- Compare posterior over mu: SVI vs SGLD tail (last 3000 iters) ---
    print("\nPlotting posterior comparison for mu: SVI vs SGLD tail...")
    print("\nPosterior overlays for mu (curves only): Gibbs baseline vs SGLD vs SVI")
    if RUN_BASELINE_GIBBS and gibbs_out is not None:
        assert sgld_result.get("sgld_matched_to_pseudotrue", False), \
            "[assert] SGLD outputs must be matched to pseudo-true order before 3-way overlays."
        plot_posterior_mu_threeway_overlay(
            title_prefix=f"{which} ",
            true_means=data["true_means"],
            samples_means_sgld=sgld_result["samples_means"],
            sgld_tail_T=5000,
            svi_params=svi_params,
            svi_n_samples=3000,
            svi_seed=0,
            gibbs_mu_samps=gibbs_out["mu_samps"],
            max_params=5,
            n_cols=1,
            grid_std=4.0,
        )
    else:
        print("[warn] Gibbs baseline disabled; falling back to SVI vs SGLD plot.")
        plot_compare_mu_posteriors_svi_vs_sgld(
            svi_params,
            sgld_result["samples_means"],
            data["true_means"],
            tail=5000,
            n_cols=5,
            max_params=10,
            title_prefix=f"{which} ",
            show_hist=False,
        )   

    # --- Rank-uniformity (PIT) calibration for mu: SVI vs SGLD in one plot ---
    print("\nRank-uniformity calibration (mu): SVI vs SGLD")

    p_svi = rank_uniformity_mu_svi(
        svi_params,
        data["true_means"],
        n_samples=3000,
        seed=0,
    )

    p_sgld = rank_uniformity_mu_sgld(
        sgld_result["samples_means"],
        data["true_means"],
        tail=3000,
    )

    plot_rank_uniformity_compare(
        p_svi,
        p_sgld,
        label_a="SVI",
        label_b="SGLD",
        title=f"Rank-uniformity (mu): SVI vs SGLD — {which}",
    )


if __name__ == "__main__":
    main()