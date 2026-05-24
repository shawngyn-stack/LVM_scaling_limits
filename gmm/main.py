# main.py
# %%
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
# ===================== Matplotlib LaTeX + font config =====================
# Use a single backslash in the LaTeX preamble (r"\usepackage{...}"), not double.
# Using r"\\usepackage{...}" (two backslashes) makes LaTeX see "\\u" and fail.
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    # --- Emphasize titles/legends, de-emphasize axis tick numbers (2x font sizes) ---
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "axes.labelsize": 28,
    "axes.labelweight": "bold",
    "figure.labelsize": 28,
    "figure.labelweight": "bold",
    "legend.fontsize": 28,
    "legend.title_fontsize": 28,
    "figure.titlesize": 28,
    "font.weight": "bold",
    # Tick label sizes (smaller)
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,

    # --- Legend placement: keep upper-left, avoid covering data ---
    "legend.loc": "upper left",
    "legend.framealpha": 0.0,
    "legend.borderaxespad": 0.3,

    # --- Save figures as PDF by default ---
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "axes.unicode_minus": False,
})
# NOTE: For `text.latex.preamble`, use a single backslash (e.g., r"\usepackage{...}").
# Using r"\\usepackage{...}" (two backslashes) makes LaTeX see "\\u" and fail.

# Figure-level suptitle font size constant
SUPTITLE_SIZE = 48

# Global square figure size (paper-ready)
FIGSIZE_SQUARE = (9, 6)

# Color constants for plots
COLOR_REF = "#888888"          # gray for reference (true / uniform)
COLOR_SVI = "#377eb8"          # blue for SVI
COLOR_SGLD = "#e41a1c"         # red for SGLD S=1
COLOR_SGLD_S10 = "#a50f15"     # dark red for SGLD S=10

# ===================== Auto-save figures =====================
_FIG_COUNTER = 0

def save_and_show(fig=None, prefix="fig"):
    """Save current Matplotlib figure as PDF with an auto-incremented name, then show."""
    global _FIG_COUNTER
    if fig is None:
        fig = plt.gcf()
    fname = f"{prefix}_{_FIG_COUNTER:02d}.pdf"
    fig.savefig(fname)
    print(f"[figure saved] {fname}")
    _FIG_COUNTER += 1
    plt.show()

from sgld_gibbs_fixed_cov2 import generate_data, run

# Optional: averaged-gradient SGLD variant (S>1 Gibbs draws per minibatch)
try:
    from sgld_gibbs_fixed_cov2 import run_avg_gibbs
except Exception:
    run_avg_gibbs = None

# External SVI baseline (VB / Normal-Gamma). This lives in svi.py next to main.py.
# We use the "flat + fixed pi" variant to better match the SGLD setup (no explicit prior, pi=1/K).
from svi import svi_gmm_diag_flat


# %%
def match_clusters(true_means, est_means):
    from scipy.optimize import linear_sum_assignment
    K = true_means.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.linalg.norm(true_means[i] - est_means[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind


# %%
def build_gmm_dataset():
    """
    2D, 3-cluster GMM with N=10000 points.
    Covariances are diagonal but passed as full matrices to generate_data().
    """

    K = 6
    d = 8
    N = 30000

    true_means = np.array([
        [-6.0, -2.0,  0.5,  1.5, -3.0,  2.0,  0.0,  4.0],   # cluster 0
        [-1.0,  6.0, -2.0,  0.0,  3.5, -1.5,  2.5, -3.0],   # cluster 1
        [ 6.0, -3.0,  2.5, -1.0,  1.0,  4.5, -2.5,  0.5],   # cluster 2
        [ 2.0,  2.5, -5.5,  3.0, -4.0,  0.5,  3.0, -1.5],   # cluster 3
        [ 8.0,  5.0,  4.0, -4.5,  2.0, -3.5,  1.0,  2.5],   # cluster 4
        [-4.5,  1.0,  6.5,  2.0, -1.0,  3.0, -4.0, -2.0],   # cluster 5
    ])
    # Increase overlap between clusters:
    #   - shrink mean separation
    #   - inflate covariance (diagonal) slightly
    overlap_mean_scale = 0.8   # smaller => more overlap
    overlap_cov_scale = 1.60    # larger => more overlap
    true_means = true_means * overlap_mean_scale

    true_covs = np.array([
        np.diag([0.8, 0.5, 1.2, 0.6, 0.9, 1.1, 0.7, 1.4]),   # cluster 0
        np.diag([0.4, 1.3, 0.7, 1.0, 1.5, 0.6, 1.2, 0.8]),   # cluster 1
        np.diag([1.6, 0.8, 0.5, 1.4, 0.6, 1.0, 1.3, 0.7]),   # cluster 2
        np.diag([0.9, 1.1, 1.8, 0.6, 1.0, 0.9, 0.8, 1.2]),   # cluster 3
        np.diag([0.6, 0.7, 1.0, 1.5, 1.1, 0.5, 0.9, 1.3]),   # cluster 4
        np.diag([1.2, 0.6, 0.9, 0.8, 1.4, 1.0, 0.5, 1.1]),   # cluster 5
    ])
    true_covs = true_covs * overlap_cov_scale

    true_weights = np.ones(K) / K

    X, z_true = generate_data(
        K=K, d=d, N=N,
        means=true_means,
        covs=true_covs,
        weights=true_weights
    )

    return K, d, N, X, z_true, true_means, true_covs, true_weights


# %%
def run_sgld_gibbs_example():
    K, d, N, X, z_true, true_means, true_covs, true_weights = build_gmm_dataset()

    # Track a few *ambiguous / overlapping* datapoints to compare predicted vs sampled z.
    # We pick points whose true-parameter posterior p(z|x,theta*) is most uncertain,
    # i.e., smallest max_k p(z=k|x).
    m_track = min(5, N)

    K_local = K
    logw = np.log(np.maximum(true_weights, 1e-300))
    logp = np.zeros((N, K_local), dtype=float)

    # compute log p(z=k|x) up to an additive constant
    for k in range(K_local):
        s2 = np.maximum(np.diag(true_covs[k]).astype(float), 1e-12)
        diff = X - np.asarray(true_means[k], dtype=float)[None, :]
        loglik = -0.5 * (
            np.sum(np.log(2.0 * np.pi * s2))
            + np.sum((diff ** 2) / s2[None, :], axis=1)
        )
        logp[:, k] = logw[k] + loglik

    # normalize for posterior probs
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)

    maxp = p.max(axis=1)

    # Track only four ambiguous points (top four ranks)
    # Ranks are 1-indexed: rank=1 is the most ambiguous (smallest maxp).
    desired_ranks = [2, 6, 11, 100]
    order = np.argsort(maxp)  # ascending by ambiguity score

    track_indices = []
    used = set()
    for r in desired_ranks:
        j = int(r) - 1  # convert to 0-index
        if j < 0:
            continue
        if j >= order.size:
            j = order.size - 1
        g = int(order[j])
        if g not in used:
            track_indices.append(g)
            used.add(g)

    print("[z-track] tracking overlapped points by rank:", desired_ranks)
    print("[z-track] tracking global indices:", track_indices)
    print("[z-track] their max posterior probs (smaller => more overlap):", maxp[track_indices])

    # Hyperparameters (your requested setup)
    w1 = 1
    w2 = 1
    batch_size = 50
    stepsize =  batch_size * w1 / N    # = 0.01 for N=10000, batch_size=25
    iters = 40000
    inverse_temperature = N / w2

    print("Running SGLD+Gibbs with:")
    print(f"  K = {K}, d = {d}, N = {N}")
    print(f"  batch_size = {batch_size}, stepsize = {stepsize}, iters = {iters}")

    mu_final, samples_means, sigma_final, samples_sigma, z0_history, z_track_history = run(
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
        track_indices=track_indices,
    )

    return {
        "K": K,
        "d": d,
        "N": N,
        "X": X,
        "w1": w1,
        "w2": w2,
        "batch_size": batch_size,
        "stepsize": stepsize,
        "iters": iters,
        "z_true": z_true,
        "true_means": true_means,
        "true_covs": true_covs,
        "true_weights": true_weights,
        "mu_final": mu_final,
        "sigma_final": sigma_final,
        "samples_means": samples_means,
        "samples_sigma": samples_sigma,
        "z0_history": z0_history,
        "track_indices": track_indices,
        "z_track_history": z_track_history,
    }


# %%
def plot_data_and_true_means(X, z_true, true_means):
    plt.figure(figsize=FIGSIZE_SQUARE)
    K = true_means.shape[0]

    # If d>2, visualize only the first two coordinates.
    if X.shape[1] != 2:
        print(f"[plot] X has d={X.shape[1]} dims; plotting only dims (0,1).")

    # scatter data colored by true cluster assignments
    for k in range(K):
        mask = (z_true == k)
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            s=4,
            alpha=0.35,
            label=f"Cluster {k}"
        )

    # plot true means
    plt.scatter(
        true_means[:, 0],
        true_means[:, 1],
        marker="x",
        s=100,
        linewidths=3,
        label="True means"
    )

    plt.title("2D GMM data (true clusters)", fontweight="bold")
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.legend()
    plt.tight_layout()
    save_and_show()


# %%
# Helper: plot datapoints colored by estimated clusters (MAP under estimated parameters),
# overlaying both TRUE means and estimated means.
def plot_data_with_estimated_clusters(X, true_means, est_means, est_sigma, *, title="Data colored by estimated clusters", max_points=8000, seed=0):
    """Scatter plot of data colored by MAP cluster under estimated params.

    Notes:
      - If d>2, we plot only dims (0,1).
      - Uses diagonal covariance with std `est_sigma` (K,d).
      - Assumes uniform mixture weights.
    """
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=float)
    true_means = np.asarray(true_means, dtype=float)
    est_means = np.asarray(est_means, dtype=float)
    est_sigma = np.asarray(est_sigma, dtype=float)

    N, d = X.shape
    K = est_means.shape[0]

    # Subsample for speed/visual clarity
    max_points = int(max_points)
    if max_points > 0 and N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
        Xp = X[idx]
    else:
        idx = None
        Xp = X

    if d != 2:
        print(f"[plot] X has d={d} dims; plotting only dims (0,1).")

    # Compute log p(z=k|x) up to additive constant (uniform weights)
    logp = np.zeros((Xp.shape[0], K), dtype=float)
    for k in range(K):
        s2 = np.maximum(est_sigma[k] ** 2, 1e-12)  # (d,)
        diff = Xp - est_means[k][None, :]
        loglik = -0.5 * (
            np.sum(np.log(2.0 * np.pi * s2))
            + np.sum((diff ** 2) / s2[None, :], axis=1)
        )
        logp[:, k] = loglik

    z_hat = np.argmax(logp, axis=1)

    plt.figure(figsize=FIGSIZE_SQUARE)
    cmap = plt.get_cmap("tab10")
    for k in range(K):
        mask = (z_hat == k)
        if np.any(mask):
            plt.scatter(
                Xp[mask, 0],
                Xp[mask, 1],
                s=4,
                alpha=0.35,
                color=cmap(k % 10),
                label=f"Est cluster {k}",
            )

    # TRUE means (gray x)
    plt.scatter(
        true_means[:, 0],
        true_means[:, 1],
        marker="x",
        s=120,
        linewidths=3.0,
        color=COLOR_REF,
        label="True means",
    )

    # Estimated means (black circle)
    plt.scatter(
        est_means[:, 0],
        est_means[:, 1],
        marker="o",
        s=70,
        edgecolors="black",
        facecolors="none",
        linewidths=2.0,
        label="Estimated means",
    )

    plt.title(title)
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.legend(loc="best", ncol=2)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    save_and_show()

    return z_hat, idx


# %%
# Helper for p(z|x,true) and plotting predicted vs sampled z for tracked points
def _predict_posterior_z_true_params(x, true_means, true_covs, true_weights):
    """Compute p(z=k | x, theta*) under the TRUE GMM parameters (diagonal covs)."""
    K = true_means.shape[0]
    logw = np.log(np.maximum(true_weights, 1e-300))
    logp = np.zeros(K, dtype=float)
    x = np.asarray(x, dtype=float)
    for k in range(K):
        s2 = np.maximum(np.diag(true_covs[k]).astype(float), 1e-12)
        diff = x - np.asarray(true_means[k], dtype=float)
        loglik = -0.5 * (np.sum(np.log(2.0 * np.pi * s2)) + np.sum((diff ** 2) / s2))
        logp[k] = logw[k] + loglik
    logp -= np.max(logp)
    p = np.exp(logp)
    return p / np.maximum(p.sum(), 1e-300)


def plot_predicted_vs_sampled_z_for_points(
    X,
    true_means,
    true_covs,
    true_weights,
    z_track_history,
    track_indices,
    inv_perm,
    *,
    tail_frac=0.5,
    tail_last=20000,
    z_track_history_s10=None,
    inv_perm_s10=None,
):
    """Plot predicted p(z|x,true) vs empirical sampled z from Gibbs for each tracked point.

    If `z_track_history_s10` is provided, overlay a second empirical bar series (e.g., SGLD avg-Gibbs S=10).
    """
    K = true_means.shape[0]
    # Arrange tracked points in a 2x2 grid (expects 4 points)
    nplots = len(track_indices)
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_SQUARE, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, g in zip(axes, track_indices):
        g = int(g)
        hist = np.asarray(z_track_history.get(g, []), dtype=int)

        # Optionally restrict to the last `tail_last` iterations of tracking.
        if tail_last is not None:
            tail_last_int = int(tail_last)
            if tail_last_int > 0 and hist.size > tail_last_int:
                hist = hist[-tail_last_int:]

        start = int(hist.size * tail_frac)
        tail = hist[start:]
        tail = tail[tail >= 0]  # ignore not-seen-yet

        # empirical from Gibbs samples (cluster ids are in 'estimated' label space)
        if tail.size > 0:
            tail_true = inv_perm[tail]
            emp = np.bincount(tail_true, minlength=K).astype(float)
            emp = emp / np.maximum(emp.sum(), 1.0)
        else:
            emp = np.zeros(K, dtype=float)

        # predicted from true parameters
        pred = _predict_posterior_z_true_params(X[g], true_means, true_covs, true_weights)

        xs = np.arange(K)
        # Set x-ticks to show cluster labels as 1,2,...,K (not 0-based)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in range(1, K + 1)])

        # Reference (predicted under TRUE params): gray
        ax.bar(
            xs - 0.25,
            pred,
            width=0.25,
            alpha=0.70,
            color=COLOR_REF,
            label=r"Pred $p(z\mid x,\theta^\star)$",
        )

        # Empirical from SGLD (S=1): red
        ax.bar(
            xs,
            emp,
            width=0.25,
            alpha=0.75,
            color=COLOR_SGLD,
            label="Empirical z (Gibbs, S=1)",
        )

        # Optional empirical overlay for S=10 run: dark red
        if (z_track_history_s10 is not None):
            if inv_perm_s10 is None:
                inv_perm_s10 = inv_perm
            hist10 = np.asarray(z_track_history_s10.get(g, []), dtype=int)
            if tail_last is not None:
                tail_last_int = int(tail_last)
                if tail_last_int > 0 and hist10.size > tail_last_int:
                    hist10 = hist10[-tail_last_int:]
            start10 = int(hist10.size * tail_frac)
            tail10 = hist10[start10:]
            tail10 = tail10[tail10 >= 0]
            if tail10.size > 0:
                tail10_true = inv_perm_s10[tail10]
                emp10 = np.bincount(tail10_true, minlength=K).astype(float)
                emp10 = emp10 / np.maximum(emp10.sum(), 1.0)
            else:
                emp10 = np.zeros(K, dtype=float)

            ax.bar(
                xs + 0.25,
                emp10,
                width=0.25,
                alpha=0.75,
                color=COLOR_SGLD_S10,
                label="Empirical z (Gibbs, S=10)",
            )

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("")
        ax.set_title(rf"$z_{{{g}}}$")
        ax.grid(True, alpha=0.25)

    # Turn off any unused panels (defensive)
    for ax in axes[len(track_indices):]:
        ax.axis("off")
    fig.supxlabel("cluster label")
    # Move legend outside top-center of plot, at figure level.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=0.8,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_show()


# %%
def plot_mu_trajectories(samples_means, true_means):
    """
    samples_means: (iters, K, d)
    true_means: (K, d)
    """
    est_means = samples_means.mean(axis=0)
    perm = match_clusters(true_means, est_means)
    samples_means = samples_means[:, perm, :]
    true_means = true_means

    iters = samples_means.shape[0]
    K, d = samples_means.shape[1], samples_means.shape[2]
    t = np.arange(iters)

    plt.figure(figsize=FIGSIZE_SQUARE)
    for k in range(K):
        for j in range(d):
            plt.plot(t, samples_means[:, k, j], label=f"mu[{k},{j}]")
            plt.hlines(true_means[k, j], 0, iters - 1, linestyles="dashed")
    plt.title("Trajectories of all mu parameters")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend(ncol=2)
    plt.tight_layout()
    save_and_show()


# %%
def plot_sigma_trajectories(samples_sigma, true_covs, samples_means, true_means):
    """
    samples_sigma: (iters, K, d)
    true_covs: (K, d, d)
    samples_means: (iters, K, d)
    true_means: (K, d)
    """
    # First match clusters using means
    est_means = samples_means.mean(axis=0)
    perm = match_clusters(true_means, est_means)
    samples_sigma = samples_sigma[:, perm, :]
    true_sigma = np.sqrt(np.diagonal(true_covs, axis1=1, axis2=2))

    iters = samples_sigma.shape[0]
    K, d = samples_sigma.shape[1], samples_sigma.shape[2]
    t = np.arange(iters)

    plt.figure(figsize=FIGSIZE_SQUARE)
    for k in range(K):
        for j in range(d):
            plt.plot(t, samples_sigma[:, k, j], label=f"sigma[{k},{j}]")
            plt.hlines(true_sigma[k, j], 0, iters - 1, linestyles="dashed")
    plt.title("Trajectories of all sigma parameters")
    plt.xlabel("Iteration")
    plt.ylabel("Std dev")
    plt.legend(ncol=2)
    plt.tight_layout()
    save_and_show()


# ===================== SVI benchmark for diagonal-covariance GMM =====================

def svi_gmm_diag_local(
    X,
    K,
    iters=5000,
    batch_size=25,
    tau0=10.0,
    kappa=0.7,
    seed=0,
    init_means=None,
    verbose_every=500,
    prior=None,
):
    """Stochastic variational inference (SVI) baseline for a diagonal-covariance GMM.

    This is a *variational Bayes* baseline (not online-EM):
      - q(z_n) is categorical with responsibilities r_{n,k}
      - q(mu_{k,j}, tau_{k,j}) factorizes as Normal-Gamma per (k,j)
      - mixture weights are fixed uniform pi = 1/K (matches your SGLD setup)

    Per dimension j:
      tau_{k,j} ~ Gamma(a_{k,j}, b_{k,j})  (shape a, rate b)
      mu_{k,j} | tau_{k,j} ~ Normal(m_{k,j}, (beta_{k,j} * tau_{k,j})^{-1})

    We run SVI by maintaining global *expected sufficient statistics* and updating them
    with Robbins–Monro steps using scaled minibatch stats.

    Returns:
      m: (K,d) variational mean of mu
      beta,a,b: (K,d) Normal-Gamma parameters
      mu_hist: (T,K,d) history of m
      params_hist: optional dict with histories of beta/a/b
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    N, d = X.shape

    # Prior hyperparameters (very weak by default; effectively flat-ish)
    if prior is None:
        prior = {
            "m0": np.zeros(d, dtype=float),
            "beta0": 1e-3,
            "a0": 1e-3,
            "b0": 1e-3,
        }
    m0 = np.asarray(prior["m0"], dtype=float)
    beta0 = float(prior["beta0"])
    a0 = float(prior["a0"])
    b0 = float(prior["b0"])

    # Initialize variational parameters
    if init_means is None:
        m = X[rng.choice(N, size=K, replace=False)].copy()
    else:
        m = np.asarray(init_means, dtype=float).copy()

    # Initialize with reasonable dispersion from data
    data_var = np.var(X, axis=0) + 1e-6
    # Normal-Gamma params
    beta = np.full((K, d), beta0 + 1.0, dtype=float)
    a = np.full((K, d), a0 + 0.5, dtype=float)
    b = np.tile((b0 + 0.5 * data_var)[None, :], (K, 1)).astype(float)

    # Maintain global expected sufficient statistics for each cluster:
    # Nk_hat: expected cluster counts
    # xsum_hat: sum r x
    # xsqsum_hat: sum r x^2
    Nk_hat = np.full(K, 1.0, dtype=float)
    xsum_hat = np.zeros((K, d), dtype=float)
    xsqsum_hat = np.zeros((K, d), dtype=float)

    mu_hist = []
    beta_hist = []
    a_hist = []
    b_hist = []

    # Helpers for expectations under Normal-Gamma
    def _E_tau(a_, b_):
        return a_ / np.maximum(b_, 1e-12)

    def _E_log_tau(a_, b_):
        # autograd/scipy: digamma is not in autograd.numpy; use a smooth approx
        # Here we use scipy.special.digamma if available via scipy.
        from scipy.special import digamma
        return digamma(a_) - np.log(np.maximum(b_, 1e-12))

    for t in range(1, int(iters) + 1):
        rho = (tau0 + t) ** (-kappa)

        B = min(int(batch_size), N)
        idx = rng.choice(N, size=B, replace=False)
        Xb = X[idx]

        # Compute responsibilities r_{n,k} using an expected log-likelihood under q
        # Approximate E_q[log N(x | mu_k, diag(tau_k^{-1}))]
        # = sum_j 0.5(E[log tau]-log 2pi) -0.5 E[tau] * E[(x-mu)^2]
        # with E[(x-mu)^2] ≈ (x-m)^2 + Var(mu), Var(mu) ≈ 1/(beta*E[tau]).
        Etau = _E_tau(a, b)              # (K,d)
        Elogtau = _E_log_tau(a, b)       # (K,d)
        var_mu_approx = 1.0 / np.maximum(beta * Etau, 1e-12)  # (K,d)

        # logp: (B,K)
        logp = np.zeros((B, K), dtype=float)
        const = 0.5 * (Elogtau - np.log(2.0 * np.pi))  # (K,d)
        for k in range(K):
            diff2 = (Xb - m[k]) ** 2  # (B,d)
            quad = -0.5 * np.sum(Etau[k] * (diff2 + var_mu_approx[k]), axis=1)  # (B,)
            logp[:, k] = np.sum(const[k], axis=0) + quad

        # stabilize + normalize (uniform pi)
        logp = logp - logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        r = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)  # (B,K)

        # Minibatch sufficient stats (for diagonal dims)
        Nk_b = r.sum(axis=0) + 1e-12  # (K,)
        xsum_b = r.T @ Xb             # (K,d)
        xsqsum_b = r.T @ (Xb ** 2)    # (K,d)

        # Scale to full data
        scale = float(N) / float(B)
        Nk_tilde = Nk_b * scale
        xsum_tilde = xsum_b * scale
        xsqsum_tilde = xsqsum_b * scale

        # Robbins–Monro update of global expected sufficient stats
        Nk_hat = (1.0 - rho) * Nk_hat + rho * Nk_tilde
        xsum_hat = (1.0 - rho) * xsum_hat + rho * xsum_tilde
        xsqsum_hat = (1.0 - rho) * xsqsum_hat + rho * xsqsum_tilde

        # Convert suff stats -> batch VB posterior params (per (k,j))
        # Given Nk, xsum, xsqsum:
        #   xbar = xsum/Nk
        #   S = sum r (x-xbar)^2 = xsqsum - Nk*xbar^2
        xbar = xsum_hat / np.maximum(Nk_hat[:, None], 1e-12)              # (K,d)
        S = xsqsum_hat - Nk_hat[:, None] * (xbar ** 2)                    # (K,d)
        S = np.maximum(S, 1e-12)

        # IMPORTANT: keep beta and a as (K,d) arrays (not (K,1)), since we model a per-dimension
        # Normal-Gamma factor q(mu_{k,j}, tau_{k,j}). Counts Nk_hat are shared across dims but broadcast.
        beta = beta0 + Nk_hat[:, None] * np.ones((1, d), dtype=float)
        m = (beta0 * m0[None, :] + Nk_hat[:, None] * xbar) / np.maximum(beta, 1e-12)
        a = a0 + 0.5 * Nk_hat[:, None] * np.ones((1, d), dtype=float)

        # b update includes mean-shrinkage term
        mean_shrink = 0.5 * (beta0 * Nk_hat[:, None] / np.maximum(beta, 1e-12)) * ((xbar - m0[None, :]) ** 2)
        b = b0 + 0.5 * S + mean_shrink

        mu_hist.append(m.copy())
        beta_hist.append(beta.copy())
        a_hist.append(a.copy())
        b_hist.append(b.copy())

        if verbose_every and (t % int(verbose_every) == 0):
            print(
                f"[SVI-VB] iter {t}/{iters}  rho={rho:.4g}  "
                f"Nk(min/med/max)={Nk_hat.min():.1f}/{np.median(Nk_hat):.1f}/{Nk_hat.max():.1f}"
            )

    mu_hist = np.asarray(mu_hist)
    beta_hist = np.asarray(beta_hist)
    a_hist = np.asarray(a_hist)
    b_hist = np.asarray(b_hist)

    return {
        "mu": m,
        "beta": beta,
        "a": a,
        "b": b,
        "mu_hist": mu_hist,
        "beta_hist": beta_hist,
        "a_hist": a_hist,
        "b_hist": b_hist,
        "Nk": Nk_hat,
        "_kind": "svi_vb_normal_gamma_diag",
        "tau0": float(tau0),
        "kappa": float(kappa),
        "batch_size": int(batch_size),
        "prior": {"beta0": beta0, "a0": a0, "b0": b0},
    }


def _apply_perm_to_hist(arr, perm):
    """Permute cluster axis of a history array of shape (T,K,...)"""
    return arr[:, perm, ...]

# Fixed color map for parameters (k,j)
def get_param_color_map(K, d):
    cmap = plt.get_cmap('tab10')
    colors = {}
    idx = 0
    for k in range(K):
        for j in range(d):
            colors[(k, j)] = cmap(idx % 10)
            idx += 1
    return colors


def plot_mu_trajectories_compare(samples_means_sgld, mu_hist_svi, true_means):
    """Plot SGLD and SVI mu trajectories together in one figure with true lines, using consistent colors."""
    it_sgld = samples_means_sgld.shape[0]
    it_svi = mu_hist_svi.shape[0]
    T = min(it_sgld, it_svi)

    K, d = true_means.shape
    t = np.arange(T)
    colors = get_param_color_map(K, d)

    plt.figure(figsize=FIGSIZE_SQUARE)
    for k in range(K):
        for j in range(d):
            c = colors[(k, j)]
            plt.plot(t, samples_means_sgld[:T, k, j], color=c, alpha=0.85,
                     label=f"mu[{k},{j}] SGLD")
            plt.plot(t, mu_hist_svi[:T, k, j], color=c, alpha=0.45, linestyle='--',
                     label=f"mu[{k},{j}] SVI")
            plt.hlines(true_means[k, j], 0, T - 1, colors=c,
                       linestyles='dotted', linewidth=2.0,
                       label=f"mu[{k},{j}] TRUE")
    plt.title("Trajectories of mu (matched): SGLD vs SVI vs TRUE")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend(ncol=3)
    plt.tight_layout()
    save_and_show()


# ===================== Helper: plot trajectories for a subset of mu parameters =====================
def plot_mu_iterates_subset(
    samples_means,
    true_means,
    *,
    params=None,
    max_params=10,
    title="Mu iterates",
    stride=50,
):
    """Plot trajectories (iterates) for a subset of mu parameters.

    Args:
      samples_means: (T,K,d) array
      true_means: (K,d) array
      params: optional list of (k,j) tuples. If None, auto-select up to max_params.
      max_params: number of params to plot if params is None
      title: plot title
      stride: thinning for visualization (plot every `stride` iterations)
    """
    samples_means = np.asarray(samples_means, dtype=float)
    true_means = np.asarray(true_means, dtype=float)
    T, K, d = samples_means.shape

    # Default: start with a curated set, then fill.
    if params is None:
        curated = [(0, 1), (1, 1), (2, 0), (4, 1), (0, 0), (1, 4), (2, 7), (3, 1), (3, 3), (4, 5)]
        idxs = []
        used = set()
        for (k, j) in curated:
            k = int(k); j = int(j)
            if 0 <= k < K and 0 <= j < d and (k, j) not in used:
                idxs.append((k, j))
                used.add((k, j))
        max_params = int(max_params)
        for k in range(K):
            for j in range(d):
                if len(idxs) >= max_params:
                    break
                if (k, j) not in used:
                    idxs.append((k, j))
                    used.add((k, j))
            if len(idxs) >= max_params:
                break
        params = idxs[:max_params]
    else:
        params = [(int(k), int(j)) for (k, j) in params]
        params = [(k, j) for (k, j) in params if (0 <= k < K and 0 <= j < d)]
        params = params[: int(max_params)]

    stride = max(int(stride), 1)
    t = np.arange(0, T, stride)

    plt.figure(figsize=FIGSIZE_SQUARE)
    for (k, j) in params:
        series = samples_means[::stride, k, j]
        plt.plot(t, series, linewidth=1.6, alpha=0.90, label=rf"$\mu_{{{k},{j}}}$")
        # True reference line
        plt.hlines(float(true_means[k, j]), t[0], t[-1], linestyles='dotted', linewidth=1.8, alpha=0.75)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(r"$\mu$")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"[plot] tight_layout failed: {e}")
    save_and_show()


def plot_sigma_trajectories_compare(samples_sigma_sgld, sigma_hist_svi, true_covs, true_means, samples_means_sgld):
    """Plot SGLD and SVI sigma trajectories together in one figure with true lines, using consistent colors."""
    # match based on means already done upstream; true sigma from true_covs
    true_sigma = np.sqrt(np.diagonal(true_covs, axis1=1, axis2=2))

    it_sgld = samples_sigma_sgld.shape[0]
    it_svi = sigma_hist_svi.shape[0]
    T = min(it_sgld, it_svi)

    K, d = true_sigma.shape
    t = np.arange(T)
    colors = get_param_color_map(K, d)
    plt.figure(figsize=FIGSIZE_SQUARE)
    for k in range(K):
        for j in range(d):
            c = colors[(k, j)]
            plt.plot(t, samples_sigma_sgld[:T, k, j], color=c, alpha=0.85,
                     label=f"sigma[{k},{j}] SGLD")
            plt.plot(t, sigma_hist_svi[:T, k, j], color=c, alpha=0.45, linestyle='--',
                     label=f"sigma[{k},{j}] SVI")
            plt.hlines(true_sigma[k, j], 0, T - 1, colors=c,
                       linestyles='dotted', linewidth=2.0,
                       label=f"sigma[{k},{j}] TRUE")
    plt.title("Trajectories of sigma (matched): SGLD vs SVI vs TRUE")
    plt.xlabel("Iteration")
    plt.ylabel("Std dev")
    plt.legend(ncol=3)
    plt.tight_layout()
    save_and_show()


def plot_posterior_mu_sgld_vs_svi(samples_means_sgld, mu_svi, beta_svi, a_svi, b_svi, true_means, *, tail_T=3000, max_params=16, samples_means_s10=None):
    """Posterior comparison for mu: SGLD tail Gaussian fit vs SVI Gaussian approx.

    Plots are arranged in a 2x2 grid (for the requested 4 parameters) with a single shared legend.
    """
    T = samples_means_sgld.shape[0]
    tail = samples_means_sgld[-min(tail_T, T):]
    tail_s10 = None
    if samples_means_s10 is not None:
        samples_means_s10 = np.asarray(samples_means_s10, dtype=float)
        T10 = samples_means_s10.shape[0]
        tail_s10 = samples_means_s10[-min(tail_T, T10):]

    K, d = mu_svi.shape

    # --- Fixed 2x2 panel: user-selected parameters ---
    # Requested (k,j): (4,0), (5,1), (0,2), (2,0)
    idxs = [(4, 0), (5, 1), (0, 2), (2, 0)]

    # Keep only valid indices in case K or d changes
    idxs = [(int(k), int(j)) for (k, j) in idxs if (0 <= int(k) < K and 0 <= int(j) < d)]

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_SQUARE, sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, (k, j) in zip(axes, idxs):
        series = np.asarray(tail[:, k, j], dtype=float)

        # SGLD Gaussian approximation fit to tail samples (mean + std)
        m_sgld = float(np.mean(series))
        s_sgld = float(np.std(series, ddof=1))
        if not np.isfinite(s_sgld) or s_sgld <= 0:
            s_sgld = 1e-6

        # SVI-VB (Normal-Gamma) Gaussian marginal approx for mu_{k,j}
        m = float(mu_svi[k, j])
        a_kj = float(a_svi[k, j])
        b_kj = float(b_svi[k, j])
        beta_kj = float(beta_svi[k, j])
        denom = (a_kj - 1.0) if (a_kj > 1.0) else max(a_kj, 1e-6)
        s2 = b_kj / max(denom * beta_kj, 1e-12)
        s = np.sqrt(max(s2, 1e-12))
        # Use a plotting range that covers both SGLD-fit and SVI Gaussian spreads
        x_lo = min(m_sgld - 5 * s_sgld, m - 5 * s)
        x_hi = max(m_sgld + 5 * s_sgld, m + 5 * s)
        xs = np.linspace(x_lo, x_hi, 400)

        # Plot SGLD posterior (red)
        ax.plot(
            xs,
            norm.pdf(xs, loc=m_sgld, scale=s_sgld),
            color=COLOR_SGLD,
            linewidth=2.0,
            alpha=0.90,
            label="SGLD",
        )

        # Optional: SGLD averaged-Gibbs S=10 posterior overlay (deep red solid)
        if tail_s10 is not None:
            series10 = np.asarray(tail_s10[:, k, j], dtype=float)
            m_s10 = float(np.mean(series10))
            s_s10 = float(np.std(series10, ddof=1))
            if not np.isfinite(s_s10) or s_s10 <= 0:
                s_s10 = 1e-6
            ax.plot(
                xs,
                norm.pdf(xs, loc=m_s10, scale=s_s10),
                color=COLOR_SGLD_S10,
                linewidth=2.2,
                linestyle='-',
                alpha=0.95,
                label="SGLD (S=10)",
            )

        # Plot SVI posterior (blue)
        ax.plot(
            xs,
            norm.pdf(xs, loc=m, scale=s),
            color=COLOR_SVI,
            linewidth=2.0,
            alpha=0.90,
            label="SVI",
        )

        # TRUE reference (gray)
        tv = float(true_means[k, j])
        ax.axvline(
            tv,
            color=COLOR_REF,
            linestyle='dotted',
            linewidth=2.0,
            alpha=0.90,
            label=r"True $\mu_*$",
        )

        ax.set_title(rf"$\mu_{{{k},{j}}}$")
        ax.grid(True, alpha=0.25)
        ax.grid(True, alpha=0.25)

    # Turn off any unused axes
    for ax in axes[len(idxs):]:
        ax.axis('off')

    # Single shared legend (avoid clutter)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(labels),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.6,
            columnspacing=0.8,
        )
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    except Exception as e:
        print(f"[plot] tight_layout failed: {e}")
    save_and_show()
def plot_posterior_mu_sgld_vs_sgld(
    samples_means_a,
    samples_means_b,
    true_means,
    *,
    tail_T=3000,
    max_params=5,
    label_a="SGLD (S=1)",
    label_b="SGLD (avg Gibbs, S=10)",
):
    """Posterior comparison for mu: two SGLD variants via Gaussian fits to tail samples."""
    samples_means_a = np.asarray(samples_means_a, dtype=float)
    samples_means_b = np.asarray(samples_means_b, dtype=float)
    true_means = np.asarray(true_means, dtype=float)

    Ta = samples_means_a.shape[0]
    Tb = samples_means_b.shape[0]
    tail_a = samples_means_a[-min(int(tail_T), Ta):]
    tail_b = samples_means_b[-min(int(tail_T), Tb):]

    K, d = true_means.shape

    requested = [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1)]
    idxs, used = [], set()
    for (k, j) in requested:
        if 0 <= k < K and 0 <= j < d and (k, j) not in used:
            idxs.append((k, j))
            used.add((k, j))

    if max_params is None:
        max_params = len(idxs)
    max_params = int(max_params)

    all_idx = [(k, j) for k in range(K) for j in range(d)]
    for (k, j) in all_idx:
        if len(idxs) >= min(max_params, len(all_idx)):
            break
        if (k, j) not in used:
            idxs.append((k, j))
            used.add((k, j))

    colors = get_param_color_map(K, d)

    fig, axes = plt.subplots(len(idxs), 1, figsize=FIGSIZE_SQUARE, sharex=False)
    if len(idxs) == 1:
        axes = [axes]

    for ax, (k, j) in zip(axes, idxs):
        c = colors[(k, j)]
        sa = np.asarray(tail_a[:, k, j], dtype=float)
        sb = np.asarray(tail_b[:, k, j], dtype=float)

        ma = float(np.mean(sa))
        mb = float(np.mean(sb))
        s_a = float(np.std(sa, ddof=1))
        s_b = float(np.std(sb, ddof=1))
        if not np.isfinite(s_a) or s_a <= 0:
            s_a = 1e-6
        if not np.isfinite(s_b) or s_b <= 0:
            s_b = 1e-6

        tv = float(true_means[k, j])
        x_lo = min(ma - 5 * s_a, mb - 5 * s_b, tv - 5 * max(s_a, s_b))
        x_hi = max(ma + 5 * s_a, mb + 5 * s_b, tv + 5 * max(s_a, s_b))
        xs = np.linspace(x_lo, x_hi, 400)

        ax.plot(xs, norm.pdf(xs, loc=ma, scale=s_a), color=c, linewidth=2.0, alpha=0.85,
                label=f"{label_a} Gaussian fit")
        ax.plot(xs, norm.pdf(xs, loc=mb, scale=s_b), color=c, linewidth=2.0, alpha=0.75,
                linestyle="--", label=f"{label_b} Gaussian fit")
        ax.axvline(tv, color=c, linestyle="dotted", linewidth=2.0, label=f"TRUE={tv:.3f}")

        ax.set_title(f"Posterior for mu[{k},{j}] (tail_T={min(int(tail_T), Ta, Tb)})")
        ax.grid(True, alpha=0.25)
        # Remove per-axis legend for multi-subplot figure

    # Single shared legend outside axes
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(labels),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.6,
            columnspacing=0.8,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_show()

# ==============================================================
# Density overlays for centered+scaled sqrt(N)*(mu - mean)
def plot_scaled_mu_density_overlays(
    samples_means_sgld,
    N,
    w1,
    w2,
    true_covs,
    *,
    tail_len=5000,
    stride=10,
    max_params=16,
    params=None,
    bins=50,
):
    """Density plots for centered+scaled sqrt(N)*(mu - mean) using the same tail/thinning as variance calc.

    Overlays:
      1) Empirical density from SGLD samples (hist)
      2) Normal(0, s_emp): Gaussian fit using empirical std of the scaled samples
      3) Normal(0, s_pred): predicted std from theory sqrt((w1+w2)*K*sigma^2)

    Args:
      samples_means_sgld: array (T,K,d)
      N: dataset size
      w1,w2: hyperparameters
      true_covs: (K,d,d)
      tail_len: number of final iterations to use (before thinning)
      stride: thinning stride
      max_params: number of (k,j) params to plot if params is None
      params: optional list of (k,j) tuples to plot
      bins: histogram bins
    """
    samples_means_sgld = np.asarray(samples_means_sgld, dtype=float)
    T, K, d = samples_means_sgld.shape

    L = min(int(tail_len), T)
    tail = samples_means_sgld[-L:, :, :][:: int(stride)]
    # center by the tail mean (same as in variance computation)
    centered = tail - tail.mean(axis=0, keepdims=True)
    scaled = np.sqrt(float(N)) * centered  # (L/stride, K, d)

    # Default params for the first "empirical vs predicted" density figure:
    # plot exactly these 9 entries (k,j) in this order:
    # (0,0), (0,1), (1,4), (1,5), (2,7), (2,0), (3,1), (3,3), (4,5)
    if params is None:
        params = [(0, 0), (0, 1), (1, 4), (1, 5), (2, 7), (2, 0), (3, 1), (3, 3), (4, 5)]

    # keep only valid indices (in case K or d changes)
    params = [(int(k), int(j)) for (k, j) in params if (0 <= int(k) < K and 0 <= int(j) < d)]

    # predicted sigma from true_covs
    true_sigma2 = np.diagonal(true_covs, axis1=1, axis2=2)  # (K,d)

    # Arrange panels in a fixed 3x3 grid (exactly 9 params)
    n_plots = min(len(params), 9)
    params = params[:n_plots]

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_SQUARE, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, (k, j) in zip(axes, params):
        k = int(k)
        j = int(j)
        series = np.asarray(scaled[:, k, j], dtype=float)

        # empirical std (Gaussian fit)
        s_emp = float(np.std(series, ddof=1))
        # predicted std
        var_pred = float((w1 + w2) * K * true_sigma2[k, j])
        s_pred = float(np.sqrt(max(var_pred, 1e-12)))

        # histogram density: light red (empirical samples)
        ax.hist(
            series,
            bins=int(bins),
            density=True,
            alpha=0.25,
            color=COLOR_SGLD,
            label="Samples",
        )

        # overlay normals
        xs = np.linspace(series.min() - 0.5 * s_emp, series.max() + 0.5 * s_emp, 400)

        # Empirical Gaussian fit: SGLD red
        ax.plot(
            xs,
            norm.pdf(xs, loc=0.0, scale=max(s_emp, 1e-12)),
            linestyle='-',
            linewidth=2.2,
            color=COLOR_SGLD,
            label="Empirical",
        )

        # Predicted: gray reference
        ax.plot(
            xs,
            norm.pdf(xs, loc=0.0, scale=max(s_pred, 1e-12)),
            linestyle='-',
            linewidth=2.0,
            color=COLOR_REF,
            label="Predicted",
        )

        ax.set_title(rf"$\mu_{{{k},{j}}}$", fontweight="bold")
        ax.grid(True, alpha=0.25)
        # Do NOT set axis labels here (handled at figure level)

    # Control tick labels: only leftmost and bottom panels show y/x ticks
    for idx, ax in enumerate(axes):
        r = idx // ncols
        c = idx % ncols
        if c != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        if r != nrows - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])

    # Add shared axis labels at the figure level
    fig.supxlabel(r"$n^{\mathfrak w}(\mu - \hat{\mu})$")

    # Turn off any unused axes
    for ax in axes[len(params):]:
        ax.axis('off')

    # One shared legend (avoid clutter)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(labels),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.6,
            columnspacing=0.8,
        )
    # tight_layout can fail when mathtext/legends get complex; fall back gracefully.
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    except Exception as e:
        print(f"[plot] tight_layout failed: {e}")
    save_and_show()


def plot_rank_uniformity_compare_mu(samples_means_sgld, mu_svi, beta_svi, a_svi, b_svi, true_means, *, start_frac=0.5):
    """
    Rank-uniformity compare for ALL mu entries (K*d): SGLD tail vs SVI-VB (Normal-Gamma).
    """
    T = samples_means_sgld.shape[0]
    start = int(T * start_frac)
    tail = samples_means_sgld[start:]

    K, d = true_means.shape
    ps_sgld = []
    ps_svi = []

    for k in range(K):
        for j in range(d):
            series = np.asarray(tail[:, k, j], dtype=float)
            tv = float(true_means[k, j])
            ps_sgld.append(float(np.mean(series > tv)))

            m = float(mu_svi[k, j])
            a_kj = float(a_svi[k, j])
            b_kj = float(b_svi[k, j])
            beta_kj = float(beta_svi[k, j])
            # This is the marginal variance of mu under the Normal-Gamma variational posterior (SVI-VB Normal-Gamma).
            denom = (a_kj - 1.0) if (a_kj > 1.0) else max(a_kj, 1e-6)
            s2 = b_kj / max(denom * beta_kj, 1e-12)
            s = np.sqrt(max(s2, 1e-12))
            ps_svi.append(float(1.0 - norm.cdf(tv, loc=m, scale=s)))

    ps_sgld = np.sort(np.asarray(ps_sgld))
    ps_svi = np.sort(np.asarray(ps_svi))

    # Note: we use q = (i-0.5)/D for a standard QQ-style uniform quantile grid

    # Calibration / rank-uniformity plot (QQ-style): sorted p-values vs Uniform(0,1) quantiles
    # ICML-friendly: connected lines + light reference diagonal
    Dp = ps_sgld.size
    q = (np.arange(1, Dp + 1) - 0.5) / Dp  # uniform quantiles in (0,1)

    plt.figure(figsize=FIGSIZE_SQUARE)

    # Reference diagonal (gray)
    plt.plot(
        q,
        q,
        linestyle='-',
        linewidth=2.0,
        alpha=0.70,
        color=COLOR_REF,
        label='Uniform reference',
    )

    # SGLD (red)
    plt.plot(
        q,
        ps_sgld,
        linestyle='-',
        marker='o',
        markersize=3.5,
        linewidth=1.2,
        alpha=0.90,
        color=COLOR_SGLD,
        label='SGLD',
    )

    # SVI (blue)
    plt.plot(
        q,
        ps_svi,
        linestyle='--',
        marker='x',
        markersize=4.0,
        linewidth=1.2,
        alpha=0.90,
        color=COLOR_SVI,
        label='SVI',
    )

    plt.xlabel("Uniform quantile")
    plt.ylabel(r"Empirical $P(\mu > \mu^\star)$ (sorted)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    save_and_show()


# %%
def main():
    """
    Run multiple independent SGLD+Gibbs chains (option c) and
    average the empirical stationary variances of sqrt(N)*mu.
    We still plot data and trajectories for the FIRST chain.
    """
    n_chains = 1

    # Use tail mean for label matching (avoid burn-in contamination)
    tail_len_match = 5000

    # Random seed for averaged-Gibbs SGLD (sgld-avg)
    sgld_avg_seed = 123

    var_scaled_list = []
    sigma2_tail_list = []
    first_chain_result = None
    first_chain_samples_means = None
    first_chain_samples_sigma = None
    first_chain_inv_perm = None
    first_chain_perm = None

    for r in range(n_chains):
        print(f"\n=== Running chain {r+1} / {n_chains} ===")
        np.random.seed(23 + r)  # per-chain reproducible seed
        result = run_sgld_gibbs_example()

        X = result["X"]
        z_true = result["z_true"]
        true_means = result["true_means"]
        samples_means = np.array(result["samples_means"])   # (iters, K, d)
        samples_sigma = np.array(result["samples_sigma"])   # (iters, K, d)

        # Match clusters using TAIL mean of samples_means (last 5000 iters)
        tail_len_match_eff = min(5000, samples_means.shape[0])
        est_means = samples_means[-tail_len_match_eff:].mean(axis=0)              # (K, d)
        perm = match_clusters(true_means, est_means)
        inv_perm = np.argsort(perm)  # map estimated cluster id -> true cluster id
        samples_means = samples_means[:, perm, :]
        samples_sigma = samples_sigma[:, perm, :]

        # Mean L2 error between SGLD-Gibbs posterior mean and TRUE means (after tail-based matching)
        sgld_mean_est = samples_means.mean(axis=0)  # (K, d), already matched to TRUE order
        mean_l2_sgld = np.mean(np.linalg.norm(sgld_mean_est - true_means, axis=1))
        print(f"[SGLD-Gibbs] Mean L2 error (posterior mean vs TRUE means): {mean_l2_sgld:.6f}")

        T = samples_means.shape[0]
        tail_len = min(5000, T)

        stride = 10  # thinning stride

        # Tail for mu (with thinning)
        tail_mu_full = samples_means[-tail_len:, :, :]      # (tail_len, K, d)
        tail_mu = tail_mu_full[::stride, :, :]              # thinned tail
        centered_mu = tail_mu - tail_mu.mean(axis=0, keepdims=True)
        scaled_mu = np.sqrt(result["N"]) * centered_mu      # sqrt(N) * (mu - mean)
        var_scaled = np.var(scaled_mu, axis=0, ddof=1)      # (K, d)

        # Tail for sigma (with thinning)
        tail_sigma_full = samples_sigma[-tail_len:, :, :]   # (tail_len, K, d)
        tail_sigma = tail_sigma_full[::stride, :, :]        # thinned tail
        mean_sigma_tail = tail_sigma.mean(axis=0)           # (K, d)
        sigma2_tail = mean_sigma_tail ** 2                  # (K, d)

        var_scaled_list.append(var_scaled)
        sigma2_tail_list.append(sigma2_tail)

        if r == 0:
            first_chain_result = result
            first_chain_samples_means = samples_means
            first_chain_samples_sigma = samples_sigma
            first_chain_inv_perm = inv_perm
            first_chain_perm = perm

    # Single-chain empirical variances and sigma^2
    var_scaled_avg = var_scaled_list[0]
    sigma2_tail_avg = sigma2_tail_list[0]

    print("\nSingle-chain results:")
    K, d = var_scaled_avg.shape
    w1 = first_chain_result["w1"]
    w2 = first_chain_result["w2"]

    for k in range(K):
        for j in range(d):
            var_emp = var_scaled_avg[k, j]
            var_theory = (w1 + w2) * K * sigma2_tail_avg[k, j]
            print(
                f"  Param mu[{k},{j}]: Var(sqrt(N)*mu) = {var_emp:.6f}, "
                f"Theory ((w1+w2)*K*sigma^2) = {var_theory:.6f}"
            )

    # Density overlays for the same centered+scaled samples used in Var(sqrt(N)*mu)
    plot_scaled_mu_density_overlays(
        first_chain_samples_means,
        N=first_chain_result["N"],
        w1=w1,
        w2=w2,
        true_covs=first_chain_result["true_covs"],
        tail_len=5000,
        stride=stride,
        max_params=16,
    )
    # Use the first chain for plotting
    X = first_chain_result["X"]
    z_true = first_chain_result["z_true"]
    true_means = first_chain_result["true_means"]

    # 1. Plot raw data and true means
    plot_data_and_true_means(X, z_true, true_means)

    # Plot predicted vs sampled z for tracked datapoints
    plot_predicted_vs_sampled_z_for_points(
        X,
        true_means,
        first_chain_result["true_covs"],
        first_chain_result["true_weights"],
        first_chain_result["z_track_history"],
        first_chain_result["track_indices"],
        first_chain_inv_perm,
        tail_frac=0.5,
        tail_last=20000,
    )

    # Trajectories for a subset of mu parameters (S=1)
    plot_mu_iterates_subset(
        first_chain_samples_means,
        true_means,
        max_params=10,
        title="Mu iterates (SGLD, S=1): 10 parameters",
        stride=50,
    )

    # ===================== Run SVI benchmark on the same dataset =====================
    # Use the external SVI (VB / Normal-Gamma) implementation from svi.py.
    # We use the flat+fixed-pi variant so it matches the SGLD setup (pi=1/K, very weak priors).
    svi_iters = 20000  # match SGLD iterations for easier comparison
    svi_out = svi_gmm_diag_flat(
        X,
        K=first_chain_result["K"],
        iters=svi_iters,
        batch_size=25,
        tau0=10.0,
        kappa=0.7,
        seed=123,
        init_method="gmm_em",  # requested: EM-GMM init
    )

    # svi.py uses key "m" for the variational mean of mu
    mu_svi = np.asarray(svi_out["m"], dtype=float)
    beta_svi = np.asarray(svi_out["beta"], dtype=float)
    a_svi = np.asarray(svi_out["a"], dtype=float)
    b_svi = np.asarray(svi_out["b"], dtype=float)

    # Match SVI clusters to TRUE using final mu
    perm_svi = match_clusters(true_means, mu_svi)
    mu_svi = mu_svi[perm_svi]
    beta_svi = beta_svi[perm_svi]
    a_svi = a_svi[perm_svi]
    b_svi = b_svi[perm_svi]

    # Calibration plot before running the S=10 averaged-Gibbs variant
    plot_rank_uniformity_compare_mu(
        first_chain_samples_means,
        mu_svi,
        beta_svi,
        a_svi,
        b_svi,
        true_means,
        start_frac=0.5,
    )

    # ===================== Store for overlay: S=10 samples (init to None) =====================
    samples_means_s10_for_overlay = None

    # ===================== Tracking comparison plots =====================
    # The flat SVI baseline does not return per-iteration histories here; we focus on posterior/cali plots.

    # ===================== Extra run: averaged-Gibbs SGLD with S=10 =====================
    if run_avg_gibbs is None:
        print("[SGLD-avg] run_avg_gibbs is not available in sgld_gibbs_fixed_cov2.py; skipping S=10 comparison.")
    else:
        print("\n=== Running averaged-Gibbs SGLD (run_avg_gibbs) on the SAME dataset for posterior comparison ===")

        # Re-seed so the averaged-Gibbs run is reproducible
        np.random.seed(int(sgld_avg_seed))
        print(f"[SGLD-avg] random seed = {sgld_avg_seed}")

        out_s10 = run_avg_gibbs(
            K=first_chain_result["K"],
            d=first_chain_result["d"],
            X=first_chain_result["X"],
            stepsize=first_chain_result["stepsize"],
            batch_size=first_chain_result["batch_size"],
            iters=first_chain_result["iters"],
            true_means=first_chain_result["true_means"],
            true_covs=first_chain_result["true_covs"],
            true_weights=first_chain_result["true_weights"],
            inverse_temperature=first_chain_result["N"] / first_chain_result["w2"],
            use_precond=True,
            gibbs_S=10,
        )
        print(f"[SGLD-avg] gibbs_S used = {1}")
        # run_avg_gibbs may return 6 or 7 values depending on implementation.
        if isinstance(out_s10, (tuple, list)) and len(out_s10) == 7:
            (mu_s10, samples_means_s10, sigma_s10, samples_sigma_s10, z0_hist_s10, _, _) = out_s10
        elif isinstance(out_s10, (tuple, list)) and len(out_s10) == 6:
            (mu_s10, samples_means_s10, sigma_s10, samples_sigma_s10, z0_hist_s10, _) = out_s10
        else:
            raise ValueError(f"run_avg_gibbs returned unexpected object/arity: {type(out_s10)} / {getattr(out_s10, '__len__', lambda: 'NA')()}")

        samples_means_s10 = np.asarray(samples_means_s10, dtype=float)

        # Match clusters using TAIL mean (last 5000 iters)
        true_means_ref = np.asarray(first_chain_result["true_means"], dtype=float)  # (K,d)
        tail_len_match_eff = min(5000, samples_means_s10.shape[0])
        est_means_s10 = np.asarray(samples_means_s10[-tail_len_match_eff:].mean(axis=0), dtype=float)     # (K,d)
        perm_s10_true = match_clusters(true_means_ref, est_means_s10)               # true i -> s10 j

        # Apply permutation to put S=10 samples into TRUE cluster order
        samples_means_s10 = samples_means_s10[:, perm_s10_true, :]

        # Also permute sigma samples if you use them later
        try:
            samples_sigma_s10 = np.asarray(samples_sigma_s10, dtype=float)
            samples_sigma_s10 = samples_sigma_s10[:, perm_s10_true, :]
        except Exception:
            pass

        # Diagnostic: check permutation direction explicitly
        perm_s10_true = np.asarray(perm_s10_true, dtype=int)  # interpreted as true i -> est j
        inv_perm_s10 = np.argsort(perm_s10_true)              # est j -> true i

        # (A) If perm is true->est, then est_means_s10[perm] is est means ordered by TRUE labels
        mean_match_err_true2est = np.mean(
            np.linalg.norm(true_means_ref - est_means_s10[perm_s10_true], axis=1)
        )

        # (B) If perm were accidentally est->true, the "correct" comparison would be:
        #     true_means_ref[perm] vs est_means_s10  (est ordered already)
        mean_match_err_est2true = np.mean(
            np.linalg.norm(true_means_ref[perm_s10_true] - est_means_s10, axis=1)
        )

        # Posterior-mean L2 after actually permuting samples_means_s10 into TRUE order
        sgld_s10_mean_est = samples_means_s10.mean(axis=0)
        mean_l2_sgld_s10 = np.mean(np.linalg.norm(sgld_s10_mean_est - true_means_ref, axis=1))

        print(f"[SGLD-avg] perm (true->est) = {perm_s10_true.tolist()}")
        print(f"[SGLD-avg] mean L2 (true_means - est_means[perm]): {mean_match_err_true2est:.6f}")
        print(f"[SGLD-avg] mean L2 (true_means[perm] - est_means): {mean_match_err_est2true:.6f}")
        print(f"[SGLD-avg] mean L2 (posterior mean vs TRUE means): {mean_l2_sgld_s10:.6f}")

        # Sanity: if the samples were permuted correctly, mean_l2_sgld_s10 should match (A) up to numerical noise.
        if abs(mean_l2_sgld_s10 - mean_match_err_true2est) > 1e-6:
            print("[SGLD-avg] WARNING: posterior-mean L2 != mean-match L2. This suggests a mismatch in perm application or label switching within the chain.")

        # Store for overlay on the SGLD vs SVI posterior comparison figure
        samples_means_s10_for_overlay = samples_means_s10

        # Trajectories for a subset of mu parameters (S=10)
        plot_mu_iterates_subset(
            samples_means_s10_for_overlay,
            first_chain_result["true_means"],
            max_params=10,
            title="Mu iterates (SGLD, S=10): 10 parameters",
            stride=50,
        )

        # Posterior compare: S=1 vs S=10 (Gaussian fits to tail)
        plot_posterior_mu_sgld_vs_sgld(
            first_chain_samples_means,
            samples_means_s10,
            first_chain_result["true_means"],
            tail_T=3000,
            max_params=5,
            label_a="SGLD (S=1)",
            label_b="SGLD (avg Gibbs, S=10)",
        )

        # Visual check: color datapoints by estimated clusters under SGLD-avg posterior mean params
        try:
            est_sigma_s10_mean_plot = np.asarray(samples_sigma_s10.mean(axis=0), dtype=float)  # (K,d), already permuted to TRUE order
        except Exception:
            # Fallback if sigma samples are unavailable
            est_sigma_s10_mean_plot = np.sqrt(np.maximum(np.diagonal(first_chain_result["true_covs"], axis1=1, axis2=2), 1e-12))

        plot_data_with_estimated_clusters(
            first_chain_result["X"],
            true_means_ref,
            sgld_s10_mean_est,
            est_sigma_s10_mean_plot,
            title=f"SGLD-avg (gibbs_S={1}): data colored by MAP cluster (posterior mean params)",
            max_points=8000,
            seed=0,
        )

    # ===================== Posterior comparison + calibration (mu) =====================
    plot_posterior_mu_sgld_vs_svi(
        first_chain_samples_means,
        mu_svi,
        beta_svi,
        a_svi,
        b_svi,
        true_means,
        tail_T=3000,
        max_params=16,
        samples_means_s10=samples_means_s10_for_overlay,
    )


# %%
if __name__ == "__main__":
    main()