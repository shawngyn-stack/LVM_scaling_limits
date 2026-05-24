# svi.py
import numpy as np
from scipy.special import digamma, logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

def _init_from_mu_sigma_pi(X, K, seed, init_mu0=None, init_sigma0=None, init_pi0=None, init_method="kmeans"):
    """Return (m_init, sigma_init, pi_init, r_init) for GMM SVI.

    - m_init: (K,d)
    - sigma_init: (K,d) diagonal std devs
    - pi_init: (K,) mixture weights (may be None/ignored by some callers)
    - r_init: (N,K) hard responsibilities from argmin diagonal Mahalanobis distance
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    N, d = X.shape

    # Prefer explicit init if provided
    if init_mu0 is not None:
        m = np.asarray(init_mu0, dtype=float)
        if m.shape != (K, d):
            raise ValueError(f"init_mu0 must have shape {(K, d)}, got {m.shape}")
    else:
        m = None

    if init_sigma0 is not None:
        sigma = np.asarray(init_sigma0, dtype=float)
        if sigma.shape != (K, d):
            raise ValueError(f"init_sigma0 must have shape {(K, d)}, got {sigma.shape}")
        sigma = np.maximum(sigma, 1e-6)
    else:
        sigma = None

    if init_pi0 is not None:
        pi = np.asarray(init_pi0, dtype=float)
        if pi.shape != (K,):
            raise ValueError(f"init_pi0 must have shape {(K,)}, got {pi.shape}")
        pi = np.maximum(pi, 1e-12)
        pi = pi / pi.sum()
    else:
        pi = None

    # If mu/sigma missing, build them via chosen method
    if (m is None) or (sigma is None):
        method = str(init_method)
        if method == "gmm_em":
            gm = GaussianMixture(
                n_components=K,
                covariance_type="diag",
                n_init=10,
                max_iter=200,
                reg_covar=1e-6,
                random_state=int(seed),
                init_params="kmeans",
            )
            gm.fit(X)
            m = np.asarray(gm.means_, dtype=float)
            sigma = np.sqrt(np.asarray(gm.covariances_, dtype=float))
            if pi is None:
                pi = np.asarray(gm.weights_, dtype=float)
        elif method == "kmeans":
            km = KMeans(n_clusters=K, n_init=10, random_state=int(seed))
            z_init = km.fit_predict(X)
            m = km.cluster_centers_.copy()
            # diag std devs from cluster sample variance (fallback to global if empty)
            sigma = np.empty((K, d), dtype=float)
            global_std = np.std(X, axis=0, ddof=0) + 1e-6
            for k in range(K):
                Xk = X[z_init == k]
                if Xk.shape[0] >= 2:
                    sigma[k] = np.std(Xk, axis=0, ddof=0) + 1e-6
                else:
                    sigma[k] = global_std
            if pi is None:
                counts = np.bincount(z_init, minlength=K).astype(float)
                pi = counts / np.sum(counts)
        elif method == "random":
            m = rng.standard_normal((K, d))
            sigma = rng.uniform(0.5, 2.0, size=(K, d))
            if pi is None:
                pi = np.ones(K) / K
        else:
            raise ValueError(f"Unknown init_method: {method}")

    # Build hard responsibilities via diagonal Mahalanobis distance
    # dist_{n,k} = sum_j ((x_nj - m_kj)^2 / sigma_kj^2)
    inv_sigma2 = 1.0 / (sigma ** 2)
    # (N,K,d)
    dif = X[:, None, :] - m[None, :, :]
    dist = np.sum(dif * dif * inv_sigma2[None, :, :], axis=2)  # (N,K)
    z_hard = np.argmin(dist, axis=1)
    r_init = np.zeros((N, K), dtype=float)
    r_init[np.arange(N), z_hard] = 1.0

    return m, sigma, pi, r_init

def svi_gmm_diag(
    X,
    K,
    iters=3000,
    batch_size=256,
    seed=0,
    # Priors
    alpha0=1.0,      # Dirichlet concentration
    m0=None,         # prior mean (d,)
    beta0=1.0,       # Normal-Gamma mean precision
    a0=2.0,          # Normal-Gamma shape
    b0=2.0,          # Normal-Gamma rate
    # Robbins–Monro schedule
    tau0=10.0,
    kappa=0.7,
    # Initialization (shared with SGLD)
    init_method="kmeans",
    init_mu0=None,
    init_sigma0=None,
    init_pi0=None,
    **kwargs,
):
    """
    Stochastic Variational Inference for a GMM with diagonal covariance.

    Model:
      pi ~ Dir(alpha0)
      z_n ~ Cat(pi)
      x_n | z_n=k ~ N(mu_k, diag(tau_k^{-1}))

    Variational family:
      q(pi) = Dir(alpha)
      q(mu_kj, tau_kj) = Normal-Gamma(m, beta, a, b)
      q(z_n) = Cat(r_n)

    Returns a dict of variational parameters.
    """
    # Backward-compat: allow callers to pass init=... instead of init_method=...
    if ("init" in kwargs) and (init_method is None or init_method == "kmeans"):
        init_method = kwargs["init"]
    rng = np.random.default_rng(seed)
    N, d = X.shape
    if m0 is None:
        m0 = np.zeros(d)

    # Initialization (prefer explicit init_mu0/init_sigma0/init_pi0)
    m, sigma_init, pi_init, r_init = _init_from_mu_sigma_pi(
        X, K, seed,
        init_mu0=init_mu0,
        init_sigma0=init_sigma0,
        init_pi0=init_pi0,
        init_method=init_method,
    )

    # Global variational params for q(pi)
    # Initialize alpha using hard responsibilities (and pi_init if provided for mild smoothing)
    alpha = np.full(K, alpha0, dtype=float) + r_init.sum(axis=0)
    if pi_init is not None:
        alpha = alpha + 0.1 * N * pi_init

    # Initialize Normal-Gamma factors. If we have sigma_init, match E[tau]=a/b to 1/sigma^2.
    beta = np.full((K, d), float(beta0), dtype=float)
    a = np.full((K, d), float(a0), dtype=float)
    b = np.full((K, d), float(b0), dtype=float)
    if sigma_init is not None:
        E_tau_target = 1.0 / (sigma_init ** 2)
        # Keep a at least >1 to have finite var; use max(a0, 2.0)
        a = np.full((K, d), max(float(a0), 2.0), dtype=float)
        b = a / (E_tau_target + 1e-12)

    elbo_proxy = []

    for t in range(1, iters + 1):
        rho_t = (tau0 + t) ** (-kappa)

        idx = rng.choice(N, size=min(batch_size, N), replace=False)
        Xb = X[idx]
        B = Xb.shape[0]
        scale = N / B

        # Expectations
        E_log_pi = digamma(alpha) - digamma(np.sum(alpha))  # (K,)
        E_tau = a / b                                      # (K,d)
        E_log_tau = digamma(a) - np.log(b)                 # (K,d)

        # Responsibilities
        log_r = np.zeros((B, K))
        const = -0.5 * d * np.log(2.0 * np.pi)
        for k in range(K):
            diff = Xb - m[k]
            quad = E_tau[k] * (diff ** 2) + 1.0 / beta[k]
            log_r[:, k] = (
                E_log_pi[k]
                + const
                + 0.5 * np.sum(E_log_tau[k])
                - 0.5 * np.sum(quad, axis=1)
            )

        log_r -= logsumexp(log_r, axis=1, keepdims=True)
        r = np.exp(log_r)

        # Sufficient statistics (scaled)
        Nk_hat = scale * r.sum(axis=0)            # (K,)
        xk_sum_hat = scale * (r.T @ Xb)           # (K,d)
        xk2_sum_hat = scale * (r.T @ (Xb ** 2))   # (K,d)

        # Target natural params
        alpha_tilde = alpha0 + Nk_hat

        beta_tilde = beta0 + Nk_hat[:, None]
        m_tilde = (beta0 * m0[None, :] + xk_sum_hat) / beta_tilde

        a_tilde = a0 + 0.5 * Nk_hat[:, None]
        b_tilde = b0 + 0.5 * (
            xk2_sum_hat
            + beta0 * (m0[None, :] ** 2)
            - beta_tilde * (m_tilde ** 2)
        )

        # Robbins–Monro update
        alpha = (1 - rho_t) * alpha + rho_t * alpha_tilde
        beta = (1 - rho_t) * beta + rho_t * beta_tilde
        m = (1 - rho_t) * m + rho_t * m_tilde
        a = (1 - rho_t) * a + rho_t * a_tilde
        b = (1 - rho_t) * b + rho_t * b_tilde

        elbo_proxy.append(np.mean(logsumexp(log_r, axis=1)))

    return {
        "alpha": alpha,
        "m": m,
        "beta": beta,
        "a": a,
        "b": b,
        "elbo_proxy": np.array(elbo_proxy),
    }


def svi_gmm_diag_flat(
    X,
    K,
    iters=3000,
    batch_size=256,
    seed=0,
    # "flat" / weak-prior knobs (use tiny epsilons for numerical stability)
    eps_beta0=1e-8,
    eps_a0=1e-8,
    eps_b0=1e-8,
    # Robbins–Monro schedule
    tau0=10.0,
    kappa=0.7,
    # Initialization (shared with SGLD)
    init_method="kmeans",
    init_mu0=None,
    init_sigma0=None,
    init_pi0=None,
    **kwargs,
):
    """SVI for diagonal-covariance GMM with **SGLD-matching priors**.

    Intended to match your SGLD setup:
      - Fixed mixture weights pi_k = 1/K (no q(pi), no Dirichlet prior)
      - "Flat" prior on mu (improper)  ~ approximated by beta0 -> 0
      - Flat prior on eta = log(sigma^2) (improper) corresponds to Jeffreys p(sigma^2) ∝ 1/sigma^2
        In the Normal-Gamma parameterization, we approximate this by a0,b0 -> 0.

    We keep tiny eps values to avoid division-by-zero / NaNs.

    Variational family (global): product over (k,j) Normal-Gamma(m, beta, a, b)
    Local: q(z_n) categorical responsibilities.

    Returns a dict compatible with the plotting helpers and prediction.
    """
    # Backward-compat: allow callers to pass init=... instead of init_method=...
    if ("init" in kwargs) and (init_method is None or init_method == "kmeans"):
        init_method = kwargs["init"]
    rng = np.random.default_rng(seed)
    N, d = X.shape

    # Initialization (prefer explicit init_mu0/init_sigma0/init_pi0)
    m, sigma_init, _pi_init, r_init = _init_from_mu_sigma_pi(
        X, K, seed,
        init_mu0=init_mu0,
        init_sigma0=init_sigma0,
        init_pi0=init_pi0,
        init_method=init_method,
    )

    # Fixed mixture weights: uniform
    log_pi = np.full(K, -np.log(K), dtype=float)  # log(1/K)

    # Weak/flat priors via eps
    beta0 = float(eps_beta0)
    a0 = float(eps_a0)
    b0 = float(eps_b0)

    # Global Normal-Gamma params per (k,j)
    beta = np.full((K, d), beta0, dtype=float)
    a = np.full((K, d), a0, dtype=float)
    b = np.full((K, d), b0, dtype=float)

    if sigma_init is not None:
        E_tau_target = 1.0 / (sigma_init ** 2)
        # Keep a > 1 for finite variance in Student-t marginal; use 2.0
        a = np.full((K, d), 2.0, dtype=float)
        b = a / (E_tau_target + 1e-12)

    elbo_proxy = []

    for t in range(1, iters + 1):
        rho_t = (tau0 + t) ** (-kappa)

        idx = rng.choice(N, size=min(batch_size, N), replace=False)
        Xb = X[idx]
        B = Xb.shape[0]
        scale = N / B

        # Expectations under Normal-Gamma
        # E[tau] = a/b; E[log tau] = digamma(a) - log b
        # Add small eps to denominators for stability
        E_tau = a / (b + 1e-300)
        E_log_tau = digamma(a + 1e-300) - np.log(b + 1e-300)

        # Responsibilities on minibatch: log r_nk ∝ log pi_k + E log N(x|mu,tau)
        log_r = np.zeros((B, K), dtype=float)
        const = -0.5 * d * np.log(2.0 * np.pi)
        for k in range(K):
            diff = Xb - m[k]
            quad = E_tau[k] * (diff ** 2) + 1.0 / (beta[k] + 1e-300)
            log_r[:, k] = (
                log_pi[k]
                + const
                + 0.5 * np.sum(E_log_tau[k])
                - 0.5 * np.sum(quad, axis=1)
            )

        log_r -= logsumexp(log_r, axis=1, keepdims=True)
        r = np.exp(log_r)

        # Scaled sufficient statistics
        Nk_hat = scale * r.sum(axis=0)                 # (K,)
        xk_sum_hat = scale * (r.T @ Xb)                # (K,d)
        xk2_sum_hat = scale * (r.T @ (Xb ** 2))        # (K,d)

        # Batch targets for Normal-Gamma (with beta0,a0,b0 ~ 0)
        beta_tilde = beta0 + Nk_hat[:, None]
        # With beta0 ~ 0, this is essentially the responsibility-weighted sample mean
        m_tilde = xk_sum_hat / (beta_tilde + 1e-300)

        a_tilde = a0 + 0.5 * Nk_hat[:, None]
        # With m0 absent/flat, b update reduces to 0.5*(sum r x^2 - beta_tilde*m_tilde^2)
        b_tilde = b0 + 0.5 * (xk2_sum_hat - beta_tilde * (m_tilde ** 2))
        b_tilde = np.maximum(b_tilde, 1e-300)

        # Robbins–Monro update
        beta = (1 - rho_t) * beta + rho_t * beta_tilde
        m = (1 - rho_t) * m + rho_t * m_tilde
        a = (1 - rho_t) * a + rho_t * a_tilde
        b = (1 - rho_t) * b + rho_t * b_tilde

        elbo_proxy.append(np.mean(logsumexp(log_r, axis=1)))

    return {
        # No alpha because pi is fixed
        "pi_fixed": np.ones(K) / K,
        "m": m,
        "beta": beta,
        "a": a,
        "b": b,
        "elbo_proxy": np.array(elbo_proxy),
        "_svi_kind": "flat_fixed_pi",
    }


def predict_labels_from_svi(X, svi_params):
    """
    Hard assignments using variational expectations.
    """
    alpha = svi_params.get("alpha", None)
    pi_fixed = svi_params.get("pi_fixed", None)
    m = svi_params["m"]
    beta = svi_params["beta"]
    a = svi_params["a"]
    b = svi_params["b"]

    if alpha is not None:
        E_log_pi = digamma(alpha) - digamma(np.sum(alpha))
    else:
        if pi_fixed is None:
            raise ValueError("svi_params must contain either 'alpha' or 'pi_fixed'.")
        pi_fixed = np.asarray(pi_fixed, dtype=float)
        E_log_pi = np.log(pi_fixed)

    N, d = X.shape
    K = m.shape[0]

    log_probs = np.zeros((N, K))
    const = -0.5 * d * np.log(2.0 * np.pi)
    for k in range(K):
        diff = X - m[k]
        quad = a[k] / b[k] * (diff ** 2) + 1.0 / beta[k]
        log_probs[:, k] = (
            E_log_pi[k]
            + const
            + 0.5 * np.sum(digamma(a[k]) - np.log(b[k]))
            - 0.5 * np.sum(quad, axis=1)
        )

    return np.argmax(log_probs, axis=1)


def mu_marginal_student_t_params(svi_params):
    """Return marginal Student-t parameters for each mean component mu_{k,j}.

    Under the Normal-Gamma variational factorization (shape-rate Gamma):
      tau_{k,j} ~ Gamma(a_{k,j}, b_{k,j})
      mu_{k,j} | tau_{k,j} ~ Normal(m_{k,j}, (beta_{k,j} * tau_{k,j})^{-1})

    The marginal for mu_{k,j} is Student-t with:
      df = 2 a
      loc = m
      scale^2 = b / (a * beta)

    Returns:
      df:    (K,d)
      loc:   (K,d)
      scale: (K,d)
      var:   (K,d)  (when a>1)
    """
    m = np.asarray(svi_params["m"], dtype=float)
    beta = np.asarray(svi_params["beta"], dtype=float)
    a = np.asarray(svi_params["a"], dtype=float)
    b = np.asarray(svi_params["b"], dtype=float)

    df = 2.0 * a
    loc = m
    scale = np.sqrt(b / (a * beta))

    # Marginal variance exists for a>1: Var(mu)= b / (beta * (a-1))
    var = np.full_like(scale, np.nan, dtype=float)
    mask = a > 1.0
    var[mask] = b[mask] / (beta[mask] * (a[mask] - 1.0))

    return df, loc, scale, var


def plot_posterior_means_1d(
    svi_params,
    dims=None,
    components=None,
    n_cols=3,
    grid_std=4.0,
    points=400,
    show=True,
    savepath=None,
):
    """Plot 1D marginal posteriors for mu_{k,j} as Student-t curves.

    - dims: list of dimension indices j to plot (default: all)
    - components: list of component indices k to plot (default: all)

    Produces a grid of subplots, one per (k,j).
    """
    df, loc, scale, var = mu_marginal_student_t_params(svi_params)
    K, d = loc.shape

    if dims is None:
        dims = list(range(d))
    if components is None:
        components = list(range(K))

    items = [(k, j) for k in components for j in dims]
    n = len(items)
    if n == 0:
        raise ValueError("No (component, dim) selected to plot.")

    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, (k, j) in enumerate(items):
        ax = axes[ax_idx]

        # Choose x-range using available variance; fallback to scale
        if np.isfinite(var[k, j]) and var[k, j] > 0:
            std = np.sqrt(var[k, j])
        else:
            std = scale[k, j]

        x_min = loc[k, j] - grid_std * std
        x_max = loc[k, j] + grid_std * std
        xs = np.linspace(x_min, x_max, points)

        ys = student_t.pdf(xs, df=df[k, j], loc=loc[k, j], scale=scale[k, j])
        ax.plot(xs, ys)

        ax.set_title(f"q(mu[{k},{j}])  df={df[k,j]:.1f}")
        ax.set_xlabel("mu")
        ax.set_ylabel("density")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=200)

    if show:
        plt.show()

    return fig


def sample_posterior_means(svi_params, n_samples=1000, seed=0):
    """Draw samples from the marginal q(mu) (Student-t) for each (k,j).

    Returns:
      mu_samps: (n_samples, K, d)
    """
    rng = np.random.default_rng(seed)
    df, loc, scale, _ = mu_marginal_student_t_params(svi_params)
    K, d = loc.shape

    mu_samps = np.empty((n_samples, K, d), dtype=float)
    for k in range(K):
        for j in range(d):
            mu_samps[:, k, j] = student_t.rvs(
                df=df[k, j], loc=loc[k, j], scale=scale[k, j], size=n_samples, random_state=rng
            )
    return mu_samps


def plot_posterior_means_samples(
    svi_params,
    dims=(0, 1),
    components=None,
    n_samples=2000,
    alpha=0.25,
    show=True,
    savepath=None,
):
    """2D scatter of sampled (mu_{k,d0}, mu_{k,d1}) from q(mu) for each component.

    Useful when d>=2.
    """
    d0, d1 = dims
    mu_samps = sample_posterior_means(svi_params, n_samples=n_samples, seed=0)
    K = mu_samps.shape[1]

    if components is None:
        components = list(range(K))

    plt.figure(figsize=(6, 6))
    for k in components:
        plt.scatter(mu_samps[:, k, d0], mu_samps[:, k, d1], s=6, alpha=alpha, label=f"k={k}")

    plt.xlabel(f"mu[:,{d0}]")
    plt.ylabel(f"mu[:,{d1}]")
    plt.title(f"Samples from q(mu) in dims ({d0},{d1})")
    plt.legend(markerscale=2)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200)

    if show:
        plt.show()
