import autograd.numpy as np
from autograd import grad
from scipy.stats import multivariate_normal


def _softmax(v):
    v = np.asarray(v, dtype=float)
    v = v - np.max(v)  # numerical stability
    ev = np.exp(v)
    return ev / np.sum(ev)


def generate_data(K, d, N, means, covs, weights):
    z = np.random.choice(K, size=N, p=weights)
    X = np.array([np.random.multivariate_normal(means[zi], covs[zi]) for zi in z])
    return X, z


def gibbs(X, mu, sigma, pi):
    N, K = X.shape[0], mu.shape[0]
    d = X.shape[1]
    responsibilities = np.zeros((N, K))
    for k in range(K):
        # Diagonal covariance: sigma has shape (K, d), so use per-dimension variances
        var_k = (sigma[k] ** 2).astype(float)  # shape (d,)
        cov_k = np.diag(var_k)
        logpdf = multivariate_normal.logpdf(X, mean=mu[k], cov=cov_k,allow_singular=True)
        responsibilities[:, k] = pi[k] * np.exp(logpdf)
    row_sums = responsibilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    responsibilities /= row_sums
    z = np.empty(N, dtype=int)
    for i in range(N):
        row = responsibilities[i]
        total = row.sum()
        if total == 0 or not np.isfinite(total):
            p = np.ones(K) / K
        else:
            p = row / total
            p /= p.sum()
        z[i] = np.random.choice(K, p=p)
    return z


def sgld(X, z, mu, sigma, stepsize, batch_size, inverse_temperature,
         precondition_logsigma=True,
         precond_mu_scales=None,
         precond_eta_scales=None,
         use_precond=True):
    K, d = mu.shape
    log_sigma2 = np.log(sigma ** 2)
    # Fisher preconditioning for η = log(σ²): per-observation Fisher I_η = d/2 ⇒ J^{-1}_η = 2/d

    def log_posterior_log_sigma2(log_s2, Xk, mu_k):
        """Log-likelihood for diagonal covariance with log_s2 a d-vector.

        Xk: (n_k, d), mu_k: (d,), log_s2: (d,)
        """
        sigma2 = np.exp(log_s2)              # (d,)
        diff = Xk - mu_k                     # (n_k, d)
        # Per-dimension contributions: -0.5[ log(2πσ²) + (x-μ)²/σ² ]
        log_lik_per_dim = -0.5 * (np.log(2 * np.pi * sigma2) + (diff ** 2) / sigma2)
        log_lik = np.sum(log_lik_per_dim, axis=1)  # (n_k,)
        return np.mean(log_lik)

    def log_posterior_mu(mu_k, Xk, sigma2_k):
        """Log-likelihood for diagonal covariance w.r.t. mu.

        Xk: (n_k, d), mu_k: (d,), sigma2_k: (d,)
        """
        diff = Xk - mu_k                     # (n_k, d)
        log_lik_per_dim = -0.5 * (np.log(2 * np.pi * sigma2_k) + (diff ** 2) / sigma2_k)
        log_lik = np.sum(log_lik_per_dim, axis=1)  # (n_k,)
        return np.mean(log_lik)

    for k in range(K):
        Xk = X[z == k]
        if Xk.shape[0] == 0:
            continue

        # Diagonal sigma: sigma[k] has shape (d,)
        sigma2_k = (sigma[k] ** 2).astype(float)           # (d,)

        # Preconditioner for mu: allow scalar or per-dim; broadcast to (d,)
        if use_precond and (precond_mu_scales is not None):
            precond_mu = np.asarray(precond_mu_scales[k], dtype=float)
        else:
            precond_mu = sigma2_k * K
        precond_mu = np.broadcast_to(precond_mu, (d,))

        grad_mu_fn = grad(log_posterior_mu)
        grad_mu = grad_mu_fn(mu[k], Xk, sigma2_k)          # (d,)
        noise_mu = np.random.normal(0, 1, d)               # (d,)

        mu[k] += (stepsize * precond_mu * grad_mu) / 2.0 \
                 + np.sqrt(stepsize * precond_mu / inverse_temperature) * noise_mu

        grad_log_sigma2_fn = grad(log_posterior_log_sigma2)
        g = grad_log_sigma2_fn(log_sigma2[k], Xk, mu[k])   # (d,)
        noise = np.random.normal(0, 1, d)                  # (d,)

        # Optional Fisher preconditioning for η = log(σ²)
        if precondition_logsigma:
            if use_precond and (precond_eta_scales is not None):
                precond_eta = np.asarray(precond_eta_scales[k], dtype=float)
            else:
                precond_eta = 2.0 *K
        else:
            precond_eta = 1.0
        precond_eta = np.broadcast_to(precond_eta, (d,))

        # SGLD step with matching noise scaling (diagonal per-dim)
        log_sigma2[k] += (stepsize * precond_eta * g) / 2.0 \
                         + np.sqrt(stepsize * precond_eta / inverse_temperature) * noise
        sigma[k] = np.sqrt(np.exp(log_sigma2[k]))

    return mu, sigma


def run(K, d, X, stepsize, batch_size, iters, true_means, true_covs, true_weights,
        inverse_temperature,
        precond_mu_scales=None,
        precond_eta_scales=None,
        use_precond=True,
        track_indices=None):
    mu = np.random.randn(K, d)
    # Initialize diagonal std dev per cluster and per dimension
    sigma = np.random.uniform(0.5, 2.0, size=(K, d))
    # Fixed mixture weights: uniform prior
    pi = np.ones(K) / K
    samples_means = []
    samples_sigma = []
    z0_history = []
    z0 = np.random.choice(K)

    # Track sampled z for selected global datapoint indices (for predicted-vs-sampled plots).
    # If indices are provided by the caller (e.g., overlap/ambiguous points), we keep them.
    # Otherwise we default to tracking a few random points.
    if track_indices is None:
        rng = np.random.default_rng(123)
        track_indices = [int(i) for i in rng.choice(X.shape[0], size=min(5, X.shape[0]), replace=False)]
        print("[z-track] tracking global indices (random):", track_indices)
    else:
        track_indices = [int(i) for i in track_indices]
        print("[z-track] tracking global indices (provided):", track_indices)

    z_track_history = {int(i): [] for i in track_indices}
    z_last = {int(i): None for i in track_indices}

    N = X.shape[0]
    for _ in range(iters):
        batch_idx = np.random.choice(N, size=min(batch_size, N), replace=False)
        X_batch = X[batch_idx]
        z_batch = gibbs(X_batch, mu, sigma, pi)
        if 0 in batch_idx:
            local_idx = np.where(batch_idx == 0)[0][0]
            z0 = z_batch[local_idx]
        z0_history.append(z0)

        # Record sampled z for tracked global indices whenever they appear in this minibatch.
        if track_indices:
            pos = {int(g): int(p) for p, g in enumerate(batch_idx)}
            for g in track_indices:
                if g in pos:
                    z_last[g] = int(z_batch[pos[g]])
                # store last-seen sampled value; -1 until the point is first observed
                z_track_history[g].append(int(z_last[g]) if (z_last[g] is not None) else -1)

        mu, sigma = sgld(
            X_batch, z_batch, mu, sigma,
            stepsize, batch_size, inverse_temperature,
            precondition_logsigma=True,
            precond_mu_scales=precond_mu_scales,
            precond_eta_scales=precond_eta_scales,
            use_precond=use_precond,
        )

        samples_means.append(mu.copy())
        samples_sigma.append(sigma.copy())

    samples_means = np.array(samples_means)
    samples_sigma = np.array(samples_sigma)
    return mu, samples_means, sigma, samples_sigma, z0_history, z_track_history


# ===================== New: S>1 averaged-Gibbs-gradient SGLD (kept separate) =====================

def sgld_avg_gibbs(
    X,
    mu,
    sigma,
    stepsize,
    batch_size,
    inverse_temperature,
    *,
    gibbs_S=10,
    pi=None,
    precondition_logsigma=True,
    precond_mu_scales=None,
    precond_eta_scales=None,
    use_precond=True,
    sigma_floor=1e-8,
    log_sigma2_clip=30.0,
    clip_diff=1e6,
):
    """One SGLD update using an *averaged* stochastic gradient over S Gibbs draws.

    This matches the paper definition:

        G_k^{(S)}(theta) = (1/(bS)) \sum_{i\in I_k} \sum_{s=1}^S \nabla_theta log p(x_i, z_i^{(s)} | theta)

    with iid conditional draws z_i^{(s)} ~ p(z_i | x_i, theta) on the *minibatch*.

    Implementation details
    ----------------------
    - We compute conditional probabilities r_{ik} = p(z_i=k | x_i, mu, sigma, pi)
      using the diagonal-Gaussian log-likelihood and log-sum-exp normalization.
    - For each s=1..S we sample a full minibatch assignment vector z^(s) using r.
    - We accumulate complete-data gradients for (mu, eta=log sigma^2) and average
      across s and i (divide by S and B) to match 1/(bS) in the definition.
    - This is intentionally *not* Rao-Blackwellized; it is the explicit Monte Carlo
      averaging variant that is common in practice.

    Returns
    -------
    mu, sigma, z_draws
        Updated parameters and the list of sampled z vectors (length S), each (B,).
    """
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    B, d = X.shape
    K = mu.shape[0]

    if pi is None:
        pi = np.ones(K, dtype=float) / K
    else:
        pi = np.asarray(pi, dtype=float)
        pi = np.maximum(pi, 1e-300)
        pi = pi / np.sum(pi)

    S = int(gibbs_S)
    if S <= 0:
        raise ValueError("gibbs_S must be a positive integer")

    # Work with eta = log(sigma^2) (elementwise, diagonal)
    sigma2 = np.maximum(sigma ** 2, float(sigma_floor))  # (K,d)
    log_sigma2 = np.log(sigma2)
    log_sigma2 = np.clip(log_sigma2, -float(log_sigma2_clip), float(log_sigma2_clip))

    # ---------------- Compute responsibilities r_{ik} on minibatch ----------------
    # log r_{i,k} ∝ log pi_k - 1/2 * sum_j [log(2π σ^2_{k,j}) + (x_{i,j}-mu_{k,j})^2 / σ^2_{k,j}]
    log_pi = np.log(np.maximum(pi, 1e-300))  # (K,)
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * sigma2), axis=1)  # (K,)

    log_r = np.zeros((B, K), dtype=float)
    for k in range(K):
        diff = X - mu[k][None, :]
        diff = np.clip(diff, -float(clip_diff), float(clip_diff))
        quad = -0.5 * np.sum((diff ** 2) / sigma2[k][None, :], axis=1)  # (B,)
        log_r[:, k] = log_pi[k] + log_norm[k] + quad

    # stable softmax row-wise
    log_r = log_r - np.max(log_r, axis=1, keepdims=True)
    r = np.exp(log_r)
    r = np.where(np.isfinite(r), r, 0.0)
    row_sums = r.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    r = r / row_sums

    # ---------------- Monte Carlo: sample S iid z-draws from r ----------------
    # z_draws[s][i] ~ Cat(r[i,:])
    z_draws = []
    for _ in range(S):
        zs = np.empty(B, dtype=int)
        for i in range(B):
            p = r[i]
            # extra guard
            ps = np.where(np.isfinite(p), p, 0.0)
            sps = ps.sum()
            if sps <= 0:
                ps = np.ones(K, dtype=float) / K
            else:
                ps = ps / sps
            zs[i] = int(np.random.choice(K, p=ps))
        z_draws.append(zs)

    # ---------------- Average complete-data gradients over draws ----------------
    # Per-draw per-cluster MEAN gradient, then average over S draws:
    #
    #   \tilde g_k^{(S)} = (1/S) * sum_{s=1}^S [ (1/n_k^{(s)}) * sum_{i:z_i^{(s)}=k} ∇ log p(x_i | z_i=k, theta) ].
    #
    # This matches the `sgld(...)` scaling (cluster-wise mean log-likelihood) and
    # avoids downweighting small clusters by an extra factor n_k/B.

    grad_mu_avg = np.zeros_like(mu)          # (K,d)
    grad_eta_avg = np.zeros_like(log_sigma2) # (K,d)

    for zs in z_draws:
        for k in range(K):
            mask = (zs == k)
            nk = int(np.sum(mask))
            if nk <= 0:
                continue

            Xk = X[mask]
            diff = Xk - mu[k][None, :]
            diff = np.clip(diff, -float(clip_diff), float(clip_diff))
            sigma2_k = np.maximum(sigma2[k].astype(float), float(sigma_floor))  # (d,)

            # Per-draw per-cluster MEAN complete-data gradients (1/nk factor)
            grad_mu_avg[k] += np.mean(diff / sigma2_k[None, :], axis=0)
            grad_eta_avg[k] += np.mean(
                -0.5 + 0.5 * (diff ** 2) / sigma2_k[None, :],
                axis=0,
            )

    # Average over Gibbs draws only (NOT divided by B). With gibbs_S=1 this aligns with `sgld(...)`.
    grad_mu_avg /= float(S)
    grad_eta_avg /= float(S)

    # ---------------- Apply one SGLD step using the averaged gradients ----------------
    for k in range(K):
        sigma2_k = np.maximum(sigma2[k].astype(float), float(sigma_floor))

        # Preconditioner for mu
        if use_precond and (precond_mu_scales is not None):
            precond_mu = np.asarray(precond_mu_scales[k], dtype=float)
        else:
            precond_mu = sigma2_k * K
        precond_mu = np.broadcast_to(precond_mu, (d,))

        noise_mu = np.random.normal(0, 1, d)
        mu[k] += (stepsize * precond_mu * grad_mu_avg[k]) / 2.0 \
                 + np.sqrt(stepsize * precond_mu / inverse_temperature) * noise_mu

        # Preconditioner for eta = log(sigma^2)
        if precondition_logsigma:
            if use_precond and (precond_eta_scales is not None):
                precond_eta = np.asarray(precond_eta_scales[k], dtype=float)
            else:
                precond_eta = 2.0 * K
        else:
            precond_eta = 1.0
        precond_eta = np.broadcast_to(precond_eta, (d,))

        noise_eta = np.random.normal(0, 1, d)
        log_sigma2[k] += (stepsize * precond_eta * grad_eta_avg[k]) / 2.0 \
                         + np.sqrt(stepsize * precond_eta / inverse_temperature) * noise_eta

        # Guardrails
        log_sigma2[k] = np.clip(log_sigma2[k], -float(log_sigma2_clip), float(log_sigma2_clip))
        sigma[k] = np.sqrt(np.exp(log_sigma2[k]))
        sigma[k] = np.maximum(sigma[k], float(np.sqrt(sigma_floor)))

    return mu, sigma, z_draws


def run_avg_gibbs(
    K,
    d,
    X,
    stepsize,
    batch_size,
    iters,
    true_means,
    true_covs,
    true_weights,
    inverse_temperature,
    *,
    gibbs_S=10,
    precond_mu_scales=None,
    precond_eta_scales=None,
    use_precond=True,
):
    """Run SGLD+Gibbs with S>1 averaged-gradient updates (NO z-tracking).

    Intentionally separate from `run(...)`.

    Returns (same arity as `run`):
        mu, samples_means, sigma, samples_sigma, z0_history, z_track_history

    For S>1 we do NOT track z, so:
        z0_history = []
        z_track_history = {}
    """
    mu = np.random.randn(K, d)
    sigma = np.random.uniform(0.5, 2.0, size=(K, d))
    pi = np.ones(K) / K

    samples_means = []
    samples_sigma = []

    # No z tracking for S>1
    z0_history = []
    z_track_history = {}

    N = X.shape[0]
    for _ in range(int(iters)):
        batch_idx = np.random.choice(N, size=min(int(batch_size), N), replace=False)
        X_batch = X[batch_idx]

        # S-averaged Gibbs (Monte Carlo) update (per-draw per-cluster mean gradients)
        mu, sigma, _z_draws = sgld_avg_gibbs(
            X_batch,
            mu,
            sigma,
            stepsize,
            batch_size,
            inverse_temperature,
            gibbs_S=int(gibbs_S),
            pi=pi,
            precondition_logsigma=True,
            precond_mu_scales=precond_mu_scales,
            precond_eta_scales=precond_eta_scales,
            use_precond=use_precond,
        )

        samples_means.append(mu.copy())
        samples_sigma.append(sigma.copy())

    samples_means = np.array(samples_means)
    samples_sigma = np.array(samples_sigma)

    return mu, samples_means, sigma, samples_sigma, z0_history, z_track_history