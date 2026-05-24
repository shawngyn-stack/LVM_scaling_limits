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
        # sigma[k] is now a length-d vector; build diagonal covariance
        cov_k = np.diag((sigma[k] ** 2).astype(float))
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
    # sigma is now (K, d)
    log_sigma2 = np.log((sigma ** 2).astype(float))  # shape (K, d)

    def log_posterior_log_sigma2(log_s2, Xk, mu_k):
        sigma2 = np.exp(log_s2)  # shape (d,)
        # per-dim contributions, sum across dims then across data, return mean per datum
        dif = Xk - mu_k  # (n_k, d)
        term_const = -0.5 * np.sum(np.log(2 * np.pi * sigma2))
        term_quad = -0.5 * np.sum((dif ** 2) / sigma2, axis=1)
        return (term_const + term_quad).mean()

    def log_posterior_mu(mu_k, Xk, sigma2_k):
        sigma2 = sigma2_k  # shape (d,)
        dif = Xk - mu_k  # (n_k, d)
        term_const = -0.5 * np.sum(np.log(2 * np.pi * sigma2))
        term_quad = -0.5 * np.sum((dif ** 2) / sigma2, axis=1)
        return (term_const + term_quad).mean()

    for k in range(K):
        Xk = X[z == k]
        if Xk.shape[0] == 0:
            continue

        sigma2_k = (sigma[k] ** 2).astype(float)           # shape (d,)

        # Preconditioner for mu: allow scalar or per-dim; broadcast to (d,)
        if use_precond and (precond_mu_scales is not None):
            precond_mu_k = precond_mu_scales[k]
        else:
            precond_mu_k = sigma2_k*K  # Fisher inverse per-dim by default
        precond_mu_k = np.broadcast_to(np.asarray(precond_mu_k, dtype=float), (d,))

        # Gradient wrt mu (vector) using diagonal sigma2
        grad_mu_fn = grad(log_posterior_mu)
        grad_mu = grad_mu_fn(mu[k], Xk, sigma2_k)          # shape (d,)
        noise_mu = np.random.normal(0.0, 1.0, size=d)
        mu[k] += (stepsize * precond_mu_k * grad_mu) / 2.0 \
                 + np.sqrt((stepsize * precond_mu_k) / inverse_temperature) * noise_mu

        # Gradient wrt log sigma^2 (vector per-dim)
        grad_log_sigma2_fn = grad(log_posterior_log_sigma2)
        g = grad_log_sigma2_fn(log_sigma2[k], Xk, mu[k])   # shape (d,)

        # Preconditioner for eta = log sigma^2: allow scalar or per-dim; broadcast to (d,)
        if precondition_logsigma:
            if use_precond and (precond_eta_scales is not None):
                precond_eta_k = precond_eta_scales[k]
            else:
                # Diagonal case: per-dim Fisher inverse is 2 (per observation), not 2/d
                precond_eta_k = 2.0*K
        else:
            precond_eta_k = 1.0
        precond_eta_k = np.broadcast_to(np.asarray(precond_eta_k, dtype=float), (d,))

        # SGLD step with matching noise scaling (per-dim)
        noise_eta = np.random.normal(0.0, 1.0, size=d)
        log_sigma2[k] += (stepsize * precond_eta_k * g) / 2.0 \
                         + np.sqrt((stepsize * precond_eta_k) / inverse_temperature) * noise_eta
        sigma[k] = np.sqrt(np.exp(log_sigma2[k]))

    return mu, sigma


def run(K, d, X, stepsize, batch_size, iters, true_means, true_covs, true_weights,
        inverse_temperature,
        precond_mu_scales=None,
        precond_eta_scales=None,
        use_precond=True,
        init_mu0=None,
        init_sigma0=None,
        init_pi0=None):
    # --- Initialization (optionally overridden by caller) ---
    if init_mu0 is None:
        mu = np.random.randn(K, d)
    else:
        mu = np.asarray(init_mu0, dtype=float)
        if mu.shape != (K, d):
            raise ValueError(f"init_mu0 must have shape {(K, d)}, got {mu.shape}")
        mu = mu.copy()

    if init_sigma0 is None:
        sigma = np.random.uniform(0.5, 2.0, size=(K, d))  # diagonal std devs per component
    else:
        sigma = np.asarray(init_sigma0, dtype=float)
        if sigma.shape != (K, d):
            raise ValueError(f"init_sigma0 must have shape {(K, d)}, got {sigma.shape}")
        # ensure strictly positive std devs
        sigma = np.maximum(sigma, 1e-6).copy()

    # Mixture weights (fixed for this sampler)
    if init_pi0 is None:
        pi = np.ones(K) / K
    else:
        pi = np.asarray(init_pi0, dtype=float)
        if pi.shape != (K,):
            raise ValueError(f"init_pi0 must have shape {(K,)}, got {pi.shape}")
        pi = np.maximum(pi, 1e-12)
        pi = (pi / np.sum(pi)).copy()

    samples_means = []
    samples_sigma = []
    z0_history = []
    z0 = np.random.choice(K)

    N = X.shape[0]
    for _ in range(iters):
        batch_idx = np.random.choice(N, size=min(batch_size, N), replace=False)
        X_batch = X[batch_idx]
        z_batch = gibbs(X_batch, mu, sigma, pi)
        if 0 in batch_idx:
            local_idx = np.where(batch_idx == 0)[0][0]
            z0 = z_batch[local_idx]
        z0_history.append(z0)
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
    return mu, samples_means, sigma, samples_sigma, z0_history
