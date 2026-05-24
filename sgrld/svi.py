

"""svi.py

Stochastic Variational Inference (SVI) for the semi-collapsed LDA model:

  - Topics:        pi_k (topic-word distributions) ~ Dirichlet(beta)
  - Document topic proportions theta_d are integrated out analytically.
  - Token topics z_{di} are latent.

Target (up to proportionality):
  p(w, z, pi | alpha, beta)
  = p(pi | beta) * \prod_d p(w_d, z_d | alpha, pi)

We use a mean-field variational family:
  q(pi) = \prod_k Dirichlet(lambda_k)
  q(z)  = \prod_{d,i} Categorical(r_{di})

and perform stochastic updates of lambda using minibatches of documents.

This is a practical "collapsed VB" style update:
  r_{di,k} \propto exp( E_q[log pi_{k, w_{di}}] + psi(alpha + n_{d,k}^{-i}) )
where n_{d,k}^{-i} is the expected topic count in doc d excluding token i.

Notes:
- This matches the *semi-collapsed* setup (theta integrated out). It does NOT introduce q(theta).
- For large corpora, per-document local iterations can dominate runtime; tune local_iters.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, logsumexp

from typing import List, Tuple, Dict, Optional


def generate_synthetic_lda_data(
    D: int,
    K: int,
    V: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    doc_length_range: Tuple[int, int] = (10, 20),
    seed: int = 451,
):
    """Same generator pattern you used elsewhere.

    Returns:
      corpus: list of docs, each a list of 0-based word ids
      topic_assignments: list of docs, each a list of 0-based topic ids (true z)
      topic_word_dist: (K,V) true pi
      doc_topic_dists: list of (K,) true theta_d
    """
    np.random.seed(seed)

    topic_word_dist = np.random.dirichlet([beta] * V, size=K)

    corpus = []
    topic_assignments = []
    doc_topic_dists = []

    for _ in range(D):
        eta_d = np.random.dirichlet([alpha] * K)
        doc_topic_dists.append(eta_d)

        doc_len = np.random.randint(*doc_length_range)
        doc = []
        z_d = []
        for _ in range(doc_len):
            z = np.random.choice(K, p=eta_d)
            z_d.append(z)
            word = np.random.choice(V, p=topic_word_dist[z])
            doc.append(word)

        corpus.append(doc)
        topic_assignments.append(z_d)

    return corpus, topic_assignments, topic_word_dist, doc_topic_dists


def _flatten_doc_words(doc: List[int]) -> np.ndarray:
    return np.asarray(doc, dtype=np.int64)


def _e_log_pi(lambda_kv: np.ndarray) -> np.ndarray:
    """E_q[log pi] for Dirichlet lambda.

    lambda_kv: (K,V)
    returns:   (K,V)
    """
    lam = np.asarray(lambda_kv, dtype=float)
    return digamma(lam) - digamma(lam.sum(axis=1, keepdims=True))


def _local_update_doc_r(
    w_d: np.ndarray,
    E_log_pi_kv: np.ndarray,
    alpha: float,
    local_iters: int = 20,
    tol: float = 1e-4,
) -> np.ndarray:
    """Update responsibilities r_{i,k} for a single document under theta-collapsed VB.

    Parameters
    ----------
    w_d: (L,) 0-based word ids for this doc
    E_log_pi_kv: (K,V)
    alpha: scalar symmetric Dirichlet hyperparameter

    Returns
    -------
    r: (L,K) responsibilities
    """
    L = w_d.size
    K, V = E_log_pi_kv.shape

    # initialize r uniformly
    r = np.full((L, K), 1.0 / K, dtype=float)

    # expected topic counts per doc
    n_dk = r.sum(axis=0)  # (K,)

    for _ in range(local_iters):
        r_old = r

        # Update each token i. Complexity O(L*K).
        for i in range(L):
            w = w_d[i]

            # expected counts excluding i
            n_excl = n_dk - r[i]

            # log unnormalized: E log pi_{k,w} + psi(alpha + n_excl_k)
            log_r_i = E_log_pi_kv[:, w] + digamma(alpha + n_excl)

            # normalize
            log_r_i -= logsumexp(log_r_i)
            r[i] = np.exp(log_r_i)

            # update n_dk incrementally
            n_dk = n_excl + r[i]

        # convergence check
        max_diff = np.max(np.abs(r - r_old))
        if max_diff < tol:
            break

    return r


def svi_lda_semi_collapsed(
    corpus: List[List[int]],
    K: int,
    V: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    iters: int = 2000,
    batch_docs: int = 64,
    local_iters: int = 10,
    tau0: float = 10.0,
    kappa: float = 0.7,
    seed: int = 0,
    verbose_every: int = 50,
) -> Dict[str, np.ndarray]:
    """SVI for semi-collapsed LDA (theta integrated out).

    Returns a dict with:
      lambda: (K,V) Dirichlet parameters for q(pi)
      pi_mean: (K,V) posterior mean under q(pi)
      elbo_proxy: (iters,) a simple proxy (avg token log predictive under q)

    Notes
    -----
    - This is an approximate SVI using per-document local coordinate updates for r.
    - We update only the global topic-word factors q(pi).
    """
    rng = np.random.default_rng(seed)
    D = len(corpus)

    # Global variational parameters for topics: q(pi_k) = Dir(lambda_k)
    # Initialize near the prior but with small random noise to break symmetry.
    lam = beta + rng.random((K, V)) * 0.01

    elbo_proxy = np.zeros(iters, dtype=float)

    for t in range(1, iters + 1):
        rho_t = (tau0 + t) ** (-kappa)

        # minibatch of documents
        bsize = min(batch_docs, D)
        doc_idx = rng.choice(D, size=bsize, replace=False)

        E_log_pi_kv = _e_log_pi(lam)

        # Expected topic-word counts from minibatch
        s_kw = np.zeros((K, V), dtype=float)
        token_lp = []

        for d in doc_idx:
            w_d = _flatten_doc_words(corpus[d])
            if w_d.size == 0:
                continue

            r = _local_update_doc_r(
                w_d,
                E_log_pi_kv,
                alpha=alpha,
                local_iters=local_iters,
            )

            # Accumulate expected counts for this doc
            # s_kw[k,w] += sum_i r[i,k] * 1{w_i=w}
            for i, w in enumerate(w_d):
                s_kw[:, w] += r[i]

            # ELBO proxy: average log predictive under current E_log_pi and doc r
            # log p(w_i | r_i, pi) approx log sum_k r_{ik} exp(E_log_pi_{k,w})
            # This is not a true ELBO, but is a stable monitoring statistic.
            lse = logsumexp(np.log(r + 1e-300) + E_log_pi_kv[:, w_d].T, axis=1)
            token_lp.append(np.mean(lse))

        if token_lp:
            elbo_proxy[t - 1] = float(np.mean(token_lp))
        else:
            elbo_proxy[t - 1] = np.nan

        # Scale minibatch counts to full corpus
        scale = D / bsize
        s_kw *= scale

        # Natural-parameter target for lambda: beta + expected counts
        lam_tilde = beta + s_kw

        # Robbins–Monro update
        lam = (1.0 - rho_t) * lam + rho_t * lam_tilde

        if verbose_every and (t % verbose_every == 0):
            print(f"[svi] iter {t}/{iters}  rho={rho_t:.4g}  proxy={elbo_proxy[t-1]:.4f}")

    pi_mean = lam / lam.sum(axis=1, keepdims=True)

    return {
        "lambda": lam,
        "pi_mean": pi_mean,
        "elbo_proxy": elbo_proxy,
        "_kind": "semi_collapsed_svi",
        "alpha": float(alpha),
        "beta": float(beta),
    }


def print_top_entries_pi(pi: np.ndarray, top: int = 20, title: str = ""):
    """Print top entries in a (K,V) topic-word matrix."""
    K, V = pi.shape
    flat = pi.reshape(-1)
    idxs = np.argsort(flat)[-top:][::-1]
    if title:
        print(title)
    for idx in idxs:
        k = idx // V
        w = idx % V
        print(f"  pi(topic={k}, word={w}) = {float(pi[k, w]):.6f}")


def main():
    # Small demo (keep sizes modest)
    D = 10000
    K = 3
    V = 50
    alpha = 0.1
    beta = 0.1

    corpus, z_true_docs, true_pi, true_theta_docs = generate_synthetic_lda_data(
        D, K, V, alpha=alpha, beta=beta, doc_length_range=(10, 20), seed=451
    )

    print(f"Synthetic corpus: D={D}, K={K}, V={V}, alpha={alpha}, beta={beta}")

    svi = svi_lda_semi_collapsed(
        corpus,
        K=K,
        V=V,
        alpha=alpha,
        beta=beta,
        iters=1000,
        batch_docs=64,
        local_iters=10,
        tau0=10.0,
        kappa=0.7,
        seed=0,
        verbose_every=100,
    )

    pi_mean = svi["pi_mean"]

    print("\n================ TRUE topic-word distribution (pi) ================")
    print(true_pi)

    print("\n================ SVI posterior mean pi ================")
    print(pi_mean)

    print_top_entries_pi(pi_mean, top=20, title="\nTop 20 pi entries (SVI posterior mean):")


if __name__ == "__main__":
    main()