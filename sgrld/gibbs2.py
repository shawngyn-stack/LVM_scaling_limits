import numpy as np

from typing import List, Tuple, Dict, Optional

# -------------------------
# Shared model parameters (MATCH SGRLD)
# -------------------------
SGRLD_K = 20            # number of topics
SGRLD_ALPHA = 0.1       # document-topic Dirichlet prior
SGRLD_BETA = 0.1        # topic-word Dirichlet prior

# -------------------------
# Data: 20 Newsgroups
# -------------------------

def load_20newsgroups_corpus(
    *,
    max_features: int = 10000,
    min_df: int = 10,
    max_df: float = 0.5,
    stop_words: str = "english",
    max_tokens_per_doc: int = 400,
    seed: int = 0,
) -> Tuple[List[List[int]], List[str]]:
    """Load 20 Newsgroups and build an LDA corpus as list-of-token word ids.

    Notes
    -----
    - This creates an explicit token list for each doc (repeats word ids according to counts).
      That can be large for full 20news. Use `max_tokens_per_doc` to truncate tokens per doc.

    Returns
    -------
    corpus : List[List[int]]
        Each document is a list of 0-based word ids.
    vocab : List[str]
        Vocabulary list where vocab[w] gives the token string.
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception as e:
        raise ImportError(
            "This script requires scikit-learn. Install with: pip install scikit-learn"
        ) from e

    rng = np.random.default_rng(seed)

    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = list(data.data)

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_features=int(max_features),
        min_df=int(min_df),
        max_df=float(max_df),
    )
    X = vectorizer.fit_transform(texts)  # (D, V) sparse counts
    vocab = vectorizer.get_feature_names_out().tolist()

    # Convert each sparse row into a repeated token list.
    corpus: List[List[int]] = []
    X = X.tocsr()
    for d in range(X.shape[0]):
        row = X.getrow(d)
        idx = row.indices.astype(np.int64)
        cnt = row.data.astype(np.int64)
        if idx.size == 0:
            corpus.append([])
            continue
        tokens = np.repeat(idx, cnt)
        if tokens.size > 1:
            rng.shuffle(tokens)
        if (max_tokens_per_doc is not None) and (tokens.size > int(max_tokens_per_doc)):
            tokens = tokens[: int(max_tokens_per_doc)]
        corpus.append(tokens.tolist())

    return corpus, vocab


# -------------------------
# Utilities
# -------------------------

def corpus_to_token_arrays(corpus: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Flatten corpus into token arrays.

    Returns
    -------
    doc_ids : (N,) int64 in {0,...,D-1}
    word_ids: (N,) int64 in {0,...,V-1}
    N       : total token count
    """
    doc_ids = []
    word_ids = []
    for d, doc in enumerate(corpus):
        for w in doc:
            doc_ids.append(d)
            word_ids.append(int(w))
    doc_ids = np.asarray(doc_ids, dtype=np.int64)
    word_ids = np.asarray(word_ids, dtype=np.int64)
    return doc_ids, word_ids, int(doc_ids.size)


# -------------------------
# Semi-collapsed Gibbs: p(w,z,pi | alpha,beta) with eta/theta integrated out
# -------------------------

def lda_semi_collapsed_gibbs(
    corpus: List[List[int]],
    K: int,
    V: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    iters: int = 500,
    burnin: int = 250,
    thin: int = 5,
    seed: int = 0,
    verbose_every: int = 50,
):
    """Semi-collapsed Gibbs for LDA.

    Target: p(w, z, pi | alpha, beta)

    - Keeps pi (topic-word distributions) explicit:
        pi_k ~ Dirichlet(beta + n_{k,1:V})
    - Samples each token's topic assignment with theta integrated out:
        p(z_n=k | z_-n, w, pi) ∝ (alpha + n_{d_n,k}^{-n}) * pi_{k,w_n}

    Returns
    -------
    samples_pi: (S, K, V) posterior samples of pi after burnin/thin.
                Can be large; downstream code may want to thin further.
    samples_z:  (S, N) posterior samples of z after burnin/thin
    final: dict with final state and counts
    """
    rng = np.random.default_rng(seed)

    D = len(corpus)
    doc_ids, word_ids, N = corpus_to_token_arrays(corpus)

    # init z uniformly
    z = rng.integers(0, K, size=N)

    # counts
    n_dk = np.zeros((D, K), dtype=np.int64)
    n_kw = np.zeros((K, V), dtype=np.int64)
    n_k = np.zeros(K, dtype=np.int64)

    for n in range(N):
        d = doc_ids[n]
        w = word_ids[n]
        k = z[n]
        n_dk[d, k] += 1
        n_kw[k, w] += 1
        n_k[k] += 1

    # init pi
    pi = np.empty((K, V), dtype=float)
    for k in range(K):
        pi[k] = rng.dirichlet(beta + n_kw[k])

    samples_pi = []
    samples_z = []

    for it in range(1, iters + 1):
        # (A) sample pi | z,w
        for k in range(K):
            pi[k] = rng.dirichlet(beta + n_kw[k])

        # (B) sample z | pi, z_-n, w (theta integrated out)
        for n in range(N):
            d = doc_ids[n]
            w = word_ids[n]
            k_old = z[n]

            # remove
            n_dk[d, k_old] -= 1
            n_kw[k_old, w] -= 1
            n_k[k_old] -= 1

            # probs ∝ (alpha + n_dk[d,k]) * pi[k,w]
            probs = (alpha + n_dk[d]) * pi[:, w]
            s = probs.sum()
            if (not np.isfinite(s)) or s <= 0:
                probs = np.ones(K) / K
            else:
                probs = probs / s

            k_new = rng.choice(K, p=probs)

            # add
            z[n] = k_new
            n_dk[d, k_new] += 1
            n_kw[k_new, w] += 1
            n_k[k_new] += 1

        if it > burnin and ((it - burnin) % thin == 0):
            samples_pi.append(pi.copy())
            samples_z.append(z.copy())

        if verbose_every and (it % verbose_every == 0):
            print(f"[gibbs] iter {it}/{iters}")

    samples_pi = np.asarray(samples_pi)
    samples_z = np.asarray(samples_z)

    final = {
        "pi": pi,
        "z": z,
        "doc_ids": doc_ids,
        "word_ids": word_ids,
        "n_dk": n_dk,
        "n_kw": n_kw,
        "n_k": n_k,
    }
    return samples_pi, samples_z, final


def print_top_words(pi: np.ndarray, vocab: List[str], top: int = 15, title: str = ""):
    """Print top words per topic for a (K,V) topic-word matrix pi."""
    K, V = pi.shape
    if title:
        print(title)
    for k in range(K):
        idx = np.argsort(pi[k])[-top:][::-1]
        words = [vocab[i] for i in idx]
        probs = [float(pi[k, i]) for i in idx]
        pairs = ", ".join([f"{w}:{p:.4f}" for w, p in zip(words, probs)])
        print(f"topic {k}: {pairs}")


def main():
    # ------------
    # Settings (tune these first)
    # ------------
    # Match SGRLD preprocessing
    max_features = 10000
    min_df = 10
    max_df = 0.5
    max_tokens_per_doc = 400

    # Match SGRLD model parameters
    K = SGRLD_K
    alpha = SGRLD_ALPHA
    beta = SGRLD_BETA

    iters = 500
    burnin = 250
    thin = 5
    seed = 0

    # ------------
    # Load data
    # ------------
    corpus, vocab = load_20newsgroups_corpus(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        max_tokens_per_doc=max_tokens_per_doc,
        seed=seed,
    )
    D = len(corpus)
    V = len(vocab)
    N = sum(len(doc) for doc in corpus)
    print(f"20news corpus: subset=all, D={D}, V={V}, N={N}, max_tokens_per_doc={max_tokens_per_doc}, alpha={alpha}, beta={beta}")
    print(f"[CHECK] Using parameters identical to SGRLD: K={K}, alpha={alpha}, beta={beta}")

    # ------------
    # Run Gibbs
    # ------------
    samples_pi, samples_z, final = lda_semi_collapsed_gibbs(
        corpus,
        K=K,
        V=V,
        alpha=alpha,
        beta=beta,
        iters=iters,
        burnin=burnin,
        thin=thin,
        seed=seed,
        verbose_every=50,
    )

    pi_mean = samples_pi.mean(axis=0)  # (K,V)

    # -----------------
    # Save artifacts for later comparisons (Gibbs vs SGRLD vs SVI)
    # -----------------
    # NOTE: samples_pi can be large. We optionally sub-sample before saving.
    SAVE_EVERY = 1  # set >1 to downsample the stored Gibbs pi samples
    pi_samples_save = samples_pi[:: int(SAVE_EVERY)].astype(np.float32, copy=False)
    pi_mean_save = pi_mean.astype(np.float32, copy=False)

    out_path = "gibbs_results_20ng.npz"
    np.savez_compressed(
        out_path,
        pi_samples=pi_samples_save,
        pi_mean=pi_mean_save,
        vocab=np.asarray(vocab, dtype=object),
        K=int(K),
        V=int(V),
        alpha=float(alpha),
        beta=float(beta),
        # preprocessing
        max_features=int(max_features),
        min_df=int(min_df),
        max_df=float(max_df),
        max_tokens_per_doc=int(max_tokens_per_doc),
        # sampler
        iters=int(iters),
        burnin=int(burnin),
        thin=int(thin),
        seed=int(seed),
        save_every=int(SAVE_EVERY),
    )
    print(f"\nSaved Gibbs results to: {out_path}")
    print(f"  pi_samples shape saved: {pi_samples_save.shape} (float32), pi_mean shape: {pi_mean_save.shape}")

    print("\n================ Gibbs posterior mean: top words per topic ================")
    print_top_words(pi_mean, vocab, top=15)

    # Results already saved above as a single compressed NPZ.


if __name__ == "__main__":
    main()
