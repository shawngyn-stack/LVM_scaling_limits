import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.special import psi as digamma, logsumexp  # digamma + logsumexp for SVI
from scipy.stats import beta as beta_dist, dirichlet as dirichlet_dist

# Canonical prefix for all result headers, figure titles, captions
CANONICAL_PREFIX = "[Canonical topic order]"

from typing import List, Dict

# ===================== Real dataset loader: 20 Newsgroups =====================

def load_20newsgroups_corpus(
    *,
    max_features: int = 10000,
    min_df: int = 10,
    max_df: float = 0.5,
    stop_words: str = "english",
    max_tokens_per_doc: int = 400,
    seed: int = 0,
):
    """Load 20 Newsgroups and convert to a token-id corpus: List[List[int]].

    Returns
    -------
    corpus: List[List[int]]
    V: int
    texts: List[str]
    vectorizer: fitted CountVectorizer (for vocab inspection if needed)

    Notes
    -----
    - Uses sklearn's fetch_20newsgroups + CountVectorizer.
    - Converts bag-of-words counts to a token sequence by repeating word ids.
    - Optionally truncates each document to `max_tokens_per_doc` tokens for speed.
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for 20 Newsgroups. Install via `pip install scikit-learn`."
        ) from e

    rng = np.random.default_rng(seed)

    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = list(data.data)

    vec = CountVectorizer(
        stop_words=stop_words,
        max_features=int(max_features),
        min_df=min_df,
        max_df=max_df,
    )
    X = vec.fit_transform(texts)  # sparse (D, V)
    D, V = X.shape

    # Convert sparse rows to token lists.
    corpus = []
    for d in range(D):
        row = X.getrow(d)
        idx = row.indices
        cnt = row.data
        # build token list with repetition
        tokens = []
        for j, c in zip(idx, cnt):
            # c is count for word j
            tokens.extend([int(j)] * int(c))
        # shuffle within doc (optional but helps avoid systematic ordering)
        if len(tokens) > 1:
            rng.shuffle(tokens)
        # truncate for speed if requested
        if max_tokens_per_doc is not None and len(tokens) > int(max_tokens_per_doc):
            tokens = tokens[: int(max_tokens_per_doc)]
        corpus.append(tokens)

    return corpus, int(V), texts, vec

def generate_synthetic_lda_data(D, K, V, alpha=0.1, beta=0.1, doc_length_range=(10, 20), seed=451):
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


class SGRLD_LDA:
    def __init__(self, D, V, K, alpha=0.1, beta=0.1, eta=None, epsilon=1e-2, seed=22):
        np.random.seed(seed)
        self.D = D
        self.V = V
        self.K = K
        self.alpha = alpha
        # Paper notation: beta is the symmetric Dirichlet hyperparameter for topics pi_k.
        # Backward-compat: if callers pass eta=..., treat it as beta.
        if eta is not None:
            beta = eta
        self.beta = beta
        self.epsilon = epsilon

        self.theta = np.empty((K, V))  # to be initialized externally
        self.topic_word_dist = self._normalize_theta(self.theta)

    def _normalize_theta(self, theta):
        theta_pos = np.abs(theta)
        row_sums = theta_pos.sum(axis=1, keepdims=True)
        eps = 1e-12
        zero_rows = (row_sums <= eps).flatten()
        if np.any(zero_rows):
            theta_pos[zero_rows, :] = 1.0
            row_sums = theta_pos.sum(axis=1, keepdims=True)
        return theta_pos / (row_sums + eps)

    def gibbs_sample_z(self, doc, topic_word_dist, alpha, num_sweeps=2, z_init=None):
        """Collapsed Gibbs for z within ONE document with theta integrated out.

        For token i with word w_i:
            p(z_i=k | z_-i, w, pi) ∝ (alpha + n_{d,k}^{-i}) * pi_{k,w_i}

        This uses leave-one-out counts and runs `num_sweeps` full sweeps.
        """
        L = len(doc)
        if L == 0:
            return np.zeros(0, dtype=int)

        # initialize topics: warm-start if provided
        if z_init is None:
            z = np.random.randint(0, self.K, size=L)
        else:
            z = np.asarray(z_init, dtype=int).copy()
            if z.shape[0] != L:
                raise ValueError("z_init length does not match document length")
            # ensure valid range
            z = np.clip(z, 0, self.K - 1)

        counts = np.zeros(self.K, dtype=np.int64)
        for k in z:
            counts[int(k)] += 1

        for _ in range(num_sweeps):
            for i, word in enumerate(doc):
                k_old = int(z[i])
                counts[k_old] -= 1  # leave-one-out

                probs = (counts + alpha) * topic_word_dist[:, word]
                s = probs.sum()
                if (not np.isfinite(s)) or s <= 0:
                    probs = np.ones(self.K) / self.K
                else:
                    probs = probs / s

                k_new = np.random.choice(self.K, p=probs)
                z[i] = k_new
                counts[k_new] += 1

        return z

    def update_theta(self, mini_batch_docs, mini_batch_z, total_docs, stepsize):
        grad_theta = np.zeros_like(self.theta)

        # Mini-batch estimate of the sufficient-statistics difference
        # Use expanded-mean parametrization: pi = theta / sum(theta) per topic
        pi = self._normalize_theta(self.theta)
        for doc, z in zip(mini_batch_docs, mini_batch_z):
            counts = np.zeros((self.K, self.V))
            for word, topic in zip(doc, z):
                counts[topic, word] += 1
            n_k = counts.sum(axis=1, keepdims=True)
            # For each topic k and word w, contribution is n_{dkw} - pi_{kw} * n_{dk·}
            grad_theta += (counts - pi * n_k)

        # Scale to approximate full dataset (|D| / |D_t|), as in Patterson & Teh
        grad_theta *= 1/ len(mini_batch_docs)

        # Drift term: likelihood part + prior part (paper notation: beta)
        # This corresponds to (beta - theta) + scaled sum_d [n_{dkw} - pi_{kw} n_{dk·}]
        # drift = grad_theta 
        drift = grad_theta + (self.beta - self.theta)/self.D 
        # Langevin noise with metric-induced variance proportional to theta
        noise = np.random.randn(*self.theta.shape)
        eps = stepsize
        # Keep your requested form of the update and noise term
        # self.theta += 0.5 * eps * drift 
        self.theta += 0.5 * eps * drift + np.sqrt(eps * np.maximum(self.theta, 1e-12)/self.D ) * noise

        # Reflect across the boundary 0 < theta_{k,w} using mirroring (paper)
        self.theta = np.abs(self.theta)
        # topic_word_dist is the normalized version of theta, used in Gibbs and diagnostics
        self.topic_word_dist = self._normalize_theta(self.theta)

def match_topics(true_topic_word_dist, est_topic_word_dist):
    """Match topics between true and estimated topic-word distributions.

    Uses Hungarian algorithm on a cost matrix based on L2 distance between
    rows (topics). Returns an array mapping_est_to_true of shape (K,) where
    mapping_est_to_true[k_est] = k_true, i.e. estimated topic k_est is
    matched to true topic k_true.
    """
    true = np.asarray(true_topic_word_dist)
    est = np.asarray(est_topic_word_dist)
    K = true.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.linalg.norm(true[i] - est[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping_est_to_true = np.zeros(K, dtype=int)
    for i, j in zip(row_ind, col_ind):
        # row i (true) matched to col j (est)
        mapping_est_to_true[j] = i
    return mapping_est_to_true


# ===================== Semi-collapsed SVI for LDA (q(pi) only) =====================

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
    """Update responsibilities r_{i,k} for a single document under theta-collapsed VB."""
    L = w_d.size
    K, V = E_log_pi_kv.shape

    r = np.full((L, K), 1.0 / K, dtype=float)
    n_dk = r.sum(axis=0)

    for _ in range(local_iters):
        r_old = r.copy()

        for i in range(L):
            w = int(w_d[i])
            n_excl = n_dk - r[i]
            log_r_i = E_log_pi_kv[:, w] + digamma(alpha + n_excl)
            log_r_i -= logsumexp(log_r_i)
            r[i] = np.exp(log_r_i)
            n_dk = n_excl + r[i]

        max_diff = float(np.max(np.abs(r - r_old)))
        if max_diff < tol:
            break

    return r


def svi_lda_semi_collapsed(
    corpus: List[List[int]],
    K: int,
    V: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    iters: int = 1000,
    batch_docs: int = 64,
    local_iters: int = 10,
    tau0: float = 10.0,
    kappa: float = 0.7,
    seed: int = 0,
    verbose_every: int = 50,
) -> Dict[str, np.ndarray]:
    """SVI for semi-collapsed LDA (theta integrated out): update only global q(pi)."""
    rng = np.random.default_rng(seed)
    D = len(corpus)

    lam = beta + rng.random((K, V)) * 0.01
    elbo_proxy = np.zeros(iters, dtype=float)

    for t in range(1, iters + 1):
        rho_t = (tau0 + t) ** (-kappa)
        bsize = min(batch_docs, D)
        doc_idx = rng.choice(D, size=bsize, replace=False)

        E_log_pi_kv = _e_log_pi(lam)
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

            for i, w in enumerate(w_d):
                s_kw[:, int(w)] += r[i]

            lse = logsumexp(np.log(r + 1e-300) + E_log_pi_kv[:, w_d].T, axis=1)
            token_lp.append(float(np.mean(lse)))

        elbo_proxy[t - 1] = float(np.mean(token_lp)) if token_lp else np.nan

        scale = D / bsize
        s_kw *= scale
        lam_tilde = beta + s_kw
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


def remap_topics_matrix(mat_kv: np.ndarray, mapping_est_to_true: np.ndarray) -> np.ndarray:
    """Remap a (K,V) matrix from estimated topic order to TRUE topic order."""
    K, V = mat_kv.shape
    out = np.zeros_like(mat_kv)
    for k_est in range(K):
        k_true = int(mapping_est_to_true[k_est])
        out[k_true, :] = mat_kv[k_est, :]
    return out



# ===================== Dataset selection =====================
DATASET = "20ng"  # options: "synthetic", "20ng"
# ===================== Held-out evaluation controls =====================
# Evaluate predictive log-likelihood / perplexity on held-out documents.
HOLDOUT_FRACTION = 0.1  # 10% docs held out
HOLDOUT_MAX_DOCS = 500  # cap for speed; set None for no cap

# Predictive evaluation Monte Carlo
EVAL_PI_SAMPLES_SGRLD = 10   # number of π samples from SGRLD tail (stored sparsely)
EVAL_PI_SAMPLES_SVI   = 10   # number of π samples drawn from q(pi) under SVI
EVAL_GIBBS_SWEEPS_DOC = 30   # Gibbs sweeps to estimate doc topic proportions for predictive
# ===================== Gibbs comparison controls =====================
# Run gibbs2.py first to create this artifact (pi_samples, pi_mean, vocab, etc.)
# Set to None to disable Gibbs comparisons.
GIBBS_RESULTS_PATH = "gibbs_results_20ng.npz"
# ===================== Tracking / plotting controls =====================
TOP_TRACK = 10   # track/top-K parameters (k,w) by SGRLD tail mean
TOP_PLOT  = 5    # how many of those to show in trajectory + posterior figures
TAIL_T_PLOT = 2000  # tail length used for posterior histograms

# Manual tracked indices (topic, word). Use ONLY these for tracking/plot/posterior/calibration.
MANUAL_TRACKED_INDICES = [
    (7, 4216),
    (18, 8018),
    (19, 4064),
    (11, 9421),
    (5, 5457),
    (0, 6195),
    (17, 6696),
    (14, 8424),
    (11, 3179),
    (11, 0),
]

if DATASET == "synthetic":
    D_full, K, V = 10000, 3, 50
    corpus_full, z, true_topic_word_dist, _ = generate_synthetic_lda_data(D_full, K, V)
    has_true = True
    # Fallback vocabulary for synthetic data
    vocab = np.array([f"w{j}" for j in range(V)])
elif DATASET == "20ng":
    # Real dataset: 20 Newsgroups
    # Tip: start with smaller K (e.g., 10–20). You can increase later.
    K = 20
    corpus_full, V, _texts, _vec = load_20newsgroups_corpus(
        max_features=10000,
        min_df=10,
        max_df=0.5,
        stop_words="english",
        max_tokens_per_doc=400,
        seed=0,
    )
    D_full = len(corpus_full)
    true_topic_word_dist = None  # no ground-truth topics on real data
    has_true = False
    # Save vocabulary from vectorizer
    vocab = _vec.get_feature_names_out()
else:
    raise ValueError(f"Unknown DATASET={DATASET}")

# ---- train / held-out split ----
rng_split = np.random.default_rng(2025)
all_idx = np.arange(D_full)
rng_split.shuffle(all_idx)
nh = int(max(1, int(HOLDOUT_FRACTION * D_full)))
if HOLDOUT_MAX_DOCS is not None:
    nh = int(min(nh, int(HOLDOUT_MAX_DOCS)))
heldout_idx = all_idx[:nh]
train_idx = all_idx[nh:]

corpus_train = [corpus_full[int(i)] for i in train_idx]
corpus_test  = [corpus_full[int(i)] for i in heldout_idx]
D = len(corpus_train)
print(f"[split] D_full={D_full}  D_train={len(corpus_train)}  D_test={len(corpus_test)}")

model = SGRLD_LDA(D=D, V=V, K=K)

# --- Run SVI (semi-collapsed) on the same corpus for comparison ---
# NOTE: SVI returns a variational posterior q(pi_k)=Dirichlet(lambda_k).
# We'll compare SVI posterior marginals to SGRLD tail samples.
SVI_ITERS = 2000
svi_out = svi_lda_semi_collapsed(
    corpus_train,
    K=K,
    V=V,
    alpha=model.alpha,
    beta=model.beta,
    iters=SVI_ITERS,
    batch_docs=64,
    local_iters=10,
    tau0=10.0,
    kappa=0.7,
    seed=0,
    verbose_every=200,
)
lam_svi = svi_out["lambda"]
pi_svi_mean = svi_out["pi_mean"]
# Track token-level z histories for diagnostics
Z_TRACK_DOCS = 5
rng_diag = np.random.default_rng(123)
z_track_docs = rng_diag.choice(D, size=Z_TRACK_DOCS, replace=False).tolist()
z_history_docs = {d: [] for d in z_track_docs}
# --- SGRLD initialization ---
# Paper-style expanded-mean parameterisation uses an independent Gamma prior on theta.
# We initialise theta from the same family: theta_{k,w} ~ Gamma(shape=beta, scale=1).
# (Any positive initialisation is valid; this avoids starting unrealistically close to the truth.)
model.theta = np.random.gamma(shape=model.beta, scale=1.0, size=(K, V)).astype(float)
# numerical floor to avoid zero rows
model.theta += 1e-6
model.topic_word_dist = model._normalize_theta(model.theta)
# Track ONLY the user-provided indices.
tracked_indices = [(int(k), int(w)) for (k, w) in MANUAL_TRACKED_INDICES]
tracked_indices_plot = tracked_indices[:min(TOP_PLOT, len(tracked_indices))]

# Cosmetic: highlight the first entry.
largest_idx = tracked_indices[0]

# Ensure TOP_TRACK reflects the manual list.
TOP_TRACK = len(tracked_indices)

theta_history = {idx: [] for idx in tracked_indices}
# Do NOT store full π history (too expensive on real data).
# Instead, stream an average over the last T_tail iterations.
T_tail = 2000
pi_tail_sum = np.zeros((K, V), dtype=float)
pi_tail_count = 0
# Store a small number of tail π samples for predictive evaluation
pi_tail_samples = []  # list of (K,V) arrays
pi_tail_stride = max(1, T_tail // max(EVAL_PI_SAMPLES_SGRLD, 1))
# true_values will be constructed after training, using Hungarian matching.

#
# Train
np.random.seed(0)  # global reproducibility for numpy RNG used below
num_iterations = 50000
batch_size = 100
step_size = 100.0 * batch_size / D

# Progress logging
PROGRESS_EVERY = 1000  # print once every this many iterations
t0 = time.time()

# Persistent per-document topic assignments for warm-start Gibbs
z_state = [np.random.randint(0, K, size=len(doc)).astype(int) for doc in corpus_train]

for it in range(num_iterations):
    batch_indices = np.random.choice(D, batch_size, replace=False)

    z_batch = []
    for i in batch_indices:
        doc = corpus_train[i]
        z_new = model.gibbs_sample_z(
            doc,
            model.topic_word_dist,
            model.alpha,
            num_sweeps=2,
            z_init=z_state[i],
        )
        z_state[i] = z_new
        if i in z_history_docs and it > num_iterations * 0.5:
            z_history_docs[i].append(z_state[i].copy())
        z_batch.append(z_new)

    mini_batch_docs = [corpus_train[i] for i in batch_indices]

    for idx in tracked_indices:
        k, w = idx
        theta_history[idx].append(model.topic_word_dist[k, w])

    # Stream tail-mean π over the last T_tail iterations (no full history storage)
    if it >= (num_iterations - T_tail):
        pi_tail_sum += model.topic_word_dist
        pi_tail_count += 1
        # Keep ~EVAL_PI_SAMPLES_SGRLD samples evenly spaced in the tail
        if ((it - (num_iterations - T_tail)) % pi_tail_stride == 0) and (len(pi_tail_samples) < EVAL_PI_SAMPLES_SGRLD):
            pi_tail_samples.append(np.copy(model.topic_word_dist))

    model.update_theta(mini_batch_docs, z_batch, total_docs=D, stepsize=step_size)

    # ---- progress ----
    if PROGRESS_EVERY and ((it + 1) % PROGRESS_EVERY == 0 or (it + 1) == num_iterations):
        elapsed = time.time() - t0
        it_per_sec = (it + 1) / max(elapsed, 1e-9)

        # quick snapshot of tracked probabilities (current π at tracked indices)
        vals = []
        for (k, w) in tracked_indices[:min(3, len(tracked_indices))]:
            vals.append(float(model.topic_word_dist[k, w]))
        vals_str = ", ".join([f"{v:.6g}" for v in vals]) if vals else "(none)"

        # mean doc length in the current minibatch (useful sanity check)
        mean_L = float(np.mean([len(corpus_train[i]) for i in batch_indices]))

        print(
            f"[sgrld] iter {it+1}/{num_iterations}  "
            f"{it_per_sec:.2f} it/s  elapsed={elapsed/60:.1f}m  "
            f"mean_doc_len={mean_L:.1f}  step={step_size:.3g}  "
            f"trackedπ[0:3]=[{vals_str}]"
        )


#
#
# --- Hungarian matching to fix label switching before comparison ---
# Use the final estimated topic-word distribution for matching
est_topic_word_final = np.copy(model.topic_word_dist)
K_est = est_topic_word_final.shape[0]

if has_true:
    K_true = true_topic_word_dist.shape[0]

# Tail-mean π estimate from SGRLD (streamed over the last T_tail iterations)
if pi_tail_count <= 0:
    raise RuntimeError("pi_tail_count is 0; T_tail too large or training loop did not run.")
pi_tail_mean_est = pi_tail_sum / float(pi_tail_count)

# ===================== Load Gibbs results (optional) =====================
use_gibbs = (GIBBS_RESULTS_PATH is not None)
pi_gibbs_mean = None
pi_gibbs_samples = None

if use_gibbs:
    try:
        _g = np.load(GIBBS_RESULTS_PATH, allow_pickle=True)
        pi_gibbs_mean = np.asarray(_g["pi_mean"], dtype=float)
        pi_gibbs_samples = np.asarray(_g["pi_samples"], dtype=float)
        print(f"[INFO] Loaded Gibbs results: {GIBBS_RESULTS_PATH}")
        print(f"       pi_mean shape={pi_gibbs_mean.shape}, pi_samples shape={pi_gibbs_samples.shape}")

        # Sanity checks: must match current (K,V)
        if pi_gibbs_mean.shape != (K, V):
            print(f"[WARN] Gibbs pi_mean shape {pi_gibbs_mean.shape} != (K,V)=({K},{V}). Disabling Gibbs compare.")
            use_gibbs = False
        if pi_gibbs_samples.ndim != 3 or pi_gibbs_samples.shape[1:] != (K, V):
            print(f"[WARN] Gibbs pi_samples shape {pi_gibbs_samples.shape} not compatible with (S,K,V)=(*,{K},{V}). Disabling Gibbs compare.")
            use_gibbs = False
    except FileNotFoundError:
        print(f"[WARN] Gibbs results file not found: {GIBBS_RESULTS_PATH}. Disabling Gibbs compare.")
        use_gibbs = False
    except Exception as e:
        print(f"[WARN] Failed to load Gibbs results from {GIBBS_RESULTS_PATH}: {e}. Disabling Gibbs compare.")
        use_gibbs = False

# Print TOP-10 entries from the SGRLD tail-mean π (in estimated-topic space)
print(f"\n{CANONICAL_PREFIX} Top-10 π entries | SGRLD | tail-mean π (estimated-topic order):")
flat_tail = pi_tail_mean_est.reshape(-1)
top_idx_tail10 = np.argsort(flat_tail)[-10:][::-1]
for idx in top_idx_tail10:
    k = int(idx // V)
    w = int(idx % V)
    val = float(pi_tail_mean_est[k, w])
    print(f"  SGRLD π(topic={k}, word={w}) = {val:.6f}")
print("=" * 69 + "\n")


# --- Topic matching for comparisons ---
# Canonical topic order:
#   - If Gibbs is available: use Gibbs topics as canonical (stable baseline)
#   - Else: use SGRLD tail-mean topics as canonical (previous behavior)

# Always define these to avoid NameError in downstream printing / logic.
mapping_svi_to_sgrld = None
mapping_svi_to_gibbs = None

# --- Topic matching for comparisons ---
# Canonical topic order:
#   - If Gibbs is available: use Gibbs topics as canonical (stable baseline)
#   - Else: use SGRLD tail-mean topics as canonical (previous behavior)

if use_gibbs:
    # Match SGRLD -> Gibbs (mapping_est_to_true semantics: mapping_sgrld_to_gibbs[k_sgrld] = k_gibbs)
    mapping_sgrld_to_gibbs = match_topics(pi_gibbs_mean, pi_tail_mean_est)
    pi_tail_mean_est = remap_topics_matrix(pi_tail_mean_est, mapping_sgrld_to_gibbs)

    # Remap SGRLD theta_history keys into Gibbs order
    theta_history_g = {}
    for (k_est, w), series in theta_history.items():
        k_g = int(mapping_sgrld_to_gibbs[int(k_est)])
        theta_history_g[(k_g, int(w))] = series
    theta_history = theta_history_g

    # Remap tracked indices (these were specified in SGRLD topic labels)
    tracked_indices = [(int(mapping_sgrld_to_gibbs[int(k)]), int(w)) for (k, w) in tracked_indices]
    tracked_indices_plot = tracked_indices[:min(TOP_PLOT, len(tracked_indices))]

    # Match SVI -> Gibbs
    if pi_svi_mean is not None and pi_svi_mean.shape[0] == K_est:
        mapping_svi_to_gibbs = match_topics(pi_gibbs_mean, pi_svi_mean)
        pi_svi_mean_true = remap_topics_matrix(pi_svi_mean, mapping_svi_to_gibbs)
        lam_svi_true = remap_topics_matrix(lam_svi, mapping_svi_to_gibbs)
        # Also alias for code that expects mapping_svi_to_sgrld
        mapping_svi_to_sgrld = mapping_svi_to_gibbs
    else:
        mapping_svi_to_gibbs = None
        pi_svi_mean_true = pi_svi_mean
        lam_svi_true = lam_svi
        mapping_svi_to_sgrld = mapping_svi_to_gibbs

    # For convenience later
    pi_svi_mean_sgrld = pi_svi_mean_true
    lam_svi_sgrld = lam_svi_true

    print("[INFO] Applied topic matching: SGRLD→Gibbs and SVI→Gibbs. Using Gibbs topic order as canonical.")

else:
    # Fall back: match SVI to SGRLD (your previous behavior)
    if pi_svi_mean is not None and pi_svi_mean.shape[0] == K_est:
        mapping_svi_to_sgrld = match_topics(pi_tail_mean_est, pi_svi_mean)
        pi_svi_mean_sgrld = remap_topics_matrix(pi_svi_mean, mapping_svi_to_sgrld)
        lam_svi_sgrld = remap_topics_matrix(lam_svi, mapping_svi_to_sgrld)
        pi_svi_mean_true = pi_svi_mean_sgrld
        lam_svi_true = lam_svi_sgrld
    else:
        mapping_svi_to_sgrld = None
        pi_svi_mean_sgrld = pi_svi_mean
        lam_svi_sgrld = lam_svi
        pi_svi_mean_true = pi_svi_mean_sgrld
        lam_svi_true = lam_svi_sgrld

    print("[INFO] Applied topic matching: SVI→SGRLD. Using SGRLD topic order as canonical.")

# ---- Print top-10 words per topic (SGRLD tail-mean π, AFTER matching) ----
print(f"\n================ {CANONICAL_PREFIX} Top-10 words per topic | SGRLD | tail-mean π ================")
for k in range(K):
    probs = pi_tail_mean_est[k]
    top_idx = np.argsort(probs)[-10:][::-1]
    entries = []
    for j in top_idx:
        word = vocab[j] if j < len(vocab) else str(j)
        entries.append(f"{word}:{probs[j]:.4f}")
    print(f"Topic {k}: " + ", ".join(entries))
print("=" * 77 + "\n")
# ---- Print top-10 words per topic (SVI mean π) ----
print(f"\n================ {CANONICAL_PREFIX} Top-10 words per topic | SVI | mean π =========================")
if pi_svi_mean_true is None:
    pi_svi_mean_true = pi_svi_mean
for k in range(K):
    probs = pi_svi_mean_true[k]
    top_idx = np.argsort(probs)[-10:][::-1]
    entries = []
    for j in top_idx:
        word = vocab[j] if j < len(vocab) else str(j)
        entries.append(f"{word}:{probs[j]:.4f}")
    print(f"Topic {k}: " + ", ".join(entries))
print("=" * 77 + "\n")

# Print SVI results for the user-provided tracked indices (after SVI→SGRLD matching).
print(f"\n================ {CANONICAL_PREFIX} SVI tracked indices | SGRLD tail-mean vs SVI mean and Beta =================")
if mapping_svi_to_sgrld is not None:
    print("[INFO] Applied Hungarian matching: SVI topics remapped into SGRLD topic order.")
else:
    print("[WARN] Could not match SVI topics to SGRLD (shape mismatch); printing raw SVI order.")

for (k, w) in tracked_indices:
    sgrld_tm = float(pi_tail_mean_est[k, w])
    svi_m = float(pi_svi_mean_sgrld[k, w])
    a = float(lam_svi_sgrld[k, w])
    b = float(np.sum(lam_svi_sgrld[k, :]) - lam_svi_sgrld[k, w])
    b = max(b, 1e-12)
    print(f"(topic={k}, word={w}):  SGRLD tail-mean={sgrld_tm:.6f}   SVI mean={svi_m:.6f}   SVI Beta(a={a:.3f}, b={b:.3f})")
print("=" * 89 + "\n")

# Keep the manual tracked indices selected BEFORE training.
tracked_indices_all = tracked_indices
tracked_indices_plot = tracked_indices[:min(TOP_PLOT, len(tracked_indices))]
tracked_indices = tracked_indices_all
if has_true:
    print(f"\n================ {CANONICAL_PREFIX} TRUE topic-word distribution ================")
    print(true_topic_word_dist)

    # --- Print largest entries for TRUE topic-word distribution ---
    print(f"\n{CANONICAL_PREFIX} Top-20 π entries | TRUE topic-word distribution:")
    flat_true = true_topic_word_dist.flatten()
    top_idx_true = np.argsort(flat_true)[-20:][::-1]
    for i in top_idx_true:
        k = i // V
        w = i % V
        val = float(true_topic_word_dist[k, w])
        print(f"  TRUE π(topic={k}, word={w}) = {val:.6f}")
else:
    print(f"\n[INFO] Real dataset selected: no TRUE parameters available; skipping TRUE π prints.")

# Compute mean of last 2000 iterations for each tracked (k,w)
mean_last2000 = {}
for idx, series in theta_history.items():
    arr = np.asarray(series)
    tail = arr[-2000:] if arr.shape[0] >= 2000 else arr
    mean_last2000[idx] = float(np.mean(tail))

print(f"\n================ {CANONICAL_PREFIX} Mean of last 2000 iterations per param (matched) ================")
for idx, val in mean_last2000.items():
    print(f"param (est-topic={idx[0]}, word={idx[1]}): mean last2000 = {val:.6f}")
print("=" * 69 + "\n")

if has_true and (K_est == K_true):
    mapping_est_to_true = match_topics(true_topic_word_dist, est_topic_word_final)
    print("Hungarian matching applied between true and estimated topics.")

    # --- Also match SVI topics to TRUE topics (separately) and remap SVI outputs ---
    mapping_svi_to_true = match_topics(true_topic_word_dist, pi_svi_mean)
    pi_svi_mean_true = remap_topics_matrix(pi_svi_mean, mapping_svi_to_true)
    lam_svi_true = remap_topics_matrix(lam_svi, mapping_svi_to_true)

    # Remap theta_history and tracked_indices so that the first index is the TRUE topic index
    theta_history_matched = {}
    tracked_indices_matched = []
    for (k_est, w), series in theta_history.items():
        k_true = int(mapping_est_to_true[k_est])
        idx_new = (k_true, w)
        theta_history_matched[idx_new] = series
        if idx_new not in tracked_indices_matched:
            tracked_indices_matched.append(idx_new)

    theta_history = theta_history_matched
    tracked_indices = tracked_indices_matched

    # Keep the manual tracked indices; after remapping theta_history to TRUE-topic space,
    # the same (topic, word) pairs now refer to TRUE-topic indices.
    tracked_indices_all = tracked_indices
    tracked_indices_plot = tracked_indices[:min(TOP_PLOT, len(tracked_indices))]

    # Build matched π history in TRUE-topic space using the fixed mapping
    # (pi_history not available: skip building pi_history_matched)

    # Also remap the largest_idx topic index into true-topic space
    k_est_largest, w_largest = largest_idx
    k_true_largest = int(mapping_est_to_true[k_est_largest])
    largest_idx = (k_true_largest, w_largest)
    # Note: `largest_idx` may not be among the selected top-10 indices; in that case no line is highlighted.

    # Remap final SGRLD topic-word distribution into TRUE topic index space
    pi_sgrld_matched = np.zeros_like(est_topic_word_final)
    for k_est in range(K_est):
        k_true = int(mapping_est_to_true[k_est])
        pi_sgrld_matched[k_true, :] = est_topic_word_final[k_est, :]

    # === Print TOP 20 entries of π from SGRLD tail mean (average of last iterations) ===
    # Use the same remapped tail-mean π as for plotting
    pi_tail_mean = pi_tail_mean_true

    print(f"\n{CANONICAL_PREFIX} Top-20 π entries | SGRLD | tail mean (canonical order):")
    flat_tail = pi_tail_mean.flatten()
    top_idx_tail = np.argsort(flat_tail)[-20:][::-1]
    for idx in top_idx_tail:
        k = idx // V
        w = idx % V
        val = float(pi_tail_mean[k, w])
        print(f"  SGRLD π(topic={k}, word={w}) = {val:.6f}")
else:
    print("[WARN] K_est != K_true; skipping Hungarian matching.")
    # In this case, keep SGRLD topics in their estimated order
    pi_sgrld_matched = est_topic_word_final
    # Use SVI remapped into SGRLD topic order (computed above)
    pi_svi_mean_true = pi_svi_mean_sgrld
    lam_svi_true = lam_svi_sgrld
    # Keep the canonical tracked lists determined above (manual override or automatic).
    # Do not recompute here.
    tracked_indices = tracked_indices_all
    tracked_indices_plot = tracked_indices_all[:min(TOP_PLOT, len(tracked_indices_all))]

# Now construct true_values only if available
if has_true:
    # Truth values for the plotted subset
    true_values = {idx: true_topic_word_dist[idx[0], idx[1]] for idx in tracked_indices_plot}
else:
    true_values = {}


import itertools

colors = plt.cm.tab10.colors  # up to 10 distinct colors
color_cycle = itertools.cycle(colors)
# One color per parameter, consistent across all figures.
color_map = {idx: next(color_cycle) for idx in tracked_indices}  # TOP_TRACK params

plt.figure(figsize=(12, 6))
for idx in tracked_indices_plot:
    k, w = idx
    is_largest = (idx == largest_idx)
    lw = 2.8 if is_largest else 1.0
    lab_suffix = " (largest)" if is_largest else ""
    c = color_map[idx]

    # SGRLD trajectory
    plt.plot(theta_history[idx], linewidth=lw, color=c,
             label=f"SGRLD π_{k},{w}{lab_suffix}")

    # TRUE value reference line
    if has_true and idx in true_values:
        tv = float(true_values[idx])
        plt.axhline(tv, color=c, linestyle='--', linewidth=1.2,
                    label=f"TRUE π_{k},{w}={tv:.3f}")

# Canonicalized plot title and axis labels
plt.title(f"{CANONICAL_PREFIX} Topic-word probabilities | SGRLD | tracked indices (iterations={num_iterations})")
plt.xlabel("Iteration")
plt.ylabel("Probability π_{k,w}")
# Auto y-limit: zoom to the scale of tracked probabilities (use tail + head for stability)
_vals = []
for _idx in tracked_indices_plot:
    _s = np.asarray(theta_history.get(_idx, []), dtype=float)
    if _s.size:
        _vals.append(float(np.max(_s)))
if _vals:
    _ymax = max(_vals)
    # Zoom for small probabilities: keep headroom but avoid stretching to 0.5.
    _ymax = _ymax * 1.35
    _ymax = min(_ymax, 0.08)   # cap for readability on real data (adjust if needed)
    _ymax = max(_ymax, 0.01)   # minimum visible range
    plt.ylim(0.0, _ymax)
else:
    plt.ylim(0.0, 0.5)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# ===================== Combined posterior: SGRLD tail vs SVI q(pi) =====================

def plot_posterior_sgrld_vs_svi(
    theta_hist_dict,
    indices,
    lam_svi_true,
    *,
    tail_T=2000,
    color_map=None,
    title=None,
):
    """Posterior comparison for selected (k,w): SGRLD tail Gaussian fit vs SVI Gaussian approx.

    - SGRLD: fit N(mean(tail), var(tail)) to the tail samples.
    - SVI: approximate the Beta marginal for pi_{k,w} by a Gaussian with matched mean/variance.

    Notes
    -----
    We deliberately avoid histograms to match the paper-style figures.
    """
    n = int(len(indices)) if indices is not None else 0
    if n <= 0:
        print("[WARN] No indices to plot posterior comparison")
        return

    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.6 * nrows),
        sharex=False,
        sharey=False,
    )

    axes = np.asarray(axes).reshape(-1)

    for ax, (k, w) in zip(axes[:n], indices):
        k = int(k)
        w = int(w)

        series = np.asarray(theta_hist_dict.get((k, w), []), dtype=float)
        tail = series[-tail_T:] if series.size >= int(tail_T) else series
        if tail.size == 0:
            ax.set_title(fr"$\pi_{{{k},{w}}}$ (no history)")
            ax.grid(True, alpha=0.25)
            continue

        # Consistent color per parameter
        c = None
        if isinstance(color_map, dict) and ((k, w) in color_map):
            c = color_map[(k, w)]

        # ---- SGRLD Gaussian fit (tail samples) ----
        m_sgrld = float(np.mean(tail))
        v_sgrld = float(np.var(tail, ddof=1)) if tail.size >= 2 else 0.0
        s_sgrld = float(np.sqrt(max(v_sgrld, 1e-18)))

        # ---- SVI Gaussian approx (match Beta mean/var) ----
        a = float(lam_svi_true[k, w])
        b = float(np.sum(lam_svi_true[k, :]) - lam_svi_true[k, w])
        b = max(b, 1e-12)
        m_svi = float(a / (a + b))
        v_svi = float((a * b) / (((a + b) ** 2) * (a + b + 1.0)))
        s_svi = float(np.sqrt(max(v_svi, 1e-18)))

        # Plot range: cover both spreads; probabilities are in [0,1]
        lo = max(0.0, min(m_sgrld - 5.0 * s_sgrld, m_svi - 5.0 * s_svi))
        hi = min(1.0, max(m_sgrld + 5.0 * s_sgrld, m_svi + 5.0 * s_svi))
        # If spreads are tiny, still show a reasonable local window
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
            lo = max(0.0, min(m_sgrld, m_svi) - 0.02)
            hi = min(1.0, max(m_sgrld, m_svi) + 0.02)
        xs = np.linspace(lo, hi, 500)

        # Gaussian pdfs
        def _gauss_pdf(x, m, s):
            s = max(float(s), 1e-12)
            return (1.0 / (s * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)

        ax.plot(
            xs,
            _gauss_pdf(xs, m_sgrld, s_sgrld),
            color=c,
            linewidth=2.0,
            alpha=0.90,
            label="SGRLD (Gaussian fit)",
        )
        ax.plot(
            xs,
            _gauss_pdf(xs, m_svi, s_svi),
            color=c,
            linewidth=2.0,
            linestyle="--",
            alpha=0.90,
            label="SVI (Gaussian approx)",
        )

        ax.set_title(fr"$\pi_{{{k},{w}}}$")
        ax.set_xlabel(r"$\pi_{k,w}$")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    for ax in axes[n:]:
        ax.axis("off")
    if title is None:
        title = f"{CANONICAL_PREFIX} Posterior comparison | SGRLD vs SVI (Gaussian overlays)"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ===================== Combined posterior: SGRLD tail vs SVI q(pi) =====================

# --- select up to 10 parameters for posterior comparison ---
# Priority:
#   1) user-provided tracked indices (after canonical topic remapping)
#   2) fill the remainder with the largest entries from the SGRLD tail-mean π
POSTERIOR_PLOT_N = 10

def _select_posterior_indices_10(pi_tail_mean_canon, tracked, V, n=10):
    tracked = [(int(k), int(w)) for (k, w) in (tracked or [])]
    out = []
    used = set()
    for idx in tracked:
        if idx not in used:
            out.append(idx)
            used.add(idx)
        if len(out) >= int(n):
            return out

    # Fill with top-(k,w) by tail-mean probability
    pi_flat = np.asarray(pi_tail_mean_canon, dtype=float).reshape(-1)
    top = np.argsort(pi_flat)[::-1]
    for t in top:
        k = int(t // int(V))
        w = int(t % int(V))
        idx = (k, w)
        if idx in used:
            continue
        out.append(idx)
        used.add(idx)
        if len(out) >= int(n):
            break
    return out

# Use canonical (already-remapped) tail-mean π for selection.
posterior_indices = _select_posterior_indices_10(pi_tail_mean_est, tracked_indices_all, V, n=POSTERIOR_PLOT_N)

plot_posterior_sgrld_vs_svi(
    theta_history,
    posterior_indices,
    lam_svi_true,
    tail_T=TAIL_T_PLOT,
    color_map=color_map,
    title=f"Posterior comparison (Gaussian overlays) — {len(posterior_indices)} parameters",
)

# ===================== 3-way posterior overlay: Gibbs vs SGRLD vs SVI =====================

def plot_posterior_gibbs_sgrld_svi(
    *,
    theta_hist_dict,
    gibbs_pi_samples,
    lam_svi_true,
    indices,
    tail_T=2000,
    title=None,
):
    """
    Curve-only posterior comparison for each (k,w):
      - Gibbs: Gaussian fit to Gibbs samples
      - SGRLD: Gaussian fit to tail samples
      - SVI: exact Beta marginal density
    All inputs must already be in the same canonical topic order.
    """
    if indices is None or len(indices) == 0:
        print("[WARN] No indices provided for 3-way posterior plot")
        return

    n = len(indices)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 2.8 * nrows),
        sharex=False, sharey=False
    )
    axes = np.asarray(axes).reshape(-1)

    def _gauss_pdf(x, m, s):
        s = max(float(s), 1e-12)
        return (1.0 / (s * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)

    for ax, (k, w) in zip(axes, indices):
        k = int(k)
        w = int(w)

        # ---- Gibbs Gaussian fit ----
        gibbs_series = np.asarray(gibbs_pi_samples[:, k, w], dtype=float)
        gibbs_series = gibbs_series[np.isfinite(gibbs_series)]
        m_g = float(np.mean(gibbs_series))
        v_g = float(np.var(gibbs_series, ddof=1)) if gibbs_series.size >= 2 else 0.0
        s_g = float(np.sqrt(max(v_g, 1e-18)))

        # ---- SGRLD Gaussian fit ----
        series = np.asarray(theta_hist_dict.get((k, w), []), dtype=float)
        tail = series[-tail_T:] if series.size >= tail_T else series
        m_s = float(np.mean(tail))
        v_s = float(np.var(tail, ddof=1)) if tail.size >= 2 else 0.0
        s_s = float(np.sqrt(max(v_s, 1e-18)))

        # ---- SVI Beta ----
        a = float(lam_svi_true[k, w])
        b = float(np.sum(lam_svi_true[k, :]) - lam_svi_true[k, w])
        b = max(b, 1e-12)

        # x-range
        lo = max(0.0, min(m_g - 5*s_g, m_s - 5*s_s))
        hi = min(1.0, max(m_g + 5*s_g, m_s + 5*s_s))
        if (hi - lo) < 1e-6:
            lo = max(0.0, min(m_g, m_s) - 0.02)
            hi = min(1.0, max(m_g, m_s) + 0.02)

        xs = np.linspace(lo, hi, 500)

        # Plot curves
        ax.plot(xs, _gauss_pdf(xs, m_g, s_g),
                linewidth=2.0, label="Gibbs (Gaussian fit)")
        ax.plot(xs, _gauss_pdf(xs, m_s, s_s),
                linewidth=2.0, linestyle="--", label="SGRLD (Gaussian fit)")
        ax.plot(xs, beta_dist.pdf(xs, a, b),
                linewidth=2.0, linestyle=":", label="SVI (Beta)")

        ax.set_title(fr"$\pi_{{{k},{w}}}$")
        ax.set_xlabel(r"$\pi_{k,w}$")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    for ax in axes[n:]:
        ax.axis("off")

    if title is None:
        title = f"{CANONICAL_PREFIX} Posterior overlay | Gibbs vs SGRLD vs SVI"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if use_gibbs and (pi_gibbs_samples is not None):
    plot_posterior_gibbs_sgrld_svi(
        theta_hist_dict=theta_history,
        gibbs_pi_samples=pi_gibbs_samples,
        lam_svi_true=lam_svi_true,
        indices=posterior_indices,
        tail_T=TAIL_T_PLOT,
        title=f"{CANONICAL_PREFIX} Posterior overlay | Gibbs vs SGRLD vs SVI (top-{len(posterior_indices)})",
    )
# ===================== Held-out predictive evaluation =====================

def _doc_loglik_plug_in_eta(model: SGRLD_LDA, doc_words: List[int], pi_kv: np.ndarray, alpha: float, *, gibbs_sweeps: int = 30, seed: int = 0) -> float:
    """Approximate log p(w_d | pi, alpha) using a plug-in estimate of eta_d."""
    rng = np.random.default_rng(seed)
    doc_words = np.asarray(doc_words, dtype=int)
    L = int(doc_words.size)
    if L == 0:
        return 0.0

    z0 = rng.integers(0, model.K, size=L).astype(int)
    z = model.gibbs_sample_z(doc_words.tolist(), pi_kv, alpha, num_sweeps=gibbs_sweeps, z_init=z0)

    n_dk = np.bincount(np.asarray(z, dtype=int), minlength=model.K).astype(float)
    eta = (alpha + n_dk)
    eta = eta / float(np.sum(eta))

    pw = eta @ pi_kv[:, doc_words]
    pw = np.maximum(pw, 1e-300)
    return float(np.sum(np.log(pw)))


def heldout_predictive_ll_and_perplexity_sgrld(model, corpus_test, pi_samples, pi_fallback, *, alpha, gibbs_sweeps=30):
    if pi_samples is None or len(pi_samples) == 0:
        pi_samples = [pi_fallback]

    total_ll = 0.0
    total_tokens = 0

    for d, doc in enumerate(corpus_test):
        L = len(doc)
        if L == 0:
            continue
        total_tokens += L
        lls = []
        for t, pi in enumerate(pi_samples):
            lls.append(_doc_loglik_plug_in_eta(model, doc, pi, alpha, gibbs_sweeps=gibbs_sweeps, seed=10_000 + 97 * d + t))
        total_ll += float(np.mean(lls))  # mean-of-logs (as requested)

    ppl = float(np.exp(-total_ll / max(total_tokens, 1)))
    return {"heldout_loglik": float(total_ll), "heldout_tokens": int(total_tokens), "perplexity": ppl, "T_pi": int(len(pi_samples))}


def heldout_predictive_ll_and_perplexity_svi(model, corpus_test, lam_svi_true, *, alpha, pi_samples=10, gibbs_sweeps=30, seed=0):
    rng = np.random.default_rng(seed)
    K, V = lam_svi_true.shape

    def _sample_pi():
        pi = np.empty((K, V), dtype=float)
        for k in range(K):
            pi[k, :] = dirichlet_dist.rvs(lam_svi_true[k, :], random_state=rng)[0]
        return pi

    pis = [_sample_pi() for _ in range(max(1, int(pi_samples)))]

    total_ll = 0.0
    total_tokens = 0

    for d, doc in enumerate(corpus_test):
        L = len(doc)
        if L == 0:
            continue
        total_tokens += L
        lls = []
        for t, pi in enumerate(pis):
            lls.append(_doc_loglik_plug_in_eta(model, doc, pi, alpha, gibbs_sweeps=gibbs_sweeps, seed=20_000 + 131 * d + t))
        total_ll += float(np.mean(lls))  # mean-of-logs

    ppl = float(np.exp(-total_ll / max(total_tokens, 1)))
    return {"heldout_loglik": float(total_ll), "heldout_tokens": int(total_tokens), "perplexity": ppl, "T_pi": int(len(pis))}


print(f"\n================ {CANONICAL_PREFIX} Held-out predictive evaluation ================")
print(f"{CANONICAL_PREFIX} Test docs: {len(corpus_test)}  (max_tokens_per_doc already applied if 20ng)")

if pi_tail_count <= 0:
    raise RuntimeError("pi_tail_count is 0; cannot evaluate SGRLD")
pi_tail_mean_eval = pi_tail_sum / float(pi_tail_count)

res_sgrld = heldout_predictive_ll_and_perplexity_sgrld(
    model,
    corpus_test,
    pi_tail_samples,
    pi_tail_mean_eval,
    alpha=model.alpha,
    gibbs_sweeps=EVAL_GIBBS_SWEEPS_DOC,
)
print(
    f"{CANONICAL_PREFIX} [SGRLD] heldout_loglik={res_sgrld['heldout_loglik']:.2f}  "
    f"tokens={res_sgrld['heldout_tokens']}  "
    f"avg_ll/token={res_sgrld['heldout_loglik']/max(res_sgrld['heldout_tokens'],1):.4f}  "
    f"perplexity={res_sgrld['perplexity']:.3f}  "
    f"T_pi={res_sgrld['T_pi']}"
)

res_svi = heldout_predictive_ll_and_perplexity_svi(
    model,
    corpus_test,
    lam_svi_true,
    alpha=model.alpha,
    pi_samples=EVAL_PI_SAMPLES_SVI,
    gibbs_sweeps=EVAL_GIBBS_SWEEPS_DOC,
    seed=2026,
)
print(
    f"{CANONICAL_PREFIX} [SVI ] heldout_loglik={res_svi['heldout_loglik']:.2f}  "
    f"tokens={res_svi['heldout_tokens']}  "
    f"avg_ll/token={res_svi['heldout_loglik']/max(res_svi['heldout_tokens'],1):.4f}  "
    f"perplexity={res_svi['perplexity']:.3f}  "
    f"T_pi={res_svi['T_pi']}"
)
print("=" * 63 + "\n")
# ===================== Posterior plots (SGRLD tail samples) =====================

def plot_posterior_tracked(theta_hist_dict, true_topic_word_dist, indices, *, tail_T=2000, algo_name="SGRLD"):
    """Plot approximate posterior (tail histogram) for each tracked π_{k,w}.

    Uses the last `tail_T` iterations from `theta_hist_dict[(k,w)]` as samples.
    Overlays a Gaussian approximation using the sample mean/variance and marks the true value.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    for (k, w) in indices:
        series = _np.asarray(theta_hist_dict.get((k, w), []), dtype=float)
        if series.size == 0:
            print(f"[WARN] No history for (k={k}, w={w}); skipping posterior plot")
            continue
        tail = series[-tail_T:] if series.size >= tail_T else series

        m = float(_np.mean(tail))
        v = float(_np.var(tail, ddof=1)) if tail.size >= 2 else 0.0
        sd = float(_np.sqrt(max(v, 1e-18)))

        if has_true:
            true_val = float(true_topic_word_dist[k, w])
        else:
            true_val = None

        _plt.figure(figsize=(6.5, 4.2))
        _plt.hist(tail, bins=40, density=True, alpha=0.6, label=f"{algo_name} tail samples")

        # Gaussian approximation from tail mean/var
        xs = _np.linspace(max(0.0, m - 4 * sd), min(1.0, m + 4 * sd), 400) if sd > 0 else _np.linspace(0.0, 1.0, 400)
        if sd > 0:
            gauss = (1.0 / (sd * _np.sqrt(2 * _np.pi))) * _np.exp(-0.5 * ((xs - m) / sd) ** 2)
            _plt.plot(xs, gauss, linewidth=2.0, label="Gaussian fit (mean/var)")

        if has_true and true_val is not None:
            _plt.axvline(true_val, linestyle='--', linewidth=2.0, label=f"TRUE π_{k},{w}={true_val:.3f}")
        _plt.axvline(m, linestyle='-', linewidth=1.5, label=f"Tail mean={m:.3f}")

        _plt.title(f"{CANONICAL_PREFIX} Posterior approx | {algo_name} | π(topic={k}, word={w})")
        _plt.xlabel("π_{k,w}")
        _plt.ylabel("Density")
        _xmax = float(_np.max(tail)) * 1.6
        _xmax = max(_xmax, 0.05)
        _xmax = min(_xmax, 0.5)
        _plt.xlim(0.0, _xmax)
        _plt.grid(True, alpha=0.3)
        _plt.legend()
        _plt.tight_layout()
        _plt.show()





def plot_rank_uniformity_pair(theta_hist_dict, true_topic_word_dist, k, w1, w2, algo_name="Algo", start_frac=0.5):
    """
    Rank-uniformity calibration for two parameters θ_{k,w1}, θ_{k,w2} using an iterate history.
    For each parameter j in {w1, w2}, compute p_j = mean(θ_j^{(t)} > θ_j^*), t over the tail [start_frac, 1].
    Plot sorted {p_j} against reference line y = x/(D+1) with D=2.
    """
    import matplotlib.pyplot as _plt
    D = 2
    x_ref = np.arange(1, D + 1)
    y_ref = x_ref / (D + 1.0)  # [1/3, 2/3]

    idxs = [(k, w1), (k, w2)]
    ps = []
    labels = []
    for idx in idxs:
        if idx not in theta_hist_dict or len(theta_hist_dict[idx]) == 0:
            continue
        series = np.asarray(theta_hist_dict[idx])
        start = int(len(series) * start_frac)
        tail = series[start:]
        if has_true:
            true_val = float(true_topic_word_dist[idx[0], idx[1]])
            p = float(np.mean(tail > true_val))
            ps.append(p)
            labels.append(f"θ_{{{idx[0]},{idx[1]}}}")

    if len(ps) == 0:
        print(f"[WARN] No history found for k={k}, w1={w1}, w2={w2} in {algo_name}")
        return

    ps_sorted = np.sort(np.asarray(ps))
    _plt.figure(figsize=(5, 4))
    _plt.plot(x_ref, ps_sorted, marker='o', linestyle='-', linewidth=1.2, label='Empirical')
    _plt.plot(x_ref, y_ref, linestyle='-', label='y = x/(D+1)')
    _plt.title(f"{CANONICAL_PREFIX} Calibration (rank-uniformity) | {algo_name} | (k={k}, w1={w1}, w2={w2})")
    _plt.xlabel("rank")
    _plt.ylabel("P(θ > θ*) over tail")
    _plt.ylim(-0.05, 1.05)
    _plt.grid(True, alpha=0.3)
    _plt.legend()
    _plt.tight_layout()
    _plt.show()




# --- Helper: Rank-uniformity calibration plot for all tracked parameters ---
def plot_rank_uniformity_all(theta_hist_dict, true_topic_word_dist, indices, algo_name="Algo", start_frac=0.5):
    """
    Rank-uniformity using ALL tracked parameters in `indices`.
    For each (k,w) in indices, compute p_{k,w} = mean(θ_{k,w}^{(t)} > θ*_{k,w}) over the tail [start_frac, 1].
    Plot sorted {p} against the reference line y = x/(D+1), where D = len(indices).
    """
    import matplotlib.pyplot as _plt
    if not indices:
        print(f"[WARN] No indices provided for rank-uniformity ({algo_name})")
        return
    ps = []
    if has_true:
        for (k, w) in indices:
            series = np.asarray(theta_hist_dict.get((k, w), []), dtype=float)
            if series.size == 0:
                continue
            start = int(series.size * start_frac)
            tail = series[start:]
            if tail.size == 0:
                continue
            true_val = float(true_topic_word_dist[k, w])
            p = float(np.mean(tail > true_val))
            ps.append(p)
    if len(ps) == 0:
        print(f"[WARN] No valid histories for provided indices in {algo_name}")
        return
    ps = np.sort(np.asarray(ps))
    D = ps.size
    x_ref = np.arange(1, D + 1)
    y_ref = x_ref / (D + 1.0)
    _plt.figure(figsize=(6, 5))
    _plt.plot(x_ref, ps, marker='o', linestyle='-', linewidth=1.2, label='Empirical')
    _plt.plot(x_ref, y_ref, linestyle='-', label='y = x/(D+1)')
    _plt.title(f"{CANONICAL_PREFIX} Calibration (rank-uniformity) | {algo_name} | tracked params (D={D})")
    _plt.xlabel("rank")
    _plt.ylabel("P(θ > θ*) over tail")
    _plt.ylim(-0.05, 1.05)
    _plt.grid(True, alpha=0.3)
    _plt.legend()
    _plt.tight_layout()
    _plt.show()


# --- Helper: Rank-uniformity comparison plot for SGRLD vs SVI ---
def plot_rank_uniformity_compare_sgrld_vs_svi(theta_hist_dict, true_pi, indices, lam_svi_true, *, start_frac=0.5):
    """Compare rank-uniformity p-values for SGRLD tail and SVI q(pi) marginals."""
    if not indices:
        print("[WARN] No indices for calibration compare")
        return

    # SGRLD p-values
    ps_sgrld = []
    ps_svi = []
    for (k, w) in indices:
        series = np.asarray(theta_hist_dict.get((k, w), []), dtype=float)
        if series.size == 0:
            continue
        start = int(series.size * start_frac)
        tail = series[start:]
        if tail.size == 0:
            continue
        true_val = float(true_pi[k, w])
        ps_sgrld.append(float(np.mean(tail > true_val)))

        # SVI: p = P_{q}[pi_{k,w} > true] = 1 - BetaCDF(true)
        a = float(lam_svi_true[k, w])
        b = float(np.sum(lam_svi_true[k, :]) - lam_svi_true[k, w])
        b = max(b, 1e-12)
        ps_svi.append(float(1.0 - beta_dist.cdf(true_val, a, b)))

    if len(ps_sgrld) == 0:
        print("[WARN] No valid histories for SGRLD calibration")
        return

    ps_sgrld = np.sort(np.asarray(ps_sgrld))
    ps_svi = np.sort(np.asarray(ps_svi))
    Dp = ps_sgrld.size
    x_ref = np.arange(1, Dp + 1)
    y_ref = x_ref / (Dp + 1.0)

    plt.figure(figsize=(6.5, 5.2))
    plt.plot(x_ref, ps_sgrld, marker='o', linestyle='-', linewidth=1.2, label='SGRLD empirical')
    plt.plot(x_ref, ps_svi, marker='x', linestyle='-', linewidth=1.2, label='SVI q(pi)')
    plt.plot(x_ref, y_ref, linestyle='-', label='y = x/(D+1)')
    plt.title(f"{CANONICAL_PREFIX} Calibration (rank-uniformity) | SGRLD vs SVI | tracked params (D={Dp})")
    plt.xlabel("rank")
    plt.ylabel("P(π > π*) over tail")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Vectorized rank-uniformity using ALL π entries (K*V) ---
def plot_rank_uniformity_all_pi_sgrld(pi_hist_true, true_pi, *, start_frac=0.5, title="SGRLD rank-uniformity (all π)"):
    """Rank-uniformity using all K*V π entries from SGRLD history in TRUE-topic space.

    pi_hist_true: (T, K, V) array in TRUE-topic order
    true_pi:      (K, V) true topic-word probabilities
    """
    pi_hist_true = np.asarray(pi_hist_true, dtype=float)
    true_pi = np.asarray(true_pi, dtype=float)
    T = pi_hist_true.shape[0]
    start = int(T * start_frac)
    tail = pi_hist_true[start:]
    if tail.size == 0:
        print("[WARN] Empty tail for SGRLD all-π calibration")
        return

    # p_{k,w} = P(π_{k,w} > π*_{k,w}) over tail
    p_mat = np.mean(tail > true_pi[None, :, :], axis=0)  # (K,V)
    ps = np.sort(p_mat.reshape(-1))

    Dp = ps.size
    x_ref = np.arange(1, Dp + 1)
    y_ref = x_ref / (Dp + 1.0)

    plt.figure(figsize=(7.2, 6.0))
    plt.plot(x_ref, ps, marker='.', linestyle='-', linewidth=0.9, label='SGRLD empirical (all π)')
    plt.plot(x_ref, y_ref, linestyle='-', label='y = x/(D+1)')
    plt.title(f"{CANONICAL_PREFIX} Calibration (rank-uniformity) | SGRLD | all π (D={Dp})")
    plt.xlabel("rank")
    plt.ylabel("P(π > π*) over tail")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rank_uniformity_all_pi_compare_sgrld_vs_svi(pi_hist_true, true_pi, lam_svi_true, *, start_frac=0.5):
    """Compare rank-uniformity for ALL π entries: SGRLD tail vs SVI q(pi) Beta marginals."""
    pi_hist_true = np.asarray(pi_hist_true, dtype=float)
    true_pi = np.asarray(true_pi, dtype=float)
    lam_svi_true = np.asarray(lam_svi_true, dtype=float)

    T = pi_hist_true.shape[0]
    start = int(T * start_frac)
    tail = pi_hist_true[start:]
    if tail.size == 0:
        print("[WARN] Empty tail for SGRLD all-π compare")
        return

    # SGRLD p-values for all entries
    p_sgrld = np.mean(tail > true_pi[None, :, :], axis=0).reshape(-1)

    # SVI p-values for all entries: p = 1 - BetaCDF(true)
    a = lam_svi_true
    b = lam_svi_true.sum(axis=1, keepdims=True) - lam_svi_true
    b = np.maximum(b, 1e-12)
    p_svi = (1.0 - beta_dist.cdf(true_pi, a, b)).reshape(-1)

    ps_sgrld = np.sort(p_sgrld)
    ps_svi = np.sort(p_svi)
    Dp = ps_sgrld.size
    x_ref = np.arange(1, Dp + 1)
    y_ref = x_ref / (Dp + 1.0)

    plt.figure(figsize=(7.6, 6.2))
    plt.plot(x_ref, ps_sgrld, marker='.', linestyle='-', linewidth=0.9, label='SGRLD empirical (all π)')
    plt.plot(x_ref, ps_svi, marker='.', linestyle='-', linewidth=0.9, label='SVI q(pi) (all π)')
    plt.plot(x_ref, y_ref, linestyle='-', label='y = x/(D+1)')
    plt.title(f"{CANONICAL_PREFIX} Calibration (rank-uniformity) | SGRLD vs SVI | all π (D={Dp})")
    plt.xlabel("rank")
    plt.ylabel("P(π > π*) over tail")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===================== z posterior under TRUE parameters vs empirical z_state =====================

def _topic_props_from_z(z_doc: np.ndarray, K: int) -> np.ndarray:
    z_doc = np.asarray(z_doc, dtype=int)
    counts = np.bincount(z_doc, minlength=K).astype(float)
    s = counts.sum()
    return counts / max(s, 1.0)


def estimate_true_posterior_z_props_for_doc(
    model: SGRLD_LDA,
    doc_words: List[int],
    true_pi: np.ndarray,
    alpha: float,
    *,
    burn_sweeps: int = 50,
    draws: int = 80,
    thin: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Approximate E[ topic proportions ] under p(z | w, true_pi, alpha) via Gibbs.

    Returns a (K,) vector of average topic proportions over MCMC draws.
    """
    rng = np.random.default_rng(seed)
    K = true_pi.shape[0]

    # random init
    z = rng.integers(0, K, size=len(doc_words)).astype(int)

    # burn-in
    z = model.gibbs_sample_z(doc_words, true_pi, alpha, num_sweeps=burn_sweeps, z_init=z)

    props_acc = np.zeros(K, dtype=float)
    kept = 0
    for t in range(draws):
        z = model.gibbs_sample_z(doc_words, true_pi, alpha, num_sweeps=thin, z_init=z)
        props_acc += _topic_props_from_z(z, K)
        kept += 1

    return props_acc / max(kept, 1)
def estimate_true_z_posterior_tokens(
    model,
    doc_words,
    true_pi,
    alpha,
    burn_sweeps=80,
    draws=150,
    thin=2,
    seed=0,
):
    """
    Approximate token-level posterior p(z_{di}=k | w_d, true_pi, alpha)
    using collapsed Gibbs under TRUE parameters.
    Returns array of shape (L, K).
    """
    rng = np.random.default_rng(seed)
    K = true_pi.shape[0]
    L = len(doc_words)

    z = rng.integers(0, K, size=L)

    # burn-in
    z = model.gibbs_sample_z(
        doc_words, true_pi, alpha,
        num_sweeps=burn_sweeps, z_init=z
    )

    counts = np.zeros((L, K))
    kept = 0
    for _ in range(draws):
        z = model.gibbs_sample_z(
            doc_words, true_pi, alpha,
            num_sweeps=thin, z_init=z
        )
        for i in range(L):
            counts[i, z[i]] += 1
        kept += 1

    return counts / kept


def estimate_empirical_z_posterior_tokens(z_history, K):
    """
    Empirical token-level posterior from SGRLD-Gibbs samples.
    z_history: list of z arrays for the SAME document.
    Returns (L, K).
    """
    L = len(z_history[0])
    counts = np.zeros((L, K))
    for z in z_history:
        for i in range(L):
            counts[i, int(z[i])] += 1
    return counts / len(z_history)


def plot_token_level_z_posterior(emp_probs, true_probs, doc_id):
    """
    Heatmap comparison:
      Left: empirical posterior from SGRLD-Gibbs
      Right: true posterior under p(z | w, pi*, alpha)
    """
    L, K = emp_probs.shape
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

    im0 = axes[0].imshow(emp_probs.T, aspect='auto', origin='lower')
    axes[0].set_title(f"{CANONICAL_PREFIX} Latent-variable posterior | SGRLD | empirical z (doc {doc_id})")
    axes[0].set_xlabel("token index i")
    axes[0].set_ylabel("topic k")

    im1 = axes[1].imshow(true_probs.T, aspect='auto', origin='lower')
    axes[1].set_title(f"{CANONICAL_PREFIX} Latent-variable posterior | TRUE | z (doc {doc_id})")
    axes[1].set_xlabel("token index i")

    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.show()

def plot_true_vs_empirical_z_for_docs(
    model: SGRLD_LDA,
    corpus: List[List[int]],
    z_state: List[np.ndarray],
    true_pi: np.ndarray,
    alpha: float,
    *,
    doc_indices: List[int],
    mapping_est_to_true: np.ndarray = None,
    burn_sweeps: int = 50,
    draws: int = 80,
    thin: int = 2,
):
    """Plot, for each selected doc, TRUE posterior-mean topic proportions vs empirical z_state proportions."""
    K = true_pi.shape[0]
    n = len(doc_indices)
    if n == 0:
        print("[WARN] No doc_indices provided for z comparison")
        return

    fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    x = np.arange(K)
    width = 0.38

    for ax, d in zip(axes, doc_indices):
        doc = corpus[d]
        z_emp = np.asarray(z_state[d], dtype=int)
        # Remap empirical z (estimated-topic labels) into TRUE-topic space if mapping is provided.
        if mapping_est_to_true is not None:
            mapping = np.asarray(mapping_est_to_true, dtype=int)
            if mapping.shape[0] != K:
                raise ValueError("mapping_est_to_true must have shape (K,)")
            z_emp = mapping[np.clip(z_emp, 0, K - 1)]
        emp_props = _topic_props_from_z(z_emp, K)

        true_props = estimate_true_posterior_z_props_for_doc(
            model,
            doc,
            true_pi,
            alpha,
            burn_sweeps=burn_sweeps,
            draws=draws,
            thin=thin,
            seed=123 + int(d),
        )

        ax.bar(x - width / 2, emp_props, width=width, alpha=0.65, label="empirical z_state")
        ax.bar(x + width / 2, true_props, width=width, alpha=0.65, label="TRUE posterior mean (Gibbs)")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("prop")
        ax.set_title(f"{CANONICAL_PREFIX} Latent-variable posterior | Doc {d} (len={len(doc)})")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xlabel("topic k")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"{CANONICAL_PREFIX} Latent-variable posterior | SGRLD vs TRUE | z distribution")
    plt.tight_layout()
    plt.show()


def compare_z_distribution_true_vs_empirical(
    model: SGRLD_LDA,
    corpus: List[List[int]],
    z_state: List[np.ndarray],
    true_pi: np.ndarray,
    alpha: float,
    *,
    num_docs: int = 10,
    seed: int = 0,
    mapping_est_to_true: np.ndarray = None,
    burn_sweeps: int = 50,
    draws: int = 80,
    thin: int = 2,
):
    """Pick ~num_docs documents and compare empirical vs TRUE posterior for z."""
    rng = np.random.default_rng(seed)
    D = len(corpus)
    num_docs = int(min(num_docs, D))
    doc_indices = rng.choice(D, size=num_docs, replace=False).tolist()
    plot_true_vs_empirical_z_for_docs(
        model,
        corpus,
        z_state,
        true_pi,
        alpha,
        doc_indices=doc_indices,
        mapping_est_to_true=mapping_est_to_true,
        burn_sweeps=burn_sweeps,
        draws=draws,
        thin=thin,
    )


# ===================== Diagnostics =====================
# (Optional) SGRLD-only posterior plots per-parameter
# plot_posterior_tracked(theta_history, true_topic_word_dist, tracked_indices, tail_T=2000, algo_name="SGRLD")

if has_true:
    # Rank-uniformity using ALL π entries (SGRLD only)
    # Prefer TRUE-topic-aligned history if available; otherwise fall back to raw pi_history.
    if 'pi_history_matched' in globals():
        _pi_hist_true = pi_history_matched
    else:
        _pi_hist_true = None

    if _pi_hist_true is None:
        print("[WARN] Full pi history not stored; skipping all-π rank-uniformity plots.")
    else:
        plot_rank_uniformity_all_pi_sgrld(_pi_hist_true, true_topic_word_dist, start_frac=0.5)
        # Rank-uniformity comparison: SGRLD vs SVI using ALL π entries
        plot_rank_uniformity_all_pi_compare_sgrld_vs_svi(_pi_hist_true, true_topic_word_dist, lam_svi_true, start_frac=0.5)

    # Compare z distribution under TRUE parameters vs empirical z_state (about 10 docs)
    compare_z_distribution_true_vs_empirical(
        model,
        corpus,
        z_state,
        true_topic_word_dist,
        model.alpha,
        num_docs=10,
        seed=7,
        mapping_est_to_true=(mapping_est_to_true if 'mapping_est_to_true' in globals() else None),
        burn_sweeps=60,
        draws=100,
        thin=2,
    )
    print("\n=== Token-level z posterior comparison (empirical vs true) ===")

    for d in z_history_docs:
        if len(z_history_docs[d]) < 20:
            continue

        # Remap empirical z into TRUE-topic space
        z_hist = []
        for z in z_history_docs[d]:
            if 'mapping_est_to_true' in globals():
                z_hist.append(mapping_est_to_true[z])
            else:
                z_hist.append(z)

        emp_probs = estimate_empirical_z_posterior_tokens(z_hist, K)

        true_probs = estimate_true_z_posterior_tokens(
            model,
            corpus[d],
            true_topic_word_dist,
            model.alpha,
            burn_sweeps=80,
            draws=150,
            thin=2,
            seed=1000 + d,
        )

        plot_token_level_z_posterior(emp_probs, true_probs, doc_id=d)
else:
    print("[INFO] Skipping calibration and z TRUE-vs-empirical plots (no ground truth on real data).")