import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.special import psi as digamma, logsumexp  # digamma + logsumexp for SVI
from scipy.stats import beta as beta_dist
from typing import List, Dict
# ===================== Matplotlib LaTeX + font config =====================
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    # --- Global font scaling (paper-ready) ---
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "axes.labelsize": 28,
    "axes.labelweight": "bold",
    "figure.labelsize": 28,
    "figure.labelweight": "bold",
    "legend.fontsize": 28,
    "legend.title_fontsize": 28,
    "figure.titlesize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "font.weight": "bold",

    # --- Legend defaults ---
    "legend.loc": "upper left",
    "legend.frameon": False,
    "legend.framealpha": 0.0,
    "legend.borderaxespad": 0.3,

    # --- Save figures as PDF by default ---
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "axes.unicode_minus": False,
})

# Global default figure size for square/rectangular plots
FIGSIZE_SQUARE = (9, 6)

# ===================== Global suptitle font size =====================
SUPTITLE_SIZE = 52

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
# NOTE: For `text.latex.preamble`, use a single backslash (e.g., r"\usepackage{...}").
# Using "\\usepackage" makes LaTeX see "\\u" and fail.
# ===================== Plotting palette =====================
# Reference / truth: gray
# SVI: blue
# SGRLD: red
# (Optional) SGRLD with S>1 (averaged gradients): dark red
COLOR_REF = "0.55"         # gray
COLOR_TRUE = COLOR_REF
COLOR_UNIFORM = COLOR_REF
COLOR_SVI = "tab:blue"
COLOR_SGRLD = "tab:red"
COLOR_SGRLD_S = "darkred"  # for S>1 variants if used
COLOR_SGRLD_HIST = "lightcoral"


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
    iters: int = 2000,
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


# Settings
D, K, V = 10000, 3, 50
corpus, z, true_topic_word_dist, _ = generate_synthetic_lda_data(D, K, V)
model = SGRLD_LDA(D=D, V=V, K=K)

# --- Run SVI (semi-collapsed) on the same corpus for comparison ---
# NOTE: SVI returns a variational posterior q(pi_k)=Dirichlet(lambda_k).
# We'll compare SVI posterior marginals to SGRLD tail samples.
SVI_ITERS = 1000
svi_out = svi_lda_semi_collapsed(
    corpus,
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
# Track selected (k, w) indices - pick the largest entries
TOP_PER_TOPIC = 3  # how many top words to track per topic
tracked_indices = []
for k in range(K):
    top_w = np.argsort(model.topic_word_dist[k])[::-1][:TOP_PER_TOPIC]
    for w in top_w:
        tracked_indices.append((k, int(w)))

# Always include the global largest parameter $\pi$_{k*, w*}
largest_idx = np.unravel_index(np.argmax(model.topic_word_dist), model.topic_word_dist.shape)
largest_idx = (int(largest_idx[0]), int(largest_idx[1]))
if largest_idx not in tracked_indices:
    tracked_indices.insert(0, largest_idx)

# NOTE: The largest parameter will be highlighted in plots.

theta_history = {idx: [] for idx in tracked_indices}
# Store full $\pi$ history for selecting/plotting true top entries later
pi_history = []  # list of (K, V)
# true_values will be constructed after training, using Hungarian matching.

# Train
num_iterations = 20000
batch_size = 100
step_size = 4.0 * batch_size*10 / D

# Persistent per-document topic assignments for warm-start Gibbs
z_state = [np.random.randint(0, K, size=len(doc)).astype(int) for doc in corpus]

for it in range(num_iterations):
    batch_indices = np.random.choice(D, batch_size, replace=False)

    z_batch = []
    for i in batch_indices:
        doc = corpus[i]
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

    mini_batch_docs = [corpus[i] for i in batch_indices]

    for idx in tracked_indices:
        k, w = idx
        theta_history[idx].append(model.topic_word_dist[k, w])

    pi_history.append(model.topic_word_dist.copy())
    model.update_theta(mini_batch_docs, z_batch, total_docs=D, stepsize=step_size)


#
# --- Hungarian matching to fix label switching before comparison ---
# Use the final estimated topic-word distribution for matching
est_topic_word_final = np.copy(model.topic_word_dist)
K_est = est_topic_word_final.shape[0]
K_true = true_topic_word_dist.shape[0]

# Tail-mean $\pi$ estimate from SGRLD: average of the last 2000 stored $\pi$ matrices
pi_arr = np.asarray(pi_history)
T_tail = 2000
pi_tail_mean_est = pi_arr[-T_tail:].mean(axis=0) if pi_arr.shape[0] >= T_tail else pi_arr.mean(axis=0)

# === Print true parameters and mean of last 2000 iterations ===
print("\n================ TRUE topic-word distribution ================")
print(true_topic_word_dist)

# --- Print largest 10 entries for TRUE topic-word distribution ---
print("\nTop 10 $\pi$ entries from TRUE topic-word distribution:")
flat_true = true_topic_word_dist.flatten()
top_idx_true = np.argsort(flat_true)[-20:][::-1]
for i in top_idx_true:
    k = i // V
    w = i % V
    val = float(true_topic_word_dist[k, w])
    print(f"  TRUE $\pi$(topic={k}, word={w}) = {val:.6f}")

# Compute mean of last 2000 iterations for each tracked (k,w)
mean_last2000 = {}
for idx, series in theta_history.items():
    arr = np.asarray(series)
    tail = arr[-2000:] if arr.shape[0] >= 2000 else arr
    mean_last2000[idx] = float(np.mean(tail))

print("\n================ MEAN of last 2000 iterations (matched later) ================")
for idx, val in mean_last2000.items():
    print(f"param (est-topic={idx[0]}, word={idx[1]}): mean last2000 = {val:.6f}")
print("====================================================================\n")

if K_est == K_true:
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

    # === Select TOP 10 (k,w) by SGRLD tail-mean $\pi$ IN TRUE-topic space, then build their trajectories ===
    # First remap the tail-mean $\pi$ into TRUE-topic space using the mapping
    pi_tail_mean_true = np.zeros_like(pi_tail_mean_est)
    for k_est in range(K_est):
        k_true = int(mapping_est_to_true[k_est])
        pi_tail_mean_true[k_true, :] = pi_tail_mean_est[k_est, :]

    flat_tail = pi_tail_mean_true.flatten()
    top_idx_tail10 = np.argsort(flat_tail)[-10:][::-1]
    tracked_indices = [(int(i // V), int(i % V)) for i in top_idx_tail10[:5]]

    # Build matched $\pi$ history in TRUE-topic space using the fixed mapping
    pi_history_matched = []
    for pi_t in pi_history:
        pi_t_true = np.zeros_like(pi_t)
        for k_est in range(K_est):
            k_true = int(mapping_est_to_true[k_est])
            pi_t_true[k_true, :] = pi_t[k_est, :]
        pi_history_matched.append(pi_t_true)
    pi_history_matched = np.asarray(pi_history_matched)

    # Rebuild theta_history for ONLY these top-10 indices (full trajectories)
    theta_history = {idx: list(pi_history_matched[:, idx[0], idx[1]]) for idx in tracked_indices}

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

    # === Print TOP 20 entries of $\pi$ from SGRLD tail mean (average of last iterations) ===
    # Use the same remapped tail-mean $\pi$ as for plotting
    pi_tail_mean = pi_tail_mean_true

    print("\nTop 20 $\pi$ entries from SGRLD (tail mean):")
    flat_tail = pi_tail_mean.flatten()
    top_idx_tail = np.argsort(flat_tail)[-20:][::-1]
    for idx in top_idx_tail:
        k = idx // V
        w = idx % V
        val = float(pi_tail_mean[k, w])
        print(f"  SGRLD $\pi$(topic={k}, word={w}) = {val:.6f}")
else:
    print("[WARN] K_est != K_true; skipping Hungarian matching.")
    # In this case, keep SGRLD topics in their estimated order
    pi_sgrld_matched = est_topic_word_final
    pi_svi_mean_true = pi_svi_mean
    lam_svi_true = lam_svi

# Now construct true_values using TRUE topic indices
true_values = {idx: true_topic_word_dist[idx[0], idx[1]] for idx in tracked_indices}


import itertools

colors = plt.cm.tab10.colors  # up to 10 distinct colors
color_cycle = itertools.cycle(colors)
color_map = {idx: next(color_cycle) for idx in tracked_indices}

plt.figure(figsize=(12, 6))
for idx in tracked_indices:
    k, w = idx
    is_largest = (idx == largest_idx)
    lw = 2.8 if is_largest else 1.0
    lab_suffix = " (largest)" if is_largest else ""
    c = color_map[idx]

    # SGRLD trajectory
    plt.plot(theta_history[idx], linewidth=lw, color=c,
             label=rf"SGRLD $\pi_{{{k},{w}}}${lab_suffix}")

    # TRUE value reference line
    if idx in true_values:
        tv = float(true_values[idx])
        plt.axhline(tv, color=c, linestyle='--', linewidth=1.2,
                    label=rf"TRUE $\pi_{{{k},{w}}}={tv:.3f}$")

plt.title(f"SGRLD: Selected Topic-Word Probabilities (iterations={num_iterations})")
plt.xlabel("Iteration")
plt.ylabel(r"Probability $\pi_{k,w}$")
plt.ylim(0.0, 0.5)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_and_show()


# ===================== Combined posterior: SGRLD tail vs SVI q(pi) =====================

def plot_posterior_sgrld_vs_svi(
    theta_hist_dict,
    true_pi,
    indices,
    lam_svi_true,
    *,
    tail_T=2000,
    title=r"Posterior comparison: SGRLD vs SVI",
    # Optional: supply another history for an S>1 variant (e.g., S=10) to overlay
    theta_hist_dict_s=None,
    label_s="SGLD-S",
    fixed_xlim=False,
    xlim=(0.1, 0.4),
):
    """Paper-style figure:
    - TRUE (reference) in gray
    - SVI marginal in blue
    - SGRLD tail Gaussian approx in red
    - Optional S>1 variant in dark red
    """
    n = len(indices)
    # Keep at most 4 panels for a 2x2 layout
    indices = list(indices)[:4]
    n = len(indices)
    if n == 0:
        print("[WARN] No indices to plot posterior comparison")
        return

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 6), sharex=False)
    axes = axes.flatten()

    for ax, (k, w) in zip(axes, indices):
        series = np.asarray(theta_hist_dict.get((k, w), []), dtype=float)
        tail = series[-tail_T:] if series.size >= tail_T else series
        if tail.size == 0:
            ax.set_title(f"(topic={k}, word={w}) [no history]")
            continue

        # Gaussian approximation to SGRLD tail (OU-style): N(mean, var)
        m = float(np.mean(tail))
        v = float(np.var(tail, ddof=1)) if tail.size >= 2 else 0.0
        sd = float(np.sqrt(max(v, 1e-18)))

        true_val = float(true_pi[k, w])

        # SVI marginal for pi_{k,w}: Beta(a, b)
        a = float(lam_svi_true[k, w])
        b = float(np.sum(lam_svi_true[k, :]) - lam_svi_true[k, w])
        b = max(b, 1e-12)

        if fixed_xlim:
            xmin, xmax = float(xlim[0]), float(xlim[1])
        else:
            # Fallback: simple non-adaptive window around SGRLD tail mean/var and truth
            if sd > 0:
                xmin = max(0.0, min(true_val, m - 4.0 * sd) - 0.02)
                xmax = min(1.0, max(true_val, m + 4.0 * sd) + 0.02)
            else:
                xmin = max(0.0, min(true_val, m) - 0.02)
                xmax = min(1.0, max(true_val, m) + 0.02)

        xs = np.linspace(max(1e-6, xmin), xmax, 600)

        # SGRLD Gaussian approximation
        if sd > 0:
            gauss = (1.0 / (sd * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - m) / sd) ** 2)
            ax.plot(xs, gauss, linewidth=2.0, linestyle='-', color=COLOR_SGRLD, alpha=0.9, label="SGRLD")
        else:
            ax.axvline(m, linewidth=2.0, linestyle='-', color=COLOR_SGRLD, alpha=0.9, label="SGRLD")

        # Optional S>1 curve (e.g., S=10)
        if theta_hist_dict_s is not None:
            series_s = np.asarray(theta_hist_dict_s.get((k, w), []), dtype=float)
            tail_s = series_s[-tail_T:] if series_s.size >= tail_T else series_s
            if tail_s.size > 0:
                ms = float(np.mean(tail_s))
                vs = float(np.var(tail_s, ddof=1)) if tail_s.size >= 2 else 0.0
                sds = float(np.sqrt(max(vs, 1e-18)))
                if sds > 0:
                    gauss_s = (1.0 / (sds * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - ms) / sds) ** 2)
                    ax.plot(xs, gauss_s, linewidth=2.0, linestyle='-', color=COLOR_SGRLD_S, alpha=0.9, label=label_s)
                else:
                    ax.axvline(ms, linewidth=2.0, linestyle='-', color=COLOR_SGRLD_S, alpha=0.9, label=label_s)

        # SVI marginal for pi_{k,w}: Beta(a, b)
        pdf = beta_dist.pdf(xs, a, b)
        ax.plot(xs, pdf, linewidth=2.0, linestyle='-', color=COLOR_SVI, alpha=0.9, label="SVI")

        # TRUE marker (gray) -- change to dotted style
        ax.axvline(true_val, linestyle='dotted', linewidth=2.0, color=COLOR_TRUE, label=r"True $\pi^\star$")

        ax.set_title(rf"$\pi_{{{k},{w}}}$")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, alpha=0.25)
        # No per-axis legend; will use shared legend

    # Hide any unused axes in the grid
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Shared labels

    # fig.suptitle(title, fontsize=SUPTITLE_SIZE, fontweight="bold")

    # Single shared legend (collect from plotted axes)
    handles, labels = [], []
    for ax in axes[:n]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
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

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_show(fig)
# ===================== Posterior plots (SGRLD tail samples) =====================

def plot_posterior_tracked(theta_hist_dict, true_topic_word_dist, indices, *, tail_T=2000, algo_name="SGRLD"):
    """Plot approximate posterior (tail histogram) for each tracked $\pi_{k,w}$.

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

        true_val = float(true_topic_word_dist[k, w])

        _plt.figure(figsize=(6.5, 4.2))
        _plt.hist(tail, bins=40, density=True, alpha=0.55, color=COLOR_SGRLD_HIST, label=f"{algo_name} tail samples")

        # Gaussian approximation from tail mean/var
        xs = _np.linspace(max(0.0, m - 4 * sd), min(1.0, m + 4 * sd), 400) if sd > 0 else _np.linspace(0.0, 1.0, 400)
        if sd > 0:
            gauss = (1.0 / (sd * _np.sqrt(2 * _np.pi))) * _np.exp(-0.5 * ((xs - m) / sd) ** 2)
            _plt.plot(xs, gauss, linewidth=2.0, color=COLOR_SGRLD, label="Gaussian fit (mean/var)")

        _plt.axvline(true_val, linestyle='--', linewidth=2.0, color=COLOR_TRUE, label=rf"TRUE $\pi_{{{k},{w}}}={true_val:.3f}$")

        _plt.title(rf"Posterior approx for $\pi_{{{k},{w}}}$ - {algo_name}")
        _plt.xlabel(r"$\pi_{k,w}$")
        _plt.xlim(0.0, 0.5)
        _plt.grid(True, alpha=0.3)
        _plt.legend()
        _plt.tight_layout()
        save_and_show(_plt.gcf())





def plot_rank_uniformity_pair(theta_hist_dict, true_topic_word_dist, k, w1, w2, algo_name="Algo", start_frac=0.5):
    """
    Rank-uniformity calibration for two parameters $\theta_{k,w1}$, $\theta_{k,w2}$ using an iterate history.
    For each parameter j in {w1, w2}, compute p_j = mean($\theta_j^{(t)} > \theta_j^*$), t over the tail [start_frac, 1].
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
        true_val = float(true_topic_word_dist[idx[0], idx[1]])
        p = float(np.mean(tail > true_val))
        ps.append(p)
        labels.append(rf"$\theta_{{{idx[0]},{idx[1]}}}$")

    if len(ps) == 0:
        print(f"[WARN] No history found for k={k}, w1={w1}, w2={w2} in {algo_name}")
        return

    ps_sorted = np.sort(np.asarray(ps))
    _plt.figure(figsize=(5, 4))
    _plt.plot(x_ref, ps_sorted, marker='o', linestyle='-', linewidth=1.2, color=COLOR_SGRLD, label='Empirical')
    _plt.plot(x_ref, y_ref, linestyle='-', color=COLOR_REF, label='y = x/(D+1)')
    _plt.title(f"Rank-uniformity (k={k}, w1={w1}, w2={w2}) - {algo_name}")
    _plt.xlabel("rank")
    _plt.ylabel(r"$P(\theta > \theta^\star)$ over tail")
    _plt.ylim(-0.05, 1.05)
    _plt.grid(True, alpha=0.3)
    _plt.legend()
    _plt.tight_layout()
    save_and_show(_plt.gcf())




# --- Helper: Rank-uniformity calibration plot for all tracked parameters ---
def plot_rank_uniformity_all(theta_hist_dict, true_topic_word_dist, indices, algo_name="Algo", start_frac=0.5):
    """
    Rank-uniformity using ALL tracked parameters in `indices`.
    For each (k,w) in indices, compute p_{k,w} = mean($\theta_{k,w}^{(t)} > \theta^*_{k,w}$) over the tail [start_frac, 1].
    Plot sorted {p} against the reference line y = x/(D+1), where D = len(indices).
    """
    import matplotlib.pyplot as _plt
    if not indices:
        print(f"[WARN] No indices provided for rank-uniformity ({algo_name})")
        return
    ps = []
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
    _plt.plot(x_ref, ps, marker='o', linestyle='-', linewidth=1.2, color=COLOR_SGRLD, label='Empirical')
    _plt.plot(x_ref, y_ref, linestyle='-', color=COLOR_REF, label='y = x/(D+1)')
    _plt.title(f"Rank-uniformity of tracked params - {algo_name} (D={D})")
    _plt.xlabel("rank")
    _plt.ylabel(r"$P(\theta > \theta^\star)$ over tail")
    _plt.ylim(-0.05, 1.05)
    _plt.grid(True, alpha=0.3)
    _plt.legend()
    _plt.tight_layout()
    save_and_show(_plt.gcf())


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

    plt.figure(figsize=(9, 6))
    plt.plot(x_ref, y_ref, linestyle='-', linewidth=2.4, alpha=0.85,
            color=COLOR_REF, label='Uniform reference')
    plt.plot(x_ref, ps_sgrld, marker='o', linestyle='-', linewidth=1.4,
            color=COLOR_SGRLD, label='SGRLD')
    plt.plot(x_ref, ps_svi, marker='x', linestyle='-', linewidth=1.4,
            color=COLOR_SVI, label='SVI')
    plt.title(rf"SGRLD vs SVI rank-uniformity calibration for $\pi$ (D={Dp})")
    plt.xlabel("rank")
    plt.ylabel(r"$P(\pi > \pi^\star)$ over tail")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_and_show()


# --- Vectorized rank-uniformity using ALL $\pi$ entries (K*V) ---
def plot_rank_uniformity_all_pi_sgrld(pi_hist_true, true_pi, *, start_frac=0.5, title=r"SGRLD rank-uniformity (all $\pi$)"):
    """QQ-style rank-uniformity: plot sorted p-values vs Uniform(0,1) quantiles."""
    plt.figure(figsize=FIGSIZE_SQUARE)
    pi_hist_true = np.asarray(pi_hist_true, dtype=float)
    true_pi = np.asarray(true_pi, dtype=float)
    T = pi_hist_true.shape[0]
    start = int(T * start_frac)
    tail = pi_hist_true[start:]
    if tail.size == 0:
        print("[WARN] Empty tail for SGRLD all-$\pi$ calibration")
        return

    # p_{k,w} = P($\pi$_{k,w} > $\pi$*_{k,w}) over tail
    p_mat = np.mean(tail > true_pi[None, :, :], axis=0)  # (K,V)
    ps = np.sort(p_mat.reshape(-1))

    Dp = ps.size
    q = (np.arange(1, Dp + 1) - 0.5) / Dp  # Uniform quantiles

    # Plot Uniform reference diagonal FIRST
    plt.plot(q, q, linestyle='-', linewidth=2.0, alpha=0.70, color=COLOR_REF, label='Uniform reference')
    # Empirical SGRLD curve
    plt.plot(
        q, ps,
        linestyle='-', marker='o', markersize=3.5, linewidth=1.2,
        alpha=0.90, color=COLOR_SGRLD, label='SGRLD empirical (all $\\pi$)'
    )

    plt.title(f"{title} (D={Dp})")
    plt.xlabel("Uniform quantile")
    plt.ylabel(r"Empirical $P(\pi > \pi^\star)$ (sorted)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    save_and_show()

def plot_rank_uniformity_all_pi_compare_sgrld_vs_svi(pi_hist_true, true_pi, lam_svi_true, *, start_frac=0.5):
    """QQ-style compare: SGRLD tail vs SVI q(pi) Beta marginals for ALL $\pi$ entries."""
    plt.figure(figsize=FIGSIZE_SQUARE)
    pi_hist_true = np.asarray(pi_hist_true, dtype=float)
    true_pi = np.asarray(true_pi, dtype=float)
    lam_svi_true = np.asarray(lam_svi_true, dtype=float)

    T = pi_hist_true.shape[0]
    start = int(T * start_frac)
    tail = pi_hist_true[start:]
    if tail.size == 0:
        print("[WARN] Empty tail for SGRLD all-$\pi$ compare")
        return

    # SGRLD p-values for all entries
    p_sgrld = np.mean(tail > true_pi[None, :, :], axis=0).reshape(-1)

    # SVI p-values: p = 1 - BetaCDF(true)
    a = lam_svi_true
    b = lam_svi_true.sum(axis=1, keepdims=True) - lam_svi_true
    b = np.maximum(b, 1e-12)
    p_svi = (1.0 - beta_dist.cdf(true_pi, a, b)).reshape(-1)

    ps_sgrld = np.sort(p_sgrld)
    ps_svi = np.sort(p_svi)
    Dp = ps_sgrld.size
    q = (np.arange(1, Dp + 1) - 0.5) / Dp

    # Uniform reference FIRST
    plt.plot(q, q, linestyle='-', linewidth=2.0, alpha=0.70, color=COLOR_REF, label='Uniform reference')
    # SGRLD empirical
    plt.plot(
        q, ps_sgrld,
        linestyle='-', marker='o', markersize=3.5, linewidth=1.2,
        alpha=0.90, color=COLOR_SGRLD, label='SGRLD'
    )
    # SVI empirical
    plt.plot(
        q, ps_svi,
        linestyle='-', marker='x', markersize=3.5, linewidth=1.2,
        alpha=0.95, color=COLOR_SVI, label='SVI'
    )

    plt.xlabel("Uniform quantile")
    plt.ylabel(r"Empirical $P(\pi > \pi^\star)$ (sorted)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    save_and_show()

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
    axes[0].set_title(f"Empirical z posterior (doc {doc_id})")
    axes[0].set_xlabel("token index i")
    axes[0].set_ylabel("topic k")

    im1 = axes[1].imshow(true_probs.T, aspect='auto', origin='lower')
    axes[1].set_title(f"True z posterior (doc {doc_id})")
    axes[1].set_xlabel("token index i")

    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    save_and_show(fig)

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

        ax.bar(x - width / 2, emp_props, width=width, alpha=0.80, color=COLOR_SGRLD, label="SGRLD")
        ax.bar(x + width / 2, true_props, width=width, alpha=0.80, color=COLOR_TRUE, label=r"$\theta^\star$")
        ax.set_ylabel("mass")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("fraction")
        ax.set_title(f"Doc {d} (len={len(doc)})")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xlabel("topic k")
    # Single shared legend outside the axes
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    # fig.suptitle(r"$z$ distribution: SGRLD vs reference ($\theta^\star$)", fontsize=SUPTITLE_SIZE, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 0.85, 0.92])
    save_and_show(fig)


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


# ===================== SGRLD with S>1 Gibbs averaging (separate variant) =====================

def run_sgrld_gibbs_averaged(
    corpus: List[List[int]],
    true_topic_word_dist: np.ndarray,
    *,
    D: int,
    V: int,
    K: int,
    alpha: float,
    beta: float,
    num_iterations: int,
    batch_size: int,
    step_size: float,
    gibbs_S: int = 10,
    num_sweeps: int = 2,
    seed: int = 202,
):
    """Run a second SGRLD variant that averages sufficient statistics over S Gibbs draws.

    This is intentionally kept separate from the main (S=1) loop.
    - No z tracking / diagnostics here (per your request).
    - Returns a TRUE-topic-aligned pi_history (T,K,V) and a matched theta_history_s
      for the caller-provided tracked indices.
    """
    np.random.seed(seed)

    model_s = SGRLD_LDA(D=D, V=V, K=K, alpha=alpha, beta=beta, seed=seed)

    # Initialize theta similarly
    model_s.theta = np.random.gamma(shape=model_s.beta, scale=1.0, size=(K, V)).astype(float)
    model_s.theta += 1e-6
    model_s.topic_word_dist = model_s._normalize_theta(model_s.theta)

    # warm-start z state per document (not tracked)
    z_state_s = [np.random.randint(0, K, size=len(doc)).astype(int) for doc in corpus]

    pi_history_s = []

    S = int(max(1, gibbs_S))

    for it in range(num_iterations):
        batch_indices = np.random.choice(D, batch_size, replace=False)
        mini_batch_docs = [corpus[i] for i in batch_indices]

        # Precompute current normalized pi
        pi = model_s._normalize_theta(model_s.theta)
        grad_theta = np.zeros_like(model_s.theta)

        # For each doc: average counts over S Gibbs draws, then form the same gradient contribution
        for i, doc in zip(batch_indices, mini_batch_docs):
            if len(doc) == 0:
                continue

            # Starting point for Gibbs within this doc
            z_curr = z_state_s[i]

            counts_avg = np.zeros((K, V), dtype=float)
            for _ in range(S):
                z_curr = model_s.gibbs_sample_z(
                    doc,
                    pi,
                    model_s.alpha,
                    num_sweeps=num_sweeps,
                    z_init=z_curr,
                )
                # accumulate counts for this draw
                for word, topic in zip(doc, z_curr):
                    counts_avg[int(topic), int(word)] += 1.0

            counts_avg /= float(S)
            z_state_s[i] = z_curr  # keep warm start (not tracked)

            n_k = counts_avg.sum(axis=1, keepdims=True)
            grad_theta += (counts_avg - pi * n_k)

        # Average across docs (match your S=1 code structure)
        grad_theta *= 1.0 / max(1, len(mini_batch_docs))

        # Same drift/noise form as your original update_theta()
        drift = grad_theta + (model_s.beta - model_s.theta) / model_s.D
        noise = np.random.randn(*model_s.theta.shape)
        eps = step_size
        model_s.theta += 0.5 * eps * drift + np.sqrt(eps * np.maximum(model_s.theta, 1e-12) / model_s.D) * noise
        model_s.theta = np.abs(model_s.theta)
        model_s.topic_word_dist = model_s._normalize_theta(model_s.theta)

        pi_history_s.append(model_s.topic_word_dist.copy())

    pi_history_s = np.asarray(pi_history_s)

    # Match topics for this S>1 run into TRUE-topic space
    est_final = np.copy(model_s.topic_word_dist)
    mapping_s_to_true = match_topics(true_topic_word_dist, est_final)

    pi_history_s_matched = []
    for pi_t in pi_history_s:
        pi_t_true = np.zeros_like(pi_t)
        for k_est in range(K):
            k_true = int(mapping_s_to_true[k_est])
            pi_t_true[k_true, :] = pi_t[k_est, :]
        pi_history_s_matched.append(pi_t_true)
    pi_history_s_matched = np.asarray(pi_history_s_matched)

    return {
        "model": model_s,
        "pi_history_matched": pi_history_s_matched,
        "mapping_est_to_true": mapping_s_to_true,
    }


# ===================== Diagnostics =====================
# (Optional) SGRLD-only posterior plots per-parameter
# plot_posterior_tracked(theta_history, true_topic_word_dist, tracked_indices, tail_T=2000, algo_name="SGRLD")

# Rank-uniformity using ALL $\pi$ entries (SGRLD only)
# Prefer TRUE-topic-aligned history if available; otherwise fall back to raw pi_history.
if 'pi_history_matched' in globals():
    _pi_hist_true = pi_history_matched
else:
    _pi_hist_true = np.asarray(pi_history)

plot_rank_uniformity_all_pi_sgrld(_pi_hist_true, true_topic_word_dist, start_frac=0.5)


# Rank-uniformity comparison: SGRLD vs SVI using ALL $\pi$ entries
plot_rank_uniformity_all_pi_compare_sgrld_vs_svi(_pi_hist_true, true_topic_word_dist, lam_svi_true, start_frac=0.5)
# ===================== Posterior comparison: SGRLD vs SVI =====================

plot_posterior_sgrld_vs_svi(
    theta_history,
    true_topic_word_dist,
    tracked_indices,
    lam_svi_true,
    tail_T=2000,
    title="Posterior comparison: SGRLD--Gibbs vs SVI",
)
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


# ===================== Run SGRLD with S>1 AFTER all existing plots =====================

S_AVG = 5
print(f"\n=== Running separate SGRLD variant with Gibbs averaging: S={S_AVG} (no z-tracking) ===")

sgrld_s_out = run_sgrld_gibbs_averaged(
    corpus=corpus,
    true_topic_word_dist=true_topic_word_dist,
    D=D,
    V=V,
    K=K,
    alpha=model.alpha,
    beta=model.beta,
    num_iterations=num_iterations,
    batch_size=batch_size,
    step_size=step_size,
    gibbs_S=S_AVG,
    num_sweeps=2,
    seed=909,
)

pi_history_s_matched = sgrld_s_out["pi_history_matched"]  # (T,K,V) in TRUE-topic space

# Force exactly the four requested (k,w) pairs for posterior comparison
posterior_indices = [(1, 28), (1, 10), (0, 17), (0, 35)]

# Build theta_history for the requested indices (already in TRUE-topic space)
theta_history_s = {idx: list(pi_history_s_matched[:, idx[0], idx[1]]) for idx in posterior_indices}

# Posterior comparison overlay (like GMM): TRUE gray, SVI blue, SGRLD red, SGRLD-S dark red
plot_posterior_sgrld_vs_svi(
    theta_history,
    true_topic_word_dist,
    posterior_indices,
    lam_svi_true,
    tail_T=2000,
    title=f"Posterior comparison: SGLD vs SVI",
    theta_hist_dict_s=theta_history_s,
    label_s=f"SGRLD (S={S_AVG})",
)