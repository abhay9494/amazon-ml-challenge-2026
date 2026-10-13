"""Decision rule: turn pair probabilities into per-S1 match lists.

1. Exclusivity: every S2/S3 record goes to at most one S1 - its highest-probability candidate.
2. Threshold: that link is kept only if p >= thr.
3. Expected-F0.5 subset selection per S1 (optional): among the links assigned to an S1, sorted by
   probability, keep the prefix that maximises the expected F0.5 under independent Bernoulli
   outcomes (exact computation with Poisson-binomial distributions; a prefix of size 0 scores 1
   only if no link is true).
"""
import numpy as np
import polars as pl


def assign(pairs: pl.DataFrame, pcol: str = "p") -> pl.DataFrame:
    """Keep each query's best candidate (ties broken by sid)."""
    return (
        pairs.sort(["qid", pcol, "sid"], descending=[False, True, False])
        .group_by("qid", maintain_order=True)
        .head(1)
    )


def _poibin(ps: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf for each row of ps (n_rows x m) -> (n_rows x m+1)."""
    n, m = ps.shape
    pmf = np.zeros((n, m + 1))
    pmf[:, 0] = 1.0
    for j in range(m):
        p = ps[:, j : j + 1]
        pmf[:, 1:] = pmf[:, 1:] * (1 - p) + pmf[:, :-1] * p
        pmf[:, 0] = pmf[:, 0] * (1 - p[:, 0])
    return pmf


def expected_f_prefix(p_sorted: np.ndarray, beta2: float = 0.25) -> np.ndarray:
    """p_sorted: (n x m) probabilities sorted descending (padded with 0).
    Returns (n x m+1) expected F_beta when predicting the top-k, k = 0..m."""
    n, m = p_sorted.shape
    out = np.zeros((n, m + 1))
    for k in range(m + 1):
        a = _poibin(p_sorted[:, :k]) if k > 0 else np.ones((n, 1))
        b = _poibin(p_sorted[:, k:]) if k < m else np.ones((n, 1))
        # E[F] = sum_{tp, r} P(tp) P(r) F(tp, r, k)
        tp = np.arange(a.shape[1])[:, None]
        r = np.arange(b.shape[1])[None, :]
        if k == 0:
            F = (r == 0).astype(float)  # predicting nothing: 1 iff no true match
            F = np.broadcast_to(F, (1, b.shape[1]))
        else:
            denom = beta2 * (tp + r) + k
            F = (1 + beta2) * tp / denom
        out[:, k] = np.einsum("ni,nj,ij->n", a, b, F)
    return out


def select_expected_f(links: pl.DataFrame, pcol: str = "p", max_m: int = 16, beta2: float = 0.25) -> pl.DataFrame:
    """links: (sid, qid, p) after assign+threshold (or a low floor).  Per sid keep the prefix
    (by descending p) with maximal expected F.  Returns the kept links."""
    links = links.sort(["sid", pcol], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("sid").alias("_r"), pl.len().over("sid").alias("_n")
    )
    keep_k = []
    for m, grp in links.filter(pl.col("_r") < max_m).group_by("_n"):
        m = int(m[0]) if isinstance(m, tuple) else int(m)
        mm = min(m, max_m)
        g = grp.sort(["sid", "_r"])
        sids = g["sid"].to_numpy().reshape(-1, mm)[:, 0]
        P = g[pcol].to_numpy().reshape(-1, mm).astype(np.float64)
        ef = expected_f_prefix(P, beta2)
        keep_k.append(pl.DataFrame({"sid": sids, "_k": ef.argmax(1).astype(np.int32)}))
    kk = pl.concat(keep_k)
    return links.join(kk, on="sid").filter(pl.col("_r") < pl.col("_k")).drop("_r", "_n", "_k")


def decide(pairs: pl.DataFrame, thr: float, use_ef: bool = True, floor: float = 0.05, pcol: str = "p") -> pl.DataFrame:
    """pairs: (qid, sid, p).  Returns kept links (qid, sid, p)."""
    best = assign(pairs, pcol)
    if not use_ef:
        return best.filter(pl.col(pcol) >= thr).select("qid", "sid", pcol)
    cand = best.filter(pl.col(pcol) >= floor).select("qid", "sid", pcol)
    kept = select_expected_f(cand, pcol)
    return kept.filter(pl.col(pcol) >= thr).select("qid", "sid", pcol)
