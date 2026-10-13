"""Stage 2 of candidate generation: learned pruning of the retrieved pairs.

A small LightGBM model scores every retrieved (query, S1) pair from the blocking scores and the
cheap similarities, plus within-query context (rank, gap to the best).  For every S2/S3 query we
keep only its best S1 and near-ties (score within PRUNE_RATIO of the best, at most PRUNE_K).  Each
S2/S3 record belongs to at most one S1, so this keeps the candidate set close to the number of
S2/S3 records while preserving recall.  The surviving pairs are candidate_pairs.tsv.
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from common import log

PRUNE_K = 3
PRUNE_RATIO = 0.2    # keep runner-ups whose score >= ratio * best score of the query
PRUNE_MIN = 0.01     # (legacy fixed rule)
CONFIDENT = 0.95     # adaptive rule: best candidate at least this confident -> keep top-2 near-ties
UNCERTAIN_K = 8      # otherwise keep up to this many alternatives ...
UNCERTAIN_MIN = 0.002  # ... with at least this pruner score

CHEAP = ["sn", "sa", "st", "kn", "ka", "c_ntset", "c_nsort", "c_nctset", "c_atset", "c_asort", "c_nsr"]


def context(df: pl.DataFrame) -> pl.DataFrame:
    """Within-query context features for the retrieved pairs.

    The cheap similarities are stored as UInt8; they are cast to a signed type BEFORE any
    subtraction (80 - 100 would otherwise wrap around to 236).  Rows are sorted by (qid, sid) so
    that rank ties are broken deterministically."""
    sims = ["c_ntset", "c_nsort", "c_nctset", "c_atset", "c_asort", "c_nsr"]
    df = df.sort(["qid", "sid"]).with_columns([pl.col(c).cast(pl.Int16) for c in sims])
    df = df.with_columns(
        (pl.col("c_ntset") + pl.col("c_atset")).alias("c_sum"),
        pl.max_horizontal(pl.col("c_ntset"), pl.col("c_nsr"), pl.col("c_nctset")).alias("c_nbest"),
    )
    over = "qid"
    return df.with_columns(
        pl.len().over(over).alias("x_ncand"),
        (pl.col("st") / pl.col("st").max().over(over)).alias("x_st_rel"),
        pl.col("st").rank("ordinal", descending=True).over(over).alias("x_st_rank"),
        (pl.col("c_sum") - pl.col("c_sum").max().over(over)).alias("x_csum_gap"),
        pl.col("c_sum").rank("ordinal", descending=True).over(over).alias("x_csum_rank"),
        (pl.col("c_nbest") - pl.col("c_nbest").max().over(over)).alias("x_nbest_gap"),
        (pl.col("c_atset") - pl.col("c_atset").max().over(over)).alias("x_atset_gap"),
    )


FEATS = CHEAP + ["c_sum", "c_nbest", "x_ncand", "x_st_rel", "x_st_rank", "x_csum_gap", "x_csum_rank", "x_nbest_gap", "x_atset_gap"]


def train_pruner(df: pl.DataFrame, sample_frac: float = 0.25, seed: int = 0) -> lgb.Booster:
    """df: retrieved pairs with context() features and label y."""
    qids = df.select("qid").unique().sort("qid").sample(fraction=sample_frac, seed=seed)  # deterministic
    tr = df.join(qids, on="qid", how="semi")
    X = tr.select(FEATS).to_numpy().astype(np.float32)
    y = tr["y"].to_numpy()
    log(f"pruner: training on {len(y):,} pairs ({y.mean():.3f} positive)")
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=0)
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=300)


def score(df: pl.DataFrame, model: lgb.Booster, chunk: int = 5_000_000) -> np.ndarray:
    out = []
    for i in range(0, df.height, chunk):
        out.append(model.predict(df.slice(i, chunk).select(FEATS).to_numpy().astype(np.float32)))
    return np.concatenate(out).astype(np.float32)


def prune(df: pl.DataFrame, k: int = PRUNE_K, ratio: float = PRUNE_RATIO, pmin: float = PRUNE_MIN) -> pl.DataFrame:
    """df has column p_prune.  ADAPTIVE retention (chosen on the clean, cross-fitted pruner scores,
    see README): when the query's best candidate is confident (p >= CONFIDENT) keep it plus
    near-ties (top-2, >= ratio x best); when it is uncertain keep up to UNCERTAIN_K alternatives
    with p >= UNCERTAIN_MIN.  On the clean v5 pruner scores: oracle macro F0.5 0.99492 at 4.97
    candidates per S1 (0.9 / 6 / 0.003: 0.99468 at 4.82; former fixed top-3 rule: 0.99368 at 4.43)."""
    df = df.sort(["qid", "sid"]).with_columns(
        pl.col("p_prune").rank("ordinal", descending=True).over("qid").alias("pr_rank"),
        pl.col("p_prune").max().over("qid").alias("pr_best"),
    )
    confident = pl.col("pr_best") >= CONFIDENT
    keep_conf = confident & (pl.col("pr_rank") <= 2) & ((pl.col("pr_rank") == 1) | (pl.col("p_prune") >= ratio * pl.col("pr_best")))
    keep_unc = ~confident & (pl.col("pr_rank") <= UNCERTAIN_K) & (pl.col("p_prune") >= UNCERTAIN_MIN)
    return df.filter(keep_conf | keep_unc)
