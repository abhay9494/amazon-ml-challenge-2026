"""Stage 2: collective (stacked) matcher.

The stage-1 matcher judges every (S2/S3 record, S1 entity) pair in isolation.  Stage 2 adds what
the OTHER records say about the same S1 entity: records of one business from one source share the
source's own spelling of the name and address (e.g. both S3 records say "922 Cushing" while S1 says
"924"), whereas a sibling decoy differs from the true records of its S1 (other house number, extra
name word).  For each candidate pair (q, s) we look at the "partners": the other records whose best
stage-1 candidate is s with p >= PARTNER_P, and describe how q relates to them.

Features (all per pair; the stage-1 probability itself is included):
  k_n, k_n_same_src      number of partners (all / same source as q)
  k_maxp                 best partner probability
  k_name_max, k_addr_max best name / address similarity between q and any partner
  k_name_bp, k_addr_bp   similarity to the most probable partner
  k_h_q_in_partners      share of partners whose first house number equals q's
  k_h_s_in_partners      share of partners whose first house number equals the S1 record's
  k_exact_dup            a partner has exactly q's normalised name and address
  p1, p1_rank, p1_gap    stage-1 probability, its rank among q's candidates, gap to the best

Training uses the OUT-OF-FOLD stage-1 probabilities (folds are by S1 entity, so all partners of a
pair were scored by a model that never saw that entity); test uses the fold-averaged stage-1
probabilities.  The model is again cross-fitted in 2 folds by S1 entity.
"""
import os

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from common import cache, log

PARTNER_P = 0.5
REC = ["rid", "src", "n_core_c", "a_full_c", "a_nums"]


def collective_features(pred: pl.DataFrame, nrm: pl.DataFrame) -> pl.DataFrame:
    """pred: (qid, sid, p) for all candidate pairs.  nrm: normalised records (REC columns)."""
    pred = pred.sort(["qid", "sid"]).with_columns(
        pl.col("p").rank("ordinal", descending=True).over("qid").cast(pl.Int16).alias("p1_rank"),
        (pl.col("p") - pl.col("p").max().over("qid")).alias("p1_gap"),
    )
    best = pred.filter(pl.col("p1_rank") == 1).filter(pl.col("p") >= PARTNER_P).select(
        pl.col("qid").alias("bid"), "sid", pl.col("p").alias("bp"))
    rec = nrm.select(REC)
    qinfo = rec.rename({"rid": "qid", "src": "q_src", "n_core_c": "q_n", "a_full_c": "q_a", "a_nums": "q_h"})
    binfo = rec.rename({"rid": "bid", "src": "b_src", "n_core_c": "b_n", "a_full_c": "b_a", "a_nums": "b_h"})
    sinfo = rec.select(pl.col("rid").alias("sid"), pl.col("a_nums").alias("s_h"))
    # pairs x partners (excluding the pair's own query)
    x = pred.select("qid", "sid").join(best, on="sid").filter(pl.col("bid") != pl.col("qid"))
    x = x.join(qinfo, on="qid", how="left").join(binfo, on="bid", how="left").join(sinfo, on="sid", how="left")
    x = x.with_columns(
        pl.Series("nm", process.cpdist(x["q_n"].to_list(), x["b_n"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
        pl.Series("ad", process.cpdist(x["q_a"].to_list(), x["b_a"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
        # empty strings mean "no house number" (missing evidence), never agreement
        pl.col("q_h").str.split(" ").list.first().replace("", None).alias("qh1"),
        pl.col("b_h").str.split(" ").list.first().replace("", None).alias("bh1"),
        pl.col("s_h").str.split(" ").list.first().replace("", None).alias("sh1"),
    )
    g = x.group_by("qid", "sid").agg(
        pl.len().alias("k_n"),
        (pl.col("q_src") == pl.col("b_src")).sum().alias("k_n_same_src"),
        pl.col("bp").max().alias("k_maxp"),
        pl.col("nm").max().alias("k_name_max"),
        pl.col("ad").max().alias("k_addr_max"),
        pl.col("nm").sort_by("bp", descending=True).first().alias("k_name_bp"),
        pl.col("ad").sort_by("bp", descending=True).first().alias("k_addr_bp"),
        # house-number agreement among partners that HAVE a number (null when nobody has one)
        (pl.col("qh1") == pl.col("bh1")).drop_nulls().mean().alias("k_h_q_in_partners"),
        (pl.col("sh1") == pl.col("bh1")).drop_nulls().mean().alias("k_h_s_in_partners"),
        (pl.col("qh1").is_not_null() & pl.col("bh1").is_not_null() & (pl.col("qh1") != pl.col("bh1"))).sum().alias("k_h_conflicts"),
        ((pl.col("q_n") == pl.col("b_n")) & (pl.col("q_a") == pl.col("b_a")) & (pl.col("q_a") != "")).any().alias("k_exact_dup"),
        ((pl.col("q_n") == pl.col("b_n")) & ((pl.col("q_a") == "") | (pl.col("b_a") == ""))).any().alias("k_name_only_dup"),
    )
    out = pred.join(g, on=["qid", "sid"], how="left").with_columns(
        pl.col("k_n").fill_null(0), pl.col("k_n_same_src").fill_null(0),
        *[pl.col(c).fill_null(-1.0) for c in ("k_maxp", "k_name_max", "k_addr_max", "k_name_bp", "k_addr_bp",
                                               "k_h_q_in_partners", "k_h_s_in_partners")],
        pl.col("k_exact_dup").fill_null(False).cast(pl.Int8),
        pl.col("k_name_only_dup").fill_null(False).cast(pl.Int8),
        pl.col("k_h_conflicts").fill_null(0),
    ).rename({"p": "p1"})
    return out.with_columns(pl.col(pl.Float64).cast(pl.Float32))


K_FEATS = ["p1", "p1_rank", "p1_gap", "k_n", "k_n_same_src", "k_maxp", "k_name_max", "k_addr_max",
           "k_name_bp", "k_addr_bp", "k_h_q_in_partners", "k_h_s_in_partners", "k_h_conflicts",
           "k_exact_dup", "k_name_only_dup"]
