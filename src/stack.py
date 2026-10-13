"""Stage-2 re-scorer over ALL candidate pairs (v9).
Inputs per pair: ensemble member probabilities (a, b, c), mean and spread; query competition (rank,
gap to the best other candidate, candidate count, probability mass); S1 competition (candidate
count, best / summed probability of the other queries, first-pass links excluding the pair);
18 raw-text flags (rawflags_all.py); 27 main-model similarity features.
Train: cross-fitted on the entity-family folds (OOF stacked p) and evaluated with the released
expected-F0.5 rule; test: mean of the two fold models.
usage: python stack.py train | test"""
import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import cache, log
from decide import decide

MAIN = ["n_tset", "n_ratio", "n_best_tset", "n_full_ratio", "nc_tset", "a_tset", "a_ratio", "ac_tset", "h_first_eq", "h_affix", "h_jac",
        "h_first_logdiff", "l_conflict", "xt_min", "xt_n_decoy", "n_idf_cov_s", "n_idf_cov_q", "a_idf_cov_s", "q_addr_empty",
        "n_q_oov_frac", "q_name_df", "s_name_df", "p_prune", "pr_margin", "q_src3", "n_me_s_min", "o_n_best_tset"]
RAW = ["dq", "ds", "adq", "ads", "ins", "dele", "sub", "trans", "roman", "legal_add", "legal_drop", "legal_diff",
       "accent_add", "leet", "dblspace", "caps", "punct_diff", "order"]
CTX = ["a", "b", "c", "p", "spread", "rank", "gap", "ncand", "q_sump", "s_ncand", "s_other_maxp", "s_other_sump", "s_links_other", "linked"]
FEATS = CTX + RAW + MAIN
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=0)
ROUNDS = 500


def build(split):
    if split == "train":
        o = pl.read_parquet(cache("cv_oof1.parquet")).select("qid", "sid", pl.col("p").alias("a"))
        for v in "bc":
            o = o.join(pl.read_parquet(cache(f"cv_oof1_{v}.parquet")).rename({"p": v}), on=["qid", "sid"])
    else:
        o = pl.read_parquet(cache("v7c_test_pred.parquet")).select("qid", "sid", "a", "b", "c")
    n0 = o.height
    o = o.with_columns(pl.mean_horizontal("a", "b", "c").alias("p"),
                       (pl.max_horizontal("a", "b", "c") - pl.min_horizontal("a", "b", "c")).alias("spread"))
    links = decide(o.select("qid", "sid", "p"), thr=0.7, use_ef=True).select("qid", "sid", pl.lit(1, pl.Int8).alias("linked"))
    o = o.join(links, on=["qid", "sid"], how="left").with_columns(pl.col("linked").fill_null(0))
    o = o.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("qid").cast(pl.Int16).alias("rank"),
        pl.len().over("qid").cast(pl.Int16).alias("ncand"),
        pl.col("p").sum().over("qid").alias("q_sump"),
        pl.len().over("sid").cast(pl.Int32).alias("s_ncand"),
        (pl.col("p").sum().over("sid") - pl.col("p")).alias("s_other_sump"),
        (pl.col("linked").sum().over("sid") - pl.col("linked")).alias("s_links_other"),
    )
    # best other candidate of the query / best other query of the S1 (top-2 trick)
    t2q = o.group_by("qid").agg(pl.col("p").top_k(2).sort(descending=True).alias("_t"))
    o = o.join(t2q, on="qid").with_columns(
        (pl.col("p") - pl.when(pl.col("p") >= pl.col("_t").list.get(0)).then(pl.col("_t").list.get(1, null_on_oob=True))
         .otherwise(pl.col("_t").list.get(0)).fill_null(0)).alias("gap")).drop("_t")
    t2s = o.group_by("sid").agg(pl.col("p").top_k(2).sort(descending=True).alias("_t"))
    o = o.join(t2s, on="sid").with_columns(
        pl.when(pl.col("p") >= pl.col("_t").list.get(0)).then(pl.col("_t").list.get(1, null_on_oob=True))
        .otherwise(pl.col("_t").list.get(0)).fill_null(0).alias("s_other_maxp")).drop("_t")
    raw = pl.read_parquet(cache(f"rawflags_{split}.parquet"))
    o = o.join(raw, on=["qid", "sid"], how="left")
    parts = sorted(glob.glob(cache(f"{split}_feats/part_*.parquet")))
    feats = pl.concat([pl.read_parquet(p, columns=["qid", "sid"] + MAIN) for p in parts])
    o = o.join(feats, on=["qid", "sid"], how="left")
    assert o.height == n0, (o.height, n0)
    assert o.select("qid", "sid").unique().height == n0
    return o.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def X(df):
    return df.select(FEATS).to_numpy().astype(np.float32)


if __name__ == "__main__":
    split = sys.argv[1]
    if split == "train":
        import cv
        import pipeline as P
        from folds import query_folds
        o = build("train")
        truth = P.truth_pairs(P.id_maps("train")).with_columns(pl.lit(1, pl.Int8).alias("y"))
        o = o.join(truth, on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(0)).join(query_folds().select("qid", "fold"), on="qid", how="left")
        log(f"train table {o.shape}")
        oof = []
        for k in (0, 1):
            tr = o.filter(pl.col("fold") != k)
            m = lgb.train(dict(PARAMS, seed=100 + k), lgb.Dataset(X(tr), tr["y"].to_numpy(), feature_name=FEATS, free_raw_data=True), ROUNDS)
            del tr
            m.save_model(cache(f"stack_fold{k}.txt"))
            te = o.filter(pl.col("fold") == k)
            oof.append(te.select("qid", "sid").with_columns(pl.Series("s", m.predict(X(te)).astype(np.float32))))
            log(f"stack fold {k} done")
            imp = sorted(zip(FEATS, m.feature_importance("gain")), key=lambda t: -t[1])[:15]
            log("top: " + ", ".join(f"{a}:{b:.0f}" for a, b in imp))
        oof = pl.concat(oof)
        oof.write_parquet(cache("stack_oof.parquet"))
        base = cv.per_entity(decide(o.select("qid", "sid", "p"), thr=0.7, use_ef=True)).sort("sid")
        log(f"v7c base {base['f'].mean():.6f}")
        res = {}
        for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
            d = cv.per_entity(decide(oof.select("qid", "sid", pl.col("s").alias("p")), thr=thr, use_ef=True)).sort("sid")
            diff = cv.bootstrap_diff(base["f"].to_numpy(), d["f"].to_numpy(), n=200)
            byc = dict(d.group_by("country").agg(pl.col("f").mean()).iter_rows())
            res[str(thr)] = dict(f=float(d["f"].mean()), diff=diff, by_country=byc)
            log(f"stack EF floor {thr}: F {d['f'].mean():.6f}  vs v7c {diff[0]:+.6f} [{diff[1]:+.6f}, {diff[2]:+.6f}]  {byc}")
        json.dump(res, open(cache("stack_eval.json"), "w"), indent=1)
    else:
        o = build("test")
        ms = [lgb.Booster(model_file=cache(f"stack_fold{k}.txt")) for k in (0, 1)]
        s = np.mean([m.predict(X(o)) for m in ms], axis=0).astype(np.float32)
        o.select("qid", "sid", "p").with_columns(pl.Series("s", s)).write_parquet(cache("stack_test.parquet"))
        log("test stacked probabilities written")
