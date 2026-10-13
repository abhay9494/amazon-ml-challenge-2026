"""Re-scorer v4 = v3 + partner features at a strict partner threshold (p >= 0.8) + 700 rounds.

v3: stack.py features + 25 more main-model features + collective partner features
(stage2.collective_features on the ensemble p: similarity of the record to the S1's other confident
records).  Same family folds, OOF evaluation vs the first re-scorer (stack.py).
usage: python stack3.py train | test"""
import glob
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import pipeline as P
import stage2 as S2
from common import cache, log
from decide import decide
from stack import FEATS as F1, PARAMS, ROUNDS, build

EXTRA = ["h_digits_partial", "pr_other", "c_nsort", "h_qonly", "xt_max", "n_me_q_min", "n_idf_qonly", "h_sonly", "l_qn", "xt_n_unknown",
         "h_digits_ratio", "n_idf_common", "q_nlen_chars", "o_n_idf_cov_s", "n_idf_sonly", "xt_n_noise", "c_sum", "n_jw", "o_a_tset",
         "n_q_oov", "l_common", "h_sfirst_in_q", "n_ns_partial", "n_full_tset", "q_alt_name_df"]
K = ["k_n", "k_n_same_src", "k_maxp", "k_name_max", "k_addr_max", "k_name_bp", "k_addr_bp", "k_h_q_in_partners", "k_h_s_in_partners", "k_exact_dup"]
K8 = [f"{c}_8" for c in K]
FEATS = F1 + EXTRA + K + K8
ROUNDS4 = 700


def build3(split):
    o = build(split)
    parts = sorted(glob.glob(cache(f"{split}_feats/part_*.parquet")))
    ex = pl.concat([pl.read_parquet(p, columns=["qid", "sid"] + EXTRA) for p in parts])
    o = o.join(ex, on=["qid", "sid"], how="left")
    coll = S2.collective_features(o.select("qid", "sid", "p"), P.normalize_split(split, columns=S2.REC))
    kcols = [c for c in K if c in coll.columns]
    assert len(kcols) == len(K), set(K) - set(kcols)
    o = o.join(coll.select("qid", "sid", *K), on=["qid", "sid"], how="left")
    del coll
    S2.PARTNER_P = 0.8
    coll8 = S2.collective_features(o.select("qid", "sid", "p"), P.normalize_split(split, columns=S2.REC))
    S2.PARTNER_P = 0.5
    o = o.join(coll8.select("qid", "sid", *[pl.col(c).alias(f"{c}_8") for c in K]), on=["qid", "sid"], how="left")
    return o.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def X(df):
    return df.select(FEATS).to_numpy().astype(np.float32)


if __name__ == "__main__":
    split = sys.argv[1]
    if split == "train":
        import cv
        from folds import query_folds
        o = build3("train")
        n0 = o.height
        truth = P.truth_pairs(P.id_maps("train")).with_columns(pl.lit(1, pl.Int8).alias("y"))
        o = o.join(truth, on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(0)).join(query_folds().select("qid", "fold"), on="qid", how="left")
        assert o.height == n0
        log(f"train table {o.shape}")
        oof = []
        for k in (0, 1):
            tr = o.filter(pl.col("fold") != k)
            m = lgb.train(dict(PARAMS, seed=400 + k), lgb.Dataset(X(tr), tr["y"].to_numpy(), feature_name=FEATS, free_raw_data=True), ROUNDS4)
            del tr
            m.save_model(cache(f"stack4_fold{k}.txt"))
            te = o.filter(pl.col("fold") == k)
            oof.append(te.select("qid", "sid").with_columns(pl.Series("s4", m.predict(X(te)).astype(np.float32))))
            log(f"stack4 fold {k} done")
        oof = pl.concat(oof)
        oof.write_parquet(cache("stack4_oof.parquet"))
        s1o = pl.read_parquet(cache("stack3_oof.parquet")).rename({"s3": "s"})
        s1 = cv.per_entity(decide(s1o.select("qid", "sid", pl.col("s").alias("p")), thr=0.6, use_ef=True)).sort("sid")
        base = cv.per_entity(decide(o.select("qid", "sid", "p"), thr=0.7, use_ef=True)).sort("sid")
        res = {}
        for thr in (0.6, 0.7):
            d = cv.per_entity(decide(oof.select("qid", "sid", pl.col("s4").alias("p")), thr=thr, use_ef=True)).sort("sid")
            d1 = cv.bootstrap_diff(s1["f"].to_numpy(), d["f"].to_numpy(), n=200)
            d0 = cv.bootstrap_diff(base["f"].to_numpy(), d["f"].to_numpy(), n=100)
            res[str(thr)] = dict(f=float(d["f"].mean()), vs_stack1_07=d1, vs_v7c=d0, by_country=dict(d.group_by("country").agg(pl.col("f").mean()).iter_rows()))
            log(f"stack4 EF floor {thr}: F {d['f'].mean():.6f}  vs stack3@0.6 {d1[0]:+.6f} [{d1[1]:+.6f}, {d1[2]:+.6f}]  vs v7c {d0[0]:+.6f}")
        json.dump(res, open(cache("stack4_eval.json"), "w"), indent=1)
    else:
        o = build3("test")
        ms = [lgb.Booster(model_file=cache(f"stack4_fold{k}.txt")) for k in (0, 1)]
        s3 = np.mean([m.predict(X(o)) for m in ms], axis=0).astype(np.float32)
        o.select("qid", "sid", "p").with_columns(pl.Series("s4", s3)).write_parquet(cache("stack4_test.parquet"))
        log("test stack4 written")
