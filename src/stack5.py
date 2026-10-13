"""Re-scorer v5 = v4 features + raw name identity / ambiguity features:
name_cf_eq, addr_cf_eq (casefolded, whitespace-normalised raw equality), q_cf_s1df (S1 records of
the country with the record's exact casefolded name), q_cf_qdf (S2/S3 records sharing it), s_cf_qdf
(S2/S3 records carrying the S1's exact name), q_core_qdf (S2/S3 records sharing the corrected core
name), tie_n (the record's candidates whose S1 core name equals this candidate's core name).
usage: python stack5.py train | test"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import pipeline as P
from common import cache, load_raw, log
from decide import decide
from stack import PARAMS
from stack4 import FEATS as F4, build3

IDF = ["name_cf_eq", "addr_cf_eq", "q_cf_s1df", "q_cf_qdf", "s_cf_qdf", "q_core_qdf", "tie_n"]
FEATS = F4 + IDF
ROUNDS5 = 700


def cf(col):
    return pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()


def build5(split):
    o = build3(split)
    raw = load_raw(split).select("rid", "src", "country", cf("business_name").alias("ncf"), cf("business_address").alias("acf"))
    nrm = pl.read_parquet(cache(f"{split}_norm.parquet"), columns=["rid", "n_core_c", "n_core"])
    raw = raw.join(nrm, on="rid")
    s1 = raw.filter(pl.col("src") == 1)
    q = raw.filter(pl.col("src") != 1)
    s1df = s1.group_by("country", "ncf").len().rename({"len": "q_cf_s1df"})
    qdf = q.group_by("country", "ncf").len().rename({"len": "q_cf_qdf"})
    coredf = q.group_by("country", "n_core_c").len().rename({"len": "q_core_qdf"})
    qq = q.select(pl.col("rid").alias("qid"), "country", "ncf", "acf", "n_core_c") \
        .join(s1df, on=["country", "ncf"], how="left").join(qdf, on=["country", "ncf"], how="left").join(coredf, on=["country", "n_core_c"], how="left")
    ss = s1.select(pl.col("rid").alias("sid"), pl.col("ncf").alias("s_ncf"), pl.col("acf").alias("s_acf"), pl.col("n_core").alias("s_core"), "country") \
        .join(qdf.rename({"ncf": "s_ncf", "q_cf_qdf": "s_cf_qdf"}), on=["country", "s_ncf"], how="left").drop("country")
    x = o.select("qid", "sid").join(qq, on="qid", how="left").join(ss, on="sid", how="left")
    x = x.with_columns((pl.col("ncf") == pl.col("s_ncf")).cast(pl.Int8).alias("name_cf_eq"),
                       ((pl.col("acf") == pl.col("s_acf")) & (pl.col("acf") != "")).cast(pl.Int8).alias("addr_cf_eq"),
                       pl.len().over("qid", "s_core").cast(pl.Int16).alias("tie_n"))
    x = x.select("qid", "sid", "name_cf_eq", "addr_cf_eq", *[pl.col(c).fill_null(0).cast(pl.Int32) for c in ("q_cf_s1df", "q_cf_qdf", "s_cf_qdf", "q_core_qdf")], "tie_n")
    n0 = o.height
    o = o.join(x, on=["qid", "sid"], how="left")
    assert o.height == n0
    return o


def X(df):
    return df.select(FEATS).to_numpy().astype(np.float32)


if __name__ == "__main__":
    split = sys.argv[1]
    if split == "train":
        import cv
        from folds import query_folds
        o = build5("train")
        n0 = o.height
        truth = P.truth_pairs(P.id_maps("train")).with_columns(pl.lit(1, pl.Int8).alias("y"))
        o = o.join(truth, on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(0)).join(query_folds().select("qid", "fold"), on="qid", how="left")
        assert o.height == n0
        log(f"train table {o.shape}")
        oof = []
        for k in (0, 1):
            tr = o.filter(pl.col("fold") != k)
            m = lgb.train(dict(PARAMS, seed=500 + k), lgb.Dataset(X(tr), tr["y"].to_numpy(), feature_name=FEATS, free_raw_data=True), ROUNDS5)
            del tr
            m.save_model(cache(f"stack5_fold{k}.txt"))
            te = o.filter(pl.col("fold") == k)
            oof.append(te.select("qid", "sid").with_columns(pl.Series("s5", m.predict(X(te)).astype(np.float32))))
            log(f"stack5 fold {k} done")
            imp = dict(zip(FEATS, m.feature_importance("gain")))
            log("new-feature gains: " + ", ".join(f"{c}:{imp[c]:.0f}" for c in IDF))
        oof = pl.concat(oof)
        oof.write_parquet(cache("stack5_oof.parquet"))
        base = cv.per_entity(decide(o.select("qid", "sid", "p"), thr=0.7, use_ef=True)).sort("sid")
        d = cv.per_entity(decide(oof.select("qid", "sid", pl.col("s5").alias("p")), thr=0.6, use_ef=True)).sort("sid")
        diff = cv.bootstrap_diff(base["f"].to_numpy(), d["f"].to_numpy(), n=200)
        log(f"s5@0.6: F {d['f'].mean():.6f} vs stage-1 ensemble (EF 0.7) {diff[0]:+.6f} [{diff[1]:+.6f}, {diff[2]:+.6f}]")
        json.dump(dict(f=float(d["f"].mean()), vs_stage1=diff), open(cache("stack5_eval.json"), "w"), indent=1)
    else:
        o = build5("test")
        ms = [lgb.Booster(model_file=cache(f"stack5_fold{k}.txt")) for k in (0, 1)]
        s5 = np.mean([m.predict(X(o)) for m in ms], axis=0).astype(np.float32)
        o.select("qid", "sid", "p").with_columns(pl.Series("s5", s5)).write_parquet(cache("stack5_test.parquet"))
        log("test stack5 written")
