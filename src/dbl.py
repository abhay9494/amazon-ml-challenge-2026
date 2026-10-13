"""Letter-doubling decoys: a query name token that equals an S1 name token after collapsing repeated
letters, but has MORE letters (a doubled letter the S1 spelling does not have).  True-match rate on
train links (labels), by how many such tokens and by country; count on test links."""
import re
import sys

import polars as pl

import pipeline as P
from common import cache, load_raw
from decide import decide

_REP = re.compile(r"(.)\1+")
_TOK = re.compile(r"[^\W_]+", re.UNICODE)


def toks(s):
    return [t for t in _TOK.findall(s.casefold()) if t]


def dbl_feats(qn, sn):
    """(#query tokens with an extra doubled letter vs an S1 token, #S1 tokens with an extra doubled
    letter vs a query token)"""
    qt, st = toks(qn), toks(sn)
    sset, qset = set(st), set(qt)
    sc = {}
    for t in st:
        sc.setdefault(_REP.sub(r"\1", t), []).append(t)
    qc = {}
    for t in qt:
        qc.setdefault(_REP.sub(r"\1", t), []).append(t)
    nq = ns = 0
    for t in qt:
        if t in sset:
            continue
        c = _REP.sub(r"\1", t)
        if c in sc and any(len(t) > len(u) for u in sc[c]) and not any(u == t for u in sc[c]):
            nq += 1
    for t in st:
        if t in qset:
            continue
        c = _REP.sub(r"\1", t)
        if c in qc and any(len(t) > len(u) for u in qc[c]):
            ns += 1
    return nq, ns


def pred_for(split):
    if split == "train":
        o = pl.read_parquet(cache("cv_oof1.parquet")).select("qid", "sid", pl.col("p").alias("a"))
        for v in "bc":
            o = o.join(pl.read_parquet(cache(f"cv_oof1_{v}.parquet")).rename({"p": v}), on=["qid", "sid"])
        return o.select("qid", "sid", pl.mean_horizontal("a", "b", "c").alias("p"))
    return pl.read_parquet(cache("v7c_test_pred.parquet")).select("qid", "sid", pl.mean_horizontal("a", "b", "c").alias("p"))


def flagged_links(split):
    pred = pred_for(split)
    links = decide(pred, thr=0.7, use_ef=True).join(pred, on=["qid", "sid"], how="left")
    raw = load_raw(split).select("rid", "business_name", "country")
    x = links.join(raw.select(pl.col("rid").alias("qid"), pl.col("business_name").alias("qn"), "country"), on="qid") \
             .join(raw.select(pl.col("rid").alias("sid"), pl.col("business_name").alias("sn")), on="sid")
    f = [dbl_feats(a, b) for a, b in zip(x["qn"].to_list(), x["sn"].to_list())]
    x = x.with_columns(pl.Series("dq", [a for a, _ in f], dtype=pl.Int8), pl.Series("ds", [b for _, b in f], dtype=pl.Int8))
    return pred, links, x


if __name__ == "__main__":
    pred, links, x = flagged_links("train")
    truth = P.truth_pairs(P.id_maps("train")).with_columns(pl.lit(True).alias("y"))
    x = x.join(truth, on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(False))
    with pl.Config(tbl_rows=30):
        print(x.group_by("country", pl.col("dq").clip(0, 2), pl.col("ds").clip(0, 2)).agg(pl.len(), pl.col("y").mean().round(4).alias("true"),
              (~pl.col("y")).sum().alias("FP")).sort("country", "dq", "ds"))
    x.select("qid", "sid", "p", "y", "dq", "ds", "country").write_parquet("dbl_train_links.parquet")
    _, _, t = flagged_links("test")
    print(t.group_by("country", pl.col("dq").clip(0, 2), pl.col("ds").clip(0, 2)).agg(pl.len()).sort("country", "dq", "ds"))
    t.select("qid", "sid", "p", "dq", "ds", "country").write_parquet("dbl_test_links.parquet")
