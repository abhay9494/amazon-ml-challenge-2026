"""Differences that the normalisation erases, per linked pair (raw query name vs raw S1 name):
accent_add (query has a diacritic letter where S1 has the plain letter), leet (digit inside a word),
dblspace, caps (query all upper, S1 not), punct_diff (different punctuation set), order (same token
multiset, different order).  Train: true rate; saves flags for train/test links."""
import re
import sys
import unicodedata

import polars as pl

import pipeline as P
from common import load_raw

_W = re.compile(r"[^\W_]+", re.UNICODE)


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def has_acc(s):
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", s))


def fl(qn, sn):
    qc, sc = qn.casefold(), sn.casefold()
    qt, st = _W.findall(qc), _W.findall(sc)
    sset = set(st)
    sset_plain = {strip_acc(t) for t in st}
    acc = sum(1 for t in qt if t not in sset and has_acc(t) and strip_acc(t) in sset_plain and strip_acc(t) in sset)
    leet = sum(1 for t in _W.findall(qn) if re.search(r"[A-Za-z]", t) and re.search(r"\d", t))
    return dict(
        accent_add=int(acc > 0),
        leet=int(leet > 0),
        dblspace=int("  " in qn),
        caps=int(qn.isupper() and not sn.isupper()),
        punct_diff=int(set(re.findall(r"[^\w\s]", qn)) != set(re.findall(r"[^\w\s]", sn))),
        order=int(sorted(qt) == sorted(st) and qt != st),
    )


if __name__ == "__main__":
    split = sys.argv[1]
    x = pl.read_parquet(f"edits_{split}_links.parquet").select("qid", "sid", "p", "country", "src", "dq")
    raw = load_raw(split).select("rid", "business_name")
    nm = dict(zip(raw["rid"].to_list(), raw["business_name"].to_list()))
    F = [fl(nm[a], nm[b]) for a, b in zip(x["qid"].to_list(), x["sid"].to_list())]
    x = x.with_columns(*[pl.Series(k, [f[k] for f in F], dtype=pl.Int8) for k in F[0]])
    if split == "train":
        truth = P.truth_pairs(P.id_maps("train")).with_columns(pl.lit(True).alias("y"))
        x = x.join(truth, on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(False))
        for k in F[0]:
            print(k, x.filter(pl.col(k) > 0).group_by("country").agg(pl.len(), pl.col("y").mean().round(3).alias("true"),
                  pl.col("y").filter(pl.col("p") < 0.999).mean().round(3).alias("true|p<.999"), (pl.col("p") < 0.999).sum().alias("n<.999")).sort("country").rows())
    else:
        for k in F[0]:
            print(k, x.filter(pl.col(k) > 0).group_by("country").agg(pl.len(), (pl.col("p") < 0.999).sum().alias("n<.999"), pl.col("p").mean().round(4)).sort("country").rows())
    x.write_parquet(f"erased_{split}_links.parquet")
