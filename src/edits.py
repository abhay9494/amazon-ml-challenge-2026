"""Raw-text edit flags per linked pair (query name vs S1 name, casefolded tokens), to find decoy
mechanisms that the normalisation erases.  Flags: dq (extra doubled letter), ins (one inserted
non-doubling letter), dele (one deleted letter), sub (one substituted letter), trans (adjacent swap),
roman (roman-numeral token differs), legal_add (query has a legal-form token, S1 has none),
legal_drop, legal_diff.  Train: true rate per flag; saves flags for train and test links."""
import re
import sys

import polars as pl

import pipeline as P
from common import cache, load_raw
from decide import decide

_REP = re.compile(r"(.)\1+")
_TOK = re.compile(r"[^\W_]+", re.UNICODE)
LEGAL = {"inc", "incorporated", "corp", "corporation", "co", "company", "llc", "ltd", "limited", "pvt", "private", "llp", "lp",
         "pc", "plc", "pa", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "l", "c"}
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


def toks(s):
    return [t for t in _TOK.findall(s.casefold()) if t]


def edit_kind(q, s):
    if len(q) == len(s) + 1:
        for i in range(len(q)):
            if q[:i] + q[i + 1:] == s:
                return "dq" if (i > 0 and q[i - 1] == q[i]) or (i + 1 < len(q) and q[i + 1] == q[i]) else "ins"
        if _REP.sub(r"\1", q) == _REP.sub(r"\1", s):
            return "dq"
        return None
    if len(q) == len(s) - 1:
        for i in range(len(s)):
            if s[:i] + s[i + 1:] == q:
                return "dele"
        return None
    if len(q) == len(s) and len(q) > 1:
        d = [i for i in range(len(q)) if q[i] != s[i]]
        if len(d) == 1:
            return "sub"
        if len(d) == 2 and d[1] == d[0] + 1 and q[d[0]] == s[d[1]] and q[d[1]] == s[d[0]]:
            return "trans"
    if _REP.sub(r"\1", q) == _REP.sub(r"\1", s) and len(q) > len(s):
        return "dq"
    return None


def flags(qn, sn):
    qt, st = toks(qn), toks(sn)
    qs, ss = set(qt), set(st)
    f = dict(dq=0, ins=0, dele=0, sub=0, trans=0, roman=0)
    for t in qt:
        if t in ss:
            continue
        if t in ROMAN and (ss & ROMAN):
            f["roman"] += 1
            continue
        kinds = [edit_kind(t, u) for u in st if u not in qs and len(u) >= 3 and len(t) >= 3]
        for k in ("dq", "ins", "trans", "sub", "dele"):
            if k in kinds:
                f[k] += 1
                break
    ql, sl = qs & LEGAL, ss & LEGAL
    f["legal_add"] = int(bool(ql) and not sl)
    f["legal_drop"] = int(bool(sl) and not ql)
    f["legal_diff"] = int(bool(ql) and bool(sl) and ql != sl)
    return f


def pred_for(split):
    if split == "train":
        o = pl.read_parquet(cache("cv_oof1.parquet")).select("qid", "sid", pl.col("p").alias("a"))
        for v in "bc":
            o = o.join(pl.read_parquet(cache(f"cv_oof1_{v}.parquet")).rename({"p": v}), on=["qid", "sid"])
        return o.select("qid", "sid", pl.mean_horizontal("a", "b", "c").alias("p"))
    return pl.read_parquet(cache("v7c_test_pred.parquet")).select("qid", "sid", pl.mean_horizontal("a", "b", "c").alias("p"))


def link_flags(split):
    pred = pred_for(split)
    links = decide(pred, thr=0.7, use_ef=True).join(pred, on=["qid", "sid"], how="left")
    raw = load_raw(split).select("rid", "business_name", "business_address", "country", "src")
    x = links.join(raw.select(pl.col("rid").alias("qid"), pl.col("business_name").alias("qn"), (pl.col("business_address").str.strip_chars() == "").alias("qblank"), "country", "src"), on="qid") \
             .join(raw.select(pl.col("rid").alias("sid"), pl.col("business_name").alias("sn")), on="sid")
    F = [flags(a, b) for a, b in zip(x["qn"].to_list(), x["sn"].to_list())]
    return x.with_columns(*[pl.Series(k, [f[k] for f in F], dtype=pl.Int8) for k in F[0]]).drop("qn", "sn")


if __name__ == "__main__":
    split = sys.argv[1]
    x = link_flags(split)
    if split == "train":
        truth = P.truth_pairs(P.id_maps("train")).with_columns(pl.lit(True).alias("y"))
        x = x.join(truth, on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(False))
        for k in ("dq", "ins", "dele", "sub", "trans", "roman", "legal_add", "legal_drop", "legal_diff"):
            print(k, x.filter(pl.col(k) > 0).group_by("country").agg(pl.len(), pl.col("y").mean().round(3).alias("true"),
                  pl.col("y").filter(pl.col("p") < 0.999).mean().round(3).alias("true|p<.999"), (pl.col("p") < 0.999).sum().alias("n|p<.999")).sort("country").rows())
    x.write_parquet(f"edits_{split}_links.parquet")
