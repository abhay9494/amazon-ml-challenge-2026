"""Clean-validation report: candidate oracle per stage, error budget, subgroup tables.

Reads the clean out-of-fold predictions produced by cv.py (family folds, cross-fitted pruner,
outer stage 1, nested stage 2) and the release configuration.  Writes report.json and prints
tables.  Decomposition convention (sequential): candidate absence = 1 - oracle(candidates);
matcher recall loss = oracle(candidates) - F(same true positives, no false positives);
false-positive loss = F(no FP) - F(actual)."""
import glob
import json
import os

import polars as pl

import pipeline as P
from common import cache, log
from cv import bootstrap_diff, per_entity
from decide import decide


def retrieval_true_pairs() -> pl.DataFrame:
    truth = P.truth_pairs(P.id_maps("train"))
    hits = []
    for part in sorted(glob.glob(os.path.join(cache("train_retrieval"), "part_*.parquet"))):
        hits.append(pl.read_parquet(part, columns=["qid", "sid"]).join(truth, on=["qid", "sid"]))
    return pl.concat(hits)


def oracle_f(true_pairs_kept: pl.DataFrame) -> float:
    return per_entity(true_pairs_kept.select("qid", "sid"))["f"].mean()


def main():
    rel = json.load(open(cache("release.json")))
    ids = P.id_maps("train")
    truth = P.truth_pairs(ids)
    oof = pl.read_parquet(cache("cv_oof2.parquet" if rel["stage"] == "stage2" else "cv_oof1.parquet"))
    rep = {"release": rel}

    # ---- candidate oracle per stage
    ret = retrieval_true_pairs()
    cands = pl.read_parquet(cache("train_cands.parquet"), columns=["qid", "sid", "y"])
    kept_true = cands.filter(pl.col("y") == 1).select("qid", "sid")
    rep["oracle"] = {
        "raw_retrieval": {"link_recall": ret.height / truth.height, "macro_f05": oracle_f(ret)},
        "after_pruning(final scored set)": {"link_recall": kept_true.height / truth.height, "macro_f05": oracle_f(kept_true)},
    }

    # ---- error budget of the released policy
    links = decide(oof.select("qid", "sid", "p"), **rel["policy"])
    d = per_entity(links)
    tp_links = links.join(truth, on=["qid", "sid"])
    d_nofp = per_entity(tp_links.select("qid", "sid"))
    f, f_nofp, f_or = d["f"].mean(), d_nofp["f"].mean(), rep["oracle"]["after_pruning(final scored set)"]["macro_f05"]
    rep["error_budget"] = {"macro_f05": f, "candidate_absence": 1 - f_or, "matcher_recall_loss": f_or - f_nofp,
                           "false_positive_loss": f_nofp - f,
                           "fp_links": links.height - tp_links.height, "fn_links": truth.height - tp_links.height,
                           "singleton_false_merges": d.filter((pl.col("nt") == 0) & (pl.col("np") > 0)).height}

    # ---- subgroups (S1 level)
    size = d.with_columns(pl.when(pl.col("nt") >= 6).then(6).otherwise(pl.col("nt")).alias("nt_b"))
    rep["by_country"] = {r[0]: r[1] for r in d.group_by("country").agg(pl.col("f").mean()).iter_rows()}
    rep["by_true_set_size"] = {int(r[0]): (r[1], r[2]) for r in size.group_by("nt_b").agg(pl.col("f").mean(), pl.len()).sort("nt_b").iter_rows()}

    # ---- subgroups (link level): recall of true pairs by query properties
    nrm = P.normalize_split("train", columns=["rid", "src", "native", "addr_empty", "n_ns", "country"])
    s1 = nrm.filter(pl.col("src") == 1)
    ndf = s1.group_by("country", "n_ns").len().rename({"len": "ndf"})
    q = nrm.filter(pl.col("src") != 1).join(ndf, on=["country", "n_ns"], how="left").with_columns(pl.col("ndf").fill_null(0))
    t = truth.join(q.rename({"rid": "qid"}), on="qid").join(tp_links.with_columns(pl.lit(1).alias("hit")), on=["qid", "sid"], how="left").with_columns(pl.col("hit").fill_null(0))
    t = t.join(kept_true.with_columns(pl.lit(1).alias("cand")), on=["qid", "sid"], how="left").with_columns(pl.col("cand").fill_null(0))
    def grp(col):
        return {str(r[0]): {"n": r[1], "candidate_recall": r[2], "final_recall": r[3]}
                for r in t.group_by(col).agg(pl.len(), pl.col("cand").mean(), pl.col("hit").mean()).sort(col).iter_rows()}
    t = t.with_columns(pl.when(pl.col("ndf") == 0).then(pl.lit("0")).when(pl.col("ndf") == 1).then(pl.lit("1"))
                       .when(pl.col("ndf") <= 5).then(pl.lit("2-5")).otherwise(pl.lit(">5")).alias("name_freq"))
    rep["link_recall_by"] = {c: grp(c) for c in ("country", "src", "native", "addr_empty", "name_freq")}
    fp = links.join(truth, on=["qid", "sid"], how="anti").join(q.rename({"rid": "qid"}), on="qid")
    rep["false_positives_by_src"] = {str(r[0]): r[1] for r in fp.group_by("src").len().iter_rows()}

    # ---- stage comparison with S1-level bootstrap
    if os.path.exists(cache("cv_oof1.parquet")) and os.path.exists(cache("cv_oof2.parquet")):
        o1 = pl.read_parquet(cache("cv_oof1.parquet"))
        o2 = pl.read_parquet(cache("cv_oof2.parquet"))
        d1 = per_entity(decide(o1.select("qid", "sid", "p"), **rel["policy"])).sort("sid")
        d2 = per_entity(decide(o2.select("qid", "sid", "p"), **rel["policy"])).sort("sid")
        rep["stage2_minus_stage1_same_policy"] = bootstrap_diff(d1["f"].to_numpy(), d2["f"].to_numpy())
    json.dump(rep, open(cache("report.json"), "w"), indent=1, default=str)
    log(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
