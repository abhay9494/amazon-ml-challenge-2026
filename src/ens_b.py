"""v6b: ensemble of the released v6 stage-1 models with a second cross-fitted stage-1 model of
different capacity (255 leaves / lr 0.08) on the same v6 features and family folds.  Adopted only if
the averaged OOF beats v6 in a paired bootstrap; outputs go to output_v6b (never over v6)."""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import cv
import pipeline as P
from common import cache, log
from decide import decide

PB = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)
qf = cv.query_folds()
lf = P.scan_feats("train").join(qf.lazy(), on="qid", how="left")
cols = json.load(open(cache("cv_m1_meta.json")))["cols"]
oof_b = []
if not os.path.exists(cache("cv_oof1_b.parquet")):
    for k in (0, 1):
        tr = lf.filter((pl.col("fold") != k) & ~cv._es()).collect()
        va = lf.filter((pl.col("fold") != k) & cv._es()).collect()
        X, y = P._xy(tr, cols)
        Xv, yv = P._xy(va, cols)
        del tr, va
        ds = lgb.Dataset(X, y, feature_name=cols, free_raw_data=True)
        m = lgb.train(dict(PB, seed=21 + k), ds, num_boost_round=3000, valid_sets=[lgb.Dataset(Xv, yv, reference=ds)],
                      callbacks=[lgb.log_evaluation(200), lgb.early_stopping(60, verbose=False)])
        del X, Xv, ds
        log(f"model B fold {k}: best_iter={m.best_iteration}")
        m.save_model(cache(f"cv_m1b_fold{k}.txt"), num_iteration=m.best_iteration)
        for part in P.stage_features("train"):
            te = pl.read_parquet(part).join(qf, on="qid", how="left").filter(pl.col("fold") == k)
            oof_b.append(te.select("qid", "sid").with_columns(pl.Series("p", P.predict(m, te, cols))))
    pl.concat(oof_b).write_parquet(cache("cv_oof1_b.parquet"))
ob = pl.read_parquet(cache("cv_oof1_b.parquet"))
oa = pl.read_parquet(cache("cv_oof1.parquet")).select("qid", "sid", "p")
ens = oa.join(ob.rename({"p": "pb"}), on=["qid", "sid"]).with_columns(((pl.col("p") + pl.col("pb")) / 2).alias("p")).select("qid", "sid", "p")
rel = json.load(open(cache("release.json")))
d_v6 = cv.per_entity(decide(oa, **rel["policy"])).sort("sid")
best = cv.tune_policy(ens, "v6b ensemble")
d = cv.bootstrap_diff(d_v6["f"].to_numpy(), best[2].sort("sid")["f"].to_numpy())
log(f"v6b ensemble - v6: {d[0]:+.5f}  95% CI [{d[1]:+.5f}, {d[2]:+.5f}]  (ensemble policy {best[1]})")
if d[1] > 0:
    models = [lgb.Booster(model_file=cache(f"cv_m1b_fold{k}.txt")) for k in (0, 1)]
    pa = pl.read_parquet(cache("cv_test_p1.parquet"))
    parts = []
    for part in P.stage_features("test"):
        f = pl.read_parquet(part)
        parts.append(f.select("qid", "sid").with_columns(pl.Series("pb", np.mean([P.predict(m, f, cols) for m in models], axis=0).astype(np.float32))))
    pt = pa.join(pl.concat(parts), on=["qid", "sid"]).with_columns(((pl.col("p") + pl.col("pb")) / 2).alias("p")).select("qid", "sid", "p")
    pt.write_parquet(cache("cv_test_p1_ens.parquet"))
    links = decide(pt, **best[1])
    ids = P.id_maps("test")
    out = cache("release_v6b")
    os.makedirs(out, exist_ok=True)
    P.write_lists(links, ids, "matched_entity_ids", os.path.join(out, "matching_results.tsv"))
    P.write_lists(pt.select("qid", "sid"), ids, "candidate_entity_ids", os.path.join(out, "candidate_pairs.tsv"))
    json.dump({"stage": "stage1-ensemble(v6 511-leaf + 255-leaf)", "policy": best[1], "clean_oof_macro_f05": best[0],
               "vs_v6": d}, open(os.path.join(out, "release_v6b.json"), "w"), indent=1)
    log(f"v6b written: {links.height:,} links")
else:
    log("v6b not adopted (no significant gain)")
