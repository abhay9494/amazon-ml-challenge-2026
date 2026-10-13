"""v7c: two more cross-fitted stage-1 members on the same v6 train features and family folds —
C (1023 leaves, lr 0.05, ff 0.5) and D (extra-trees, 511 leaves, lr 0.08) — added to the released
A (v6, 511 leaves) + B (255 leaves) ensemble.  A combination is adopted only if its clean OOF
macro F0.5 beats A+B in a paired bootstrap over S1 entities (95 % CI above zero).  Test
predictions use the v7 test features; outputs go to output_v7c (never over output_v7)."""
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

BASE = dict(objective="binary", bagging_freq=1, verbose=-1, num_threads=0)
VARIANTS = {
    "c": dict(BASE, learning_rate=0.05, num_leaves=1023, min_data_in_leaf=100, feature_fraction=0.5,
              bagging_fraction=0.8, lambda_l2=5.0),
    "d": dict(BASE, learning_rate=0.08, num_leaves=511, min_data_in_leaf=50, feature_fraction=0.7,
              bagging_fraction=0.7, lambda_l2=2.0, extra_trees=True),
}
qf = cv.query_folds()
lf = P.scan_feats("train").join(qf.lazy(), on="qid", how="left")
cols = json.load(open(cache("cv_m1_meta.json")))["cols"]

for v, params in VARIANTS.items():
    if os.path.exists(cache(f"cv_oof1_{v}.parquet")):
        continue
    oof = []
    for k in (0, 1):
        tr = lf.filter((pl.col("fold") != k) & ~cv._es()).collect()
        va = lf.filter((pl.col("fold") != k) & cv._es()).collect()
        X, y = P._xy(tr, cols)
        Xv, yv = P._xy(va, cols)
        del tr, va
        ds = lgb.Dataset(X, y, feature_name=cols, free_raw_data=True)
        m = lgb.train(dict(params, seed=41 + k + (10 if v == "d" else 0)), ds, num_boost_round=4000,
                      valid_sets=[lgb.Dataset(Xv, yv, reference=ds)],
                      callbacks=[lgb.log_evaluation(200), lgb.early_stopping(60, verbose=False)])
        del X, Xv, ds
        log(f"model {v.upper()} fold {k}: best_iter={m.best_iteration}")
        m.save_model(cache(f"cv_m1{v}_fold{k}.txt"), num_iteration=m.best_iteration)
        for part in P.stage_features("train"):
            te = pl.read_parquet(part).join(qf, on="qid", how="left").filter(pl.col("fold") == k)
            oof.append(te.select("qid", "sid").with_columns(pl.Series("p", P.predict(m, te, cols))))
    pl.concat(oof).write_parquet(cache(f"cv_oof1_{v}.parquet"))

o = pl.read_parquet(cache("cv_oof1.parquet")).select("qid", "sid", pl.col("p").alias("a"))
for v in ("b", "c", "d"):
    o = o.join(pl.read_parquet(cache(f"cv_oof1_{v}.parquet")).rename({"p": v}), on=["qid", "sid"])
pol = dict(thr=0.7, use_ef=True)
ref = cv.per_entity(decide(o.select("qid", "sid", ((pl.col("a") + pl.col("b")) / 2).alias("p")), **pol)).sort("sid")
log(f"A+B: {ref['f'].mean():.5f}  by country {dict(ref.group_by('country').agg(pl.col('f').mean()).iter_rows())}")
res = {}
for combo in ("abc", "abd", "abcd", "ac"):
    pred = o.select("qid", "sid", pl.mean_horizontal(*combo).alias("p"))
    f, pol_c, d = cv.tune_policy(pred, combo)
    diff = cv.bootstrap_diff(ref["f"].to_numpy(), d.sort("sid")["f"].to_numpy())
    res[combo] = dict(f=f, policy=pol_c, vs_ab=diff)
    log(f"{combo}: {f:.5f} (policy {pol_c}); vs A+B {diff[0]:+.5f} [{diff[1]:+.5f}, {diff[2]:+.5f}]"
        f"  by country {dict(d.group_by('country').agg(pl.col('f').mean()).iter_rows())}")
json.dump(res, open(cache("v7c_eval.json"), "w"), indent=1)
best = max(res, key=lambda c: res[c]["f"])
if res[best]["vs_ab"][1] <= 0:
    log(f"no combination beats A+B significantly (best {best}); v7 stays")
    sys.exit(0)
log(f"adopting {best}")
tp = pl.read_parquet(cache("v7_test_pred.parquet")).rename({"pa": "a", "pb": "b"})
for v in ("c", "d"):
    if v not in best:
        continue
    models = [lgb.Booster(model_file=cache(f"cv_m1{v}_fold{k}.txt")) for k in (0, 1)]
    parts = []
    for part in P.stage_features("test"):
        f = pl.read_parquet(part)
        parts.append(f.select("qid", "sid").with_columns(pl.Series(v, np.mean([P.predict(m, f, cols) for m in models], axis=0).astype(np.float32))))
    tp = tp.join(pl.concat(parts), on=["qid", "sid"])
tp.write_parquet(cache("v7c_test_pred.parquet"))
pt = tp.select("qid", "sid", pl.mean_horizontal(*best).alias("p"))
links = decide(pt, **res[best]["policy"])
ids = P.id_maps("test")
out = cache("release_v7c")
os.makedirs(out, exist_ok=True)
P.write_lists(links, ids, "matched_entity_ids", os.path.join(out, "matching_results.tsv"))
P.write_lists(pt.select("qid", "sid"), ids, "candidate_entity_ids", os.path.join(out, "candidate_pairs.tsv"))
json.dump({"stage": f"stage1-ensemble({best})", "members": {"a": "v6 511 leaves", "b": "255 leaves lr 0.08", "c": VARIANTS["c"], "d": VARIANTS["d"]},
           **res[best]}, open(os.path.join(out, "release_v7c.json"), "w"), indent=1, default=str)
log(f"v7c written: {links.height:,} links")
