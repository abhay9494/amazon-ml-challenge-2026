"""Re-scorer v6 = same inputs as v5 (stack5.py), different bagging / feature sampling / learning rate,
for averaging with v5.  Trains both folds, evaluates mean(s5, s6) vs s5 on the exact metric, and
scores the test set only if the average is significantly better.  usage: python stack6.py  (the released v13c uses mean(s5, s6); the test set is always scored)"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import cv
import pipeline as P
from common import cache, log
from decide import decide
from folds import query_folds
from stack5 import FEATS, X, build5

P6 = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.6,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=0)
R6 = 800

o = build5("train")
n0 = o.height
truth = P.truth_pairs(P.id_maps("train")).with_columns(pl.lit(1, pl.Int8).alias("y"))
o = o.join(truth, on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(0)).join(query_folds().select("qid", "fold"), on="qid", how="left")
assert o.height == n0
oof = []
for k in (0, 1):
    tr = o.filter(pl.col("fold") != k)
    m = lgb.train(dict(P6, seed=600 + k), lgb.Dataset(X(tr), tr["y"].to_numpy(), feature_name=FEATS, free_raw_data=True), R6)
    del tr
    m.save_model(cache(f"stack6_fold{k}.txt"))
    te = o.filter(pl.col("fold") == k)
    oof.append(te.select("qid", "sid").with_columns(pl.Series("s6", m.predict(X(te)).astype(np.float32))))
    log(f"stack6 fold {k} done")
del o
oof = pl.concat(oof).join(pl.read_parquet(cache("stack5_oof.parquet")), on=["qid", "sid"])
assert oof.height == n0
oof.write_parquet(cache("stack6_oof.parquet"))
r = cv.per_entity(decide(oof.select("qid", "sid", pl.col("s5").alias("p")), thr=0.6, use_ef=True)).sort("sid")
res = {}
for name, expr in (("s6", pl.col("s6")), ("avg56", (pl.col("s5") + pl.col("s6")) / 2)):
    d = cv.per_entity(decide(oof.select("qid", "sid", expr.alias("p")), thr=0.6, use_ef=True)).sort("sid")
    diff = cv.bootstrap_diff(r["f"].to_numpy(), d["f"].to_numpy(), n=200)
    res[name] = dict(f=float(d["f"].mean()), vs_s5=diff)
    log(f"{name}@0.6: F {d['f'].mean():.6f} vs s5 {diff[0]:+.6f} [{diff[1]:+.6f}, {diff[2]:+.6f}]")
json.dump(res, open(cache("stack6_eval.json"), "w"), indent=1)
if True:  # adopted on validation: mean(s5, s6) vs s5 +0.00003 [+0.00001, +0.00004]
    t = build5("test")
    ms = [lgb.Booster(model_file=cache(f"stack6_fold{k}.txt")) for k in (0, 1)]
    s6 = np.mean([m.predict(X(t)) for m in ms], axis=0).astype(np.float32)
    t.select("qid", "sid").with_columns(pl.Series("s6", s6)).write_parquet(cache("stack6_test.parquet"))
    log("test stack6 written")
else:
    log("average not significantly better; v12a stays")
