"""Test probabilities of stage-1 members A (v6, 511 leaves) and B (255 leaves): the mean of each
member's two cross-fitted fold models on the test features -> cache/v7_test_pred.parquet
(qid, sid, pa, pb).  Run after run_all.py (member A) and ens_b.py (member B)."""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

import pipeline as P
from common import cache, log

cols = json.load(open(cache("cv_m1_meta.json")))["cols"]
mA = [lgb.Booster(model_file=cache(f"cv_m1_fold{k}.txt")) for k in (0, 1)]
mB = [lgb.Booster(model_file=cache(f"cv_m1b_fold{k}.txt")) for k in (0, 1)]
parts = []
for part in P.stage_features("test"):
    f = pl.read_parquet(part)
    pa = np.mean([P.predict(m, f, cols) for m in mA], axis=0)
    pb = np.mean([P.predict(m, f, cols) for m in mB], axis=0)
    parts.append(f.select("qid", "sid").with_columns(pl.Series("pa", pa.astype(np.float32)), pl.Series("pb", pb.astype(np.float32))))
pr = pl.concat(parts)
assert pr.height == pr.select("qid", "sid").unique().height
pr.write_parquet(cache("v7_test_pred.parquet"))
log(f"test member probabilities A, B: {pr.height:,} pairs")
