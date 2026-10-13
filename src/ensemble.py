"""Stage-2 ensemble: extra cross-fitted stage-2 models with different hyper-parameters / seeds.

Each variant is trained exactly like pipeline.stage_match2 (2 folds by S1 entity, early stopping on
an entity-level slice) and produces out-of-fold probabilities.  The ensemble probability is the mean
over the base stage-2 model and the variants; it is adopted only if its out-of-fold macro F0.5 is
higher than the base model's.

    python ensemble.py            # train variants, evaluate, write test predictions if better
"""
import os

import lightgbm as lgb
import numpy as np
import polars as pl

import pipeline as P
from common import cache, log

VARIANTS = {
    "b": dict(num_leaves=127, learning_rate=0.05, feature_fraction=0.5, bagging_fraction=0.8, min_data_in_leaf=200, seed=201),
    "c": dict(num_leaves=511, learning_rate=0.05, feature_fraction=0.6, bagging_fraction=0.7, min_data_in_leaf=50, lambda_l2=5.0, seed=301),
}


def train_variant(name: str, params: dict, force: bool = False) -> pl.DataFrame:
    p_oof = cache(f"train_oof2{name}.parquet")
    if os.path.exists(p_oof) and not force:
        return pl.read_parquet(p_oof)
    coll = P.collective("train")
    lf = P.scan_feats("train").join(coll.lazy(), on=["qid", "sid"], how="left").with_columns(P.fold_expr().alias("fold"))
    cols = P.feature_cols(lf)
    es = (pl.col("sid").hash(seed=7) % 20) == 0
    oof = []
    for k in (0, 1):
        tr = lf.filter((pl.col("fold") != k) & ~es).collect()
        va = lf.filter((pl.col("fold") != k) & es).collect()
        X, y = P._xy(tr, cols)
        Xv, yv = P._xy(va, cols)
        del tr, va
        ds = lgb.Dataset(X, y, feature_name=cols, free_raw_data=True)
        dv = lgb.Dataset(Xv, yv, reference=ds)
        prm = dict(P.PARAMS)
        prm.update(params)
        m = lgb.train(prm, ds, num_boost_round=3000, valid_sets=[dv],
                      callbacks=[lgb.log_evaluation(200), lgb.early_stopping(50, verbose=False)])
        del X, Xv, ds, dv
        log(f"variant {name} fold {k}: best_iter={m.best_iteration}")
        m.save_model(cache(f"matcher2{name}_fold{k}.txt"), num_iteration=m.best_iteration)
        for part in P.stage_features("train"):
            te = (pl.read_parquet(part).join(coll, on=["qid", "sid"], how="left")
                  .with_columns(P.fold_expr().alias("fold")).filter(pl.col("fold") == k))
            oof.append(te.select("qid", "sid", "y").with_columns(pl.Series("p", P.predict(m, te, cols))))
        del m
    out = pl.concat(oof)
    out.write_parquet(p_oof)
    return out


def predict_test(names) -> pl.DataFrame:
    coll = P.collective("test")
    models = []
    for name in names:
        models += [lgb.Booster(model_file=cache(f"matcher2{name}_fold{k}.txt")) for k in (0, 1)]
    cols = models[0].feature_name()
    preds = []
    for part in P.stage_features("test"):
        f = pl.read_parquet(part).join(coll, on=["qid", "sid"], how="left")
        p = np.mean([P.predict(m, f, cols) for m in models], axis=0)
        preds.append(f.select("qid", "sid").with_columns(pl.Series("p", p.astype(np.float32))))
    return pl.concat(preds)


def main():
    ids = P.id_maps("train")
    base = pl.read_parquet(cache("train_oof2.parquet"))
    oofs = {"": base}
    for name, prm in VARIANTS.items():
        oofs[name] = train_variant(name, prm)
    keys = ["qid", "sid"]
    ens = base.select("qid", "sid", "y", pl.col("p").alias("p0"))
    for name in VARIANTS:
        ens = ens.join(oofs[name].select(*keys, pl.col("p").alias(f"p{name}")), on=keys, how="left")
    ens = ens.with_columns(pl.mean_horizontal([c for c in ens.columns if c.startswith("p")]).alias("p"))
    log("== base stage-2 OOF")
    rb = P.evaluate(base, ids, thrs=(0.5,))
    log("== ensemble OOF")
    re_ = P.evaluate(ens.select("qid", "sid", "y", "p"), ids, thrs=(0.3, 0.5, 0.6))
    best_base = max(r["macro_f05"] for r in rb.values())
    best_ens = max(r["macro_f05"] for r in re_.values())
    log(f"base {best_base:.5f}  ensemble {best_ens:.5f}")
    if best_ens > best_base:
        pred = predict_test([""] + list(VARIANTS))
        pred.write_parquet(cache("test_pred2_ens.parquet"))
        log("ensemble adopted: cache/test_pred2_ens.parquet")


if __name__ == "__main__":
    main()
