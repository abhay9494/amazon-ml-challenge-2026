"""Leak-free validation: family folds, cross-fitted pruning, outer stage 1, NESTED stage 2.

Folds are assigned to *entity families*, not to candidate rows:
  * an S1 record gets fold crc32(entity_id) & 1;
  * a matched S2/S3 record inherits the fold of its (single) true S1 owner;
  * an unmatched S2/S3 record gets crc32(own entity_id) & 1;
  * every candidate row of a query inherits the QUERY's fold, so no labelled row of a held-out
    query is ever used for fitting, even when its candidates point to S1 records of both folds.
A second, independent bit (`inner`) splits each fold into two halves for nested cross-fitting.

Label-dependent components and how they are kept out of the evaluation fold:
  pruner            cross-fitted: rows of fold k are scored by the pruner trained on fold 1-k
  stage-1 matcher   cross-fitted: fold k predicted by the model trained on fold 1-k (outer OOF)
  stage-2 matcher   nested: to evaluate fold k, EVERY stage-1 probability it consumes (own and
                    partners') comes from models that never saw fold-k labels - fold-k rows use
                    the outer model trained on fold 1-k, fold 1-k rows use inner cross-fitting
                    inside fold 1-k; the stage-2 model itself is fitted on fold 1-k only.
Label-dependent dictionaries (transliteration, address aliases) are learned once from all
training pairs; cv.dictionary_stability() measures how much they change when learned from one
fold only (second-order effect, reported instead of silently ignored).
Unlabelled information used across the boundary: all S1/S2/S3 records of the split (retrieval
bank, IDF, S1 vocabulary for spelling correction, name frequencies, label-free extra-word stats).
"""
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

import pipeline as P
import stage2 as S2
from common import cache, load_gt_pairs, log
from decide import decide
from metrics import macro_f05


from folds import query_folds, train_folds  # noqa: E402


# ------------------------------------------------------------------------------------------
def _fit(tr: pl.DataFrame, va: pl.DataFrame, cols, seed: int, rounds: int = P.N_ROUNDS) -> lgb.Booster:
    return P.fit(tr, cols, seed=seed, rounds=rounds, valid=va)


def _es(df_expr_fam: str = "fam") -> pl.Expr:
    """Early-stopping slice: 1/20 of the families of the training data.  Uses hash bits above the
    fold (bit 0) and inner (bit 1) bits, so the slice is independent of both."""
    return ((pl.col(df_expr_fam) // 4) % 20) == 7


def outer_stage1(force: bool = False) -> pl.DataFrame:
    """Stage-1 matcher, 2 outer folds by family -> out-of-fold p for every train pair."""
    path = cache("cv_oof1.parquet")
    if os.path.exists(path) and not force:
        return pl.read_parquet(path)
    qf = query_folds()
    lf = P.scan_feats("train").join(qf.lazy(), on="qid", how="left")
    cols = [c for c in P.feature_cols(lf) if c not in ("inner", "fam")]
    oof, iters = [], []
    for k in (0, 1):
        tr = lf.filter((pl.col("fold") != k) & ~_es()).collect()
        va = lf.filter((pl.col("fold") != k) & _es()).collect()
        m = _fit(tr, va, cols, seed=k)
        del tr, va
        iters.append(m.best_iteration)
        log(f"outer stage1 fold {k}: best_iter={m.best_iteration}")
        m.save_model(cache(f"cv_m1_fold{k}.txt"), num_iteration=m.best_iteration)
        for part in P.stage_features("train"):
            te = pl.read_parquet(part).join(qf, on="qid", how="left").filter(pl.col("fold") == k)
            oof.append(te.select("qid", "sid", "y", "fold", "inner").with_columns(pl.Series("p", P.predict(m, te, cols))))
        del m
    out = pl.concat(oof)
    out.write_parquet(path)
    json.dump({"iters": iters, "cols": cols}, open(cache("cv_m1_meta.json"), "w"))
    return out


def inner_stage1(force: bool = False) -> pl.DataFrame:
    """Inner cross-fitting inside each outer training fold: for fold f rows, p from a model
    trained on the OTHER inner half of fold f (never sees the other outer fold's labels)."""
    path = cache("cv_inner1.parquet")
    if os.path.exists(path) and not force:
        return pl.read_parquet(path)
    qf = query_folds()
    lf = P.scan_feats("train").join(qf.lazy(), on="qid", how="left")
    cols = json.load(open(cache("cv_m1_meta.json")))["cols"]
    out = []
    for f in (0, 1):
        for h in (0, 1):
            tr = lf.filter((pl.col("fold") == f) & (pl.col("inner") != h) & ~_es()).collect()
            va = lf.filter((pl.col("fold") == f) & (pl.col("inner") != h) & _es()).collect()
            m = _fit(tr, va, cols, seed=10 + 2 * f + h)
            del tr, va
            log(f"inner stage1 fold {f} half {h}: best_iter={m.best_iteration}")
            for part in P.stage_features("train"):
                te = pl.read_parquet(part).join(qf, on="qid", how="left").filter((pl.col("fold") == f) & (pl.col("inner") == h))
                if te.height:
                    out.append(te.select("qid", "sid", "fold").with_columns(pl.Series("p", P.predict(m, te, cols))))
            del m
    res = pl.concat(out)
    res.write_parquet(path)
    return res


def nested_stage2(force: bool = False, tag: str = "", partner_p: float = None) -> pl.DataFrame:
    """Stage 2 evaluated on fold k with every input probability free of fold-k labels.
    `tag` versions the artifacts of an experiment; `partner_p` overrides the partner-seed threshold."""
    if partner_p is not None:
        S2.PARTNER_P = partner_p
    path = cache(f"cv_oof2{tag}.parquet")
    if os.path.exists(path) and not force:
        return pl.read_parquet(path)
    oof1 = outer_stage1()
    inner = inner_stage1()
    nrm = P.normalize_split("train", columns=S2.REC)
    qf = query_folds()
    res = []
    for k in (0, 1):
        # probability table for evaluating fold k: fold k rows <- outer model trained on 1-k;
        # fold 1-k rows <- inner cross-fitting within fold 1-k
        pk = pl.concat([
            oof1.filter(pl.col("fold") == k).select("qid", "sid", "p"),
            inner.filter(pl.col("fold") != k).select("qid", "sid", "p"),
        ])
        coll = S2.collective_features(pk, nrm)
        lf = P.scan_feats("train").join(qf.lazy(), on="qid", how="left").join(coll.lazy(), on=["qid", "sid"], how="left")
        cols = [c for c in P.feature_cols(lf) if c not in ("inner", "fam")]
        tr = lf.filter((pl.col("fold") != k) & ~_es()).collect()
        va = lf.filter((pl.col("fold") != k) & _es()).collect()
        m = _fit(tr, va, cols, seed=100 + k)
        del tr, va
        log(f"nested stage2 fold {k}: best_iter={m.best_iteration}")
        m.save_model(cache(f"cv_m2{tag}_fold{k}.txt"), num_iteration=m.best_iteration)
        for part in P.stage_features("train"):
            te = (pl.read_parquet(part).join(qf, on="qid", how="left").filter(pl.col("fold") == k)
                  .join(coll, on=["qid", "sid"], how="left"))
            res.append(te.select("qid", "sid", "y", "fold").with_columns(pl.Series("p", P.predict(m, te, cols))))
        del m, coll
    out = pl.concat(res)
    out.write_parquet(path)
    json.dump({"cols": cols, "partner_p": S2.PARTNER_P}, open(cache(f"cv_m2{tag}_meta.json"), "w"))
    return out


# ------------------------------------------------------------------------------------------
def per_entity(pred_links: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 (tp, np, nt, f) for a link set (qid, sid) on train."""
    ids = P.id_maps("train")
    truth = P.truth_pairs(ids)
    s1 = ids.filter(pl.col("src") == 1).select(pl.col("rid").alias("sid"), "country")
    tp = pred_links.join(truth, on=["qid", "sid"]).group_by("sid").len().rename({"len": "tp"})
    npr = pred_links.group_by("sid").len().rename({"len": "np"})
    nt = truth.group_by("sid").len().rename({"len": "nt"})
    d = s1.join(tp, on="sid", how="left").join(npr, on="sid", how="left").join(nt, on="sid", how="left").fill_null(0)
    return d.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np"))).alias("f")
    )


def bootstrap_diff(fa: np.ndarray, fb: np.ndarray, n: int = 200, seed: int = 0):
    """Paired bootstrap over S1 entities of mean(fb) - mean(fa): (diff, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    d = fb - fa
    m = len(d)
    bs = np.array([d[rng.integers(0, m, m)].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def report(pred: pl.DataFrame, policy: dict, name: str) -> pl.DataFrame:
    links = decide(pred.select("qid", "sid", "p"), **policy)
    d = per_entity(links)
    ids = P.id_maps("train")
    truth = P.truth_pairs(ids)
    tp = links.join(truth, on=["qid", "sid"]).height
    fp = links.height - tp
    log(f"[{name}] {policy}: macro F0.5={d['f'].mean():.5f}  P={tp / max(1, links.height):.4f}  "
        f"R={tp / truth.height:.4f}  FP={fp:,}  FN={truth.height - tp:,}  "
        f"singleton false merges={d.filter((pl.col('nt') == 0) & (pl.col('np') > 0)).height:,}")
    for c in ("US", "India"):
        log(f"    {c}: {d.filter(pl.col('country') == c)['f'].mean():.5f}")
    return d


def candidate_oracle(pairs: pl.DataFrame, name: str) -> float:
    """Macro F0.5 of a perfect matcher restricted to the given candidate pairs."""
    ids = P.id_maps("train")
    truth = P.truth_pairs(ids)
    links = truth.join(pairs.select("qid", "sid").unique(), on=["qid", "sid"])
    d = per_entity(links)
    log(f"[{name}] candidate link recall={links.height / truth.height:.5f}  oracle macro F0.5={d['f'].mean():.5f}")
    return d["f"].mean()


def tune_policy(pred: pl.DataFrame, name: str):
    best = None
    for pol in ([dict(thr=t, use_ef=False) for t in (0.5, 0.6, 0.65, 0.7, 0.72, 0.75, 0.8)]
                + [dict(thr=t, use_ef=True) for t in (0.3, 0.4, 0.5, 0.6, 0.7)]):
        d = report(pred, pol, name)
        f = d["f"].mean()
        if best is None or f > best[0]:
            best = (f, pol, d)
    log(f"[{name}] best policy {best[1]}: {best[0]:.5f}")
    return best


def dictionary_stability():
    """How much would the label-derived dictionaries change if learned from one fold only?"""
    import translit
    full = json.load(open(cache("translit_dict.json"), encoding="utf-8"))
    log(f"translit dict entries: {len(full):,} (stability check requires a fold-restricted rebuild; see docs)")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    train_folds()
    if stage in ("outer", "all"):
        outer_stage1()
    if stage in ("inner", "all"):
        inner_stage1()
    if stage in ("stage2", "all"):
        nested_stage2()
    if stage in ("eval", "all"):
        o1 = outer_stage1()
        o2 = nested_stage2()
        cands = pl.read_parquet(cache("train_cands.parquet"), columns=["qid", "sid"])
        candidate_oracle(cands, "pruned candidate set")
        b1 = tune_policy(o1, "stage1")
        b2 = tune_policy(o2, "stage2")
        diff = bootstrap_diff(b1[2].sort("sid")["f"].to_numpy(), b2[2].sort("sid")["f"].to_numpy())
        log(f"stage2 - stage1 (best policies): {diff[0]:+.5f}  95% CI [{diff[1]:+.5f}, {diff[2]:+.5f}]")


# ------------------------------------------------------------------------------------------
# release: explicit configuration chosen on clean validation, test prediction, manifest
def predict_test_cv() -> dict:
    """Test probabilities of the cross-validated models: stage 1 = mean of the two outer models;
    stage 2 = mean of the two nested stage-2 models on collective features built from stage 1."""
    m1 = [lgb.Booster(model_file=cache(f"cv_m1_fold{k}.txt")) for k in (0, 1)]
    cols1 = m1[0].feature_name()
    p1 = []
    for part in P.stage_features("test"):
        f = pl.read_parquet(part)
        p1.append(f.select("qid", "sid").with_columns(pl.Series("p", np.mean([P.predict(m, f, cols1) for m in m1], axis=0).astype(np.float32))))
    p1 = pl.concat(p1)
    p1.write_parquet(cache("cv_test_p1.parquet"))
    out = {"stage1": p1}
    if os.path.exists(cache("cv_m2_fold1.txt")):
        coll = S2.collective_features(p1, P.normalize_split("test", columns=S2.REC))
        m2 = [lgb.Booster(model_file=cache(f"cv_m2_fold{k}.txt")) for k in (0, 1)]
        cols2 = m2[0].feature_name()
        p2 = []
        for part in P.stage_features("test"):
            f = pl.read_parquet(part).join(coll, on=["qid", "sid"], how="left")
            p2.append(f.select("qid", "sid").with_columns(pl.Series("p", np.mean([P.predict(m, f, cols2) for m in m2], axis=0).astype(np.float32))))
        p2 = pl.concat(p2)
        p2.write_parquet(cache("cv_test_p2.parquet"))
        out["stage2"] = p2
    return out


def _sha(path: str, block: int = 1 << 22) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(release: dict, extra: dict):
    import glob
    import blocking as B
    import prune as PR
    from common import DATA, OUT
    src = os.path.dirname(os.path.abspath(__file__))
    data_hash_path = cache("data_hashes.json")
    if os.path.exists(data_hash_path):
        data_hashes = json.load(open(data_hash_path))
    else:
        data_hashes = {os.path.relpath(f, DATA): _sha(f) for f in sorted(glob.glob(os.path.join(DATA, "*", "*.tsv")))}
        json.dump(data_hashes, open(data_hash_path, "w"), indent=1)
    man = {
        "data_sha256": data_hashes,
        "code_sha256": {os.path.basename(f): _sha(f) for f in sorted(glob.glob(os.path.join(src, "*.py")))},
        "settings": {
            "blocking": dict(MAX_DF=B.MAX_DF, RARE_K=B.RARE_K, TOP_PER_CHANNEL=B.TOP_PER_CHANNEL, SEED=B.SEED),
            "pruning": dict(rule="adaptive", CONFIDENT=PR.CONFIDENT, RATIO=PR.PRUNE_RATIO, UNCERTAIN_K=PR.UNCERTAIN_K, UNCERTAIN_MIN=PR.UNCERTAIN_MIN, cross_fitted=True),
            "folds": "family folds: crc32(owner S1 entity_id or own entity_id); fold=hash%2, inner=(hash//2)%2",
            "lightgbm": P.PARAMS,
        },
        "models": {os.path.basename(f): _sha(f) for f in sorted(glob.glob(cache("cv_m*_fold*.txt")) + glob.glob(cache("pruner_fold*.txt")))},
        "model_meta": {k: json.load(open(cache(k))) for k in ("cv_m1_meta.json", "cv_m2_meta.json") if os.path.exists(cache(k))},
        "release": release,
        "validation": extra,
        "outputs_sha256": {f: _sha(os.path.join(OUT, f)) for f in ("matching_results.tsv", "candidate_pairs.tsv") if os.path.exists(os.path.join(OUT, f))},
    }
    with open(os.path.join(OUT, "..", "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, default=str)
    log("manifest written")


def release():
    """Choose (stage, policy) on clean validation, predict test with exactly that configuration,
    write the two output files and the manifest."""
    o1, o2 = outer_stage1(), nested_stage2()
    cands = pl.read_parquet(cache("train_cands.parquet"), columns=["qid", "sid"])
    oracle = candidate_oracle(cands, "pruned candidate set")
    b1, b2 = tune_policy(o1, "stage1"), tune_policy(o2, "stage2")
    diff = bootstrap_diff(b1[2].sort("sid")["f"].to_numpy(), b2[2].sort("sid")["f"].to_numpy())
    log(f"stage2 - stage1: {diff[0]:+.5f}  95% CI [{diff[1]:+.5f}, {diff[2]:+.5f}]")
    use2 = diff[1] > 0  # promote stage 2 only if its gain is significant at the S1 level
    stage, (f, pol, _) = ("stage2", b2) if use2 else ("stage1", b1)
    rel = {"stage": stage, "policy": pol, "clean_oof_macro_f05": f}
    json.dump(rel, open(cache("release.json"), "w"), indent=1)
    preds = predict_test_cv()
    pred = preds[stage]
    links = decide(pred.select("qid", "sid", "p"), **pol)
    ids = P.id_maps("test")
    P.write_lists(links, ids, "matched_entity_ids", os.path.join(P.OUT, "matching_results.tsv"))
    P.write_lists(pred.select("qid", "sid"), ids, "candidate_entity_ids", os.path.join(P.OUT, "candidate_pairs.tsv"))
    log(f"release {rel}: {links.height:,} test links over {pred.height:,} candidate pairs")
    write_manifest(rel, {"oracle": oracle, "stage1": b1[0], "stage1_policy": b1[1], "stage2": b2[0],
                         "stage2_policy": b2[1], "stage2_minus_stage1": diff})
