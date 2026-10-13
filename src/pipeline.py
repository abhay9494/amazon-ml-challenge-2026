"""End-to-end pipeline.

  python pipeline.py all        # train everything on train, validate (OOF), predict test, write outputs
  python pipeline.py <stage>    # retrieval | prune | features | match | predict

Stages cache their results under cache/ so each can be iterated on separately.
"""
import glob
import os
import sys
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl

from candidates import retrieve_split
from common import OUT, cache, load_gt_pairs, log
from decide import decide
from features import REC_COLS, extra_token_stats, name_uniqueness, pair_features, token_idf
from metrics import macro_f05
from normalize import normalize_split
import prune as PR
import stage2 as S2
from folds import query_folds

QUERIES_PER_PART = 500_000


# ------------------------------------------------------------------------------------------
def id_maps(split: str):
    return normalize_split(split, columns=["rid", "entity_id", "src", "country"])


_TRUTH = {}


def truth_pairs(ids: pl.DataFrame) -> pl.DataFrame:
    """Ground-truth pairs as integer (qid, sid) row ids (cached per process)."""
    key = ids.height
    if key not in _TRUTH:
        q = ids.select(pl.col("rid").alias("qid"), pl.col("entity_id").alias("m_id"))
        s = ids.select(pl.col("rid").alias("sid"), pl.col("entity_id").alias("s1_id"))
        _TRUTH[key] = load_gt_pairs().join(q, on="m_id").join(s, on="s1_id").select("qid", "sid")
    return _TRUTH[key]


def add_labels(pairs: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    t = truth_pairs(ids).with_columns(pl.lit(1, pl.Int8).alias("y"))
    return pairs.join(t, on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(0))


# ------------------------------------------------------------------------------------------
def _with_pr_other(ret: pl.DataFrame) -> pl.DataFrame:
    """Best competing pruner score of every pair (within the full retrieved set of its query)."""
    ret = ret.with_columns(pl.col("p_prune").rank("ordinal", descending=True).over("qid").alias("_r"))
    top2 = ret.filter(pl.col("_r") <= 2).group_by("qid").agg(
        pl.col("p_prune").filter(pl.col("_r") == 1).first().alias("_p1"),
        pl.col("p_prune").filter(pl.col("_r") == 2).first().alias("_p2"),
    )
    return ret.join(top2, on="qid", how="left").with_columns(
        pl.when(pl.col("_r") == 1).then(pl.col("_p2")).otherwise(pl.col("_p1")).fill_null(0).alias("pr_other"),
    ).drop("_r", "_p1", "_p2")


def stage_prune(split: str, force: bool = False) -> pl.DataFrame:
    """Learned pruning, CROSS-FITTED on train: the pruner trained on fold 1-k scores the queries of
    fold k (folds by entity family, see folds.py); test uses the mean of both pruners.  Every
    retrieved pair with p_prune >= 0.001 is also saved (retention-rule analysis)."""
    p = cache(f"{split}_cands.parquet")
    if os.path.exists(p) and not force:
        return pl.read_parquet(p)
    parts = retrieve_split(split)
    ids = id_maps(split) if split == "train" else None
    mpaths = [cache(f"pruner_fold{k}.txt") for k in (0, 1)]
    if split == "train":
        qf = query_folds().select("qid", "fold")
        rng = np.random.default_rng(0)
        sample = rng.choice(len(parts), size=min(len(parts), 6), replace=False)
        tr = pl.concat([add_labels(PR.context(pl.read_parquet(parts[i])), ids).join(qf, on="qid", how="left") for i in sorted(sample)])
        models = []
        for k in (0, 1):
            m = PR.train_pruner(tr.filter(pl.col("fold") != k), sample_frac=0.5, seed=k)
            m.save_model(mpaths[k])
            models.append(m)
        del tr
    else:
        models = [lgb.Booster(model_file=mp) for mp in mpaths]
    kept, scores, n_ret, n_pos_ret = [], [], 0, 0
    for part in parts:
        ret = PR.context(pl.read_parquet(part))
        if ids is not None:
            ret = add_labels(ret, ids).join(qf, on="qid", how="left")
            n_pos_ret += int(ret["y"].sum())
            s0, s1 = PR.score(ret, models[0]), PR.score(ret, models[1])
            pp = np.where(ret["fold"].to_numpy() == 0, s0, s1)  # model k was trained on fold != k
            ret = ret.drop("fold")
        else:
            pp = (PR.score(ret, models[0]) + PR.score(ret, models[1])) / 2
        n_ret += ret.height
        ret = _with_pr_other(ret.with_columns(pl.Series("p_prune", pp.astype(np.float32))))
        scores.append(ret.filter(pl.col("p_prune") >= 0.001).select("qid", "sid", "p_prune", *(["y"] if ids is not None else [])))
        kept.append(PR.prune(ret))
    cands = pl.concat(kept)
    cands.write_parquet(p)
    pl.concat(scores).write_parquet(cache(f"{split}_prune_scores.parquet"))
    log(f"{split}: retrieved {n_ret:,} -> kept {cands.height:,} pairs")
    if ids is not None:
        n_true = load_gt_pairs().height
        log(f"  true pairs {n_true:,}: retrieval recall {n_pos_ret / n_true:.4f}, pruned recall {cands['y'].sum() / n_true:.4f}")
    return cands


def feats_dir(split: str) -> str:
    return cache(f"{split}_feats")


def stage_features(split: str, force: bool = False):
    """Rich pair features for the pruned candidates, written as parquet parts (whole queries per
    part, so that per-query competitor features are exact).  Returns the part paths."""
    d = feats_dir(split)
    done = os.path.join(d, "_DONE")
    if os.path.exists(done) and not force:
        return sorted(glob.glob(os.path.join(d, "part_*.parquet")))
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "*")):
        os.remove(f)
    cands = stage_prune(split)
    keep = ["qid", "sid", "pr_rank", *PR.FEATS, "p_prune", "pr_other"] + (["y"] if "y" in cands.columns else [])
    cands = cands.select(keep)
    df = normalize_split(split, columns=REC_COLS + ["src"])
    s1 = df.filter(pl.col("src") == 1)
    q = df.filter(pl.col("src") != 1).join(cands.select(pl.col("qid").alias("rid")).unique(), on="rid", how="semi")
    del df
    idf = token_idf(s1)
    xt = extra_token_stats(cands, q, s1)
    xt.write_parquet(cache(f"{split}_extra_tokens.parquet"))
    src = q.select(pl.col("rid").alias("qid"), (pl.col("src") == 3).cast(pl.Int8).alias("q_src3"))
    cands = cands.with_columns(((pl.col("qid") - pl.col("qid").min()) // QUERIES_PER_PART).alias("_part"))
    for j, c in enumerate(cands.partition_by("_part", maintain_order=True)):
        f = pair_features(c.select("qid", "sid"), q, s1, idf, xt)
        extra = c.select(
            *PR.FEATS, "p_prune", "pr_other",
            (pl.col("p_prune") - pl.col("pr_other")).alias("pr_margin"),
            *(["y"] if "y" in c.columns else []),
        )
        uniq = name_uniqueness(c.select("qid", "sid"), q, s1)
        f = pl.concat([f.drop("pid"), extra, uniq], how="horizontal").join(src, on="qid", how="left")
        f = competitor_features(f)
        f = f.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        f.write_parquet(os.path.join(d, f"part_{j:04d}.parquet"))
        log(f"  features {split} part {j}: {f.height:,} pairs")
    open(done, "w").close()
    return sorted(glob.glob(os.path.join(d, "part_*.parquet")))


def load_feats(split: str, columns=None) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p, columns=columns) for p in stage_features(split)])


COMP = ["n_best_tset", "a_tset", "n_idf_cov_s", "a_idf_cov_s", "h_first_eq"]


def competitor_features(feats: pl.DataFrame) -> pl.DataFrame:
    """For each pair, the best value of a few key similarities among the OTHER candidates of the
    same query (-1 if the query has a single candidate).  top-2 per query + one join."""
    top = feats.group_by("qid").agg(
        pl.len().alias("_n"),
        *[pl.col(c).cast(pl.Float32).top_k(2).sort(descending=True).alias(f"_t_{c}") for c in COMP],
    )
    out = feats.join(top, on="qid", how="left")
    exprs = []
    for c in COMP:
        m1 = pl.col(f"_t_{c}").list.get(0, null_on_oob=True)
        m2 = pl.col(f"_t_{c}").list.get(1, null_on_oob=True)
        exprs.append(
            pl.when(pl.col("_n") == 1).then(-1.0)
            .when(pl.col(c).cast(pl.Float32) >= m1).then(m2)
            .otherwise(m1)
            .cast(pl.Float32).alias(f"o_{c}")
        )
    return out.with_columns(exprs).drop("_n", *[f"_t_{c}" for c in COMP])


# ------------------------------------------------------------------------------------------
PARAMS = dict(
    # capacity chosen on clean validation (v6): 511 leaves / lr 0.05 beat 255 / 0.08 by
    # +0.00006..+0.00010 macro F0.5 on held-out families (paired bootstrap CI above zero)
    objective="binary", learning_rate=0.05, num_leaves=511, min_data_in_leaf=50,
    feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=2.0,
    verbose=-1, num_threads=0,
)
N_ROUNDS = 4000
MAX_TRAIN_ROWS = 6_000_000


def feature_cols(df) -> list:
    names = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    return [c for c in names if c not in ("qid", "sid", "y", "fold", "p")]


def fold_expr() -> pl.Expr:
    """2 folds by S1 entity (stable hash of the S1 row id)."""
    return (pl.col("sid").hash(seed=11) % 2).cast(pl.Int8)


def scan_feats(split: str) -> pl.LazyFrame:
    stage_features(split)
    return pl.scan_parquet(os.path.join(feats_dir(split), "part_*.parquet"))


def _xy(df: pl.DataFrame, cols):
    return df.select(cols).to_numpy().astype(np.float32), df["y"].to_numpy()


def fit(tr: pl.DataFrame, cols, seed=0, rounds=N_ROUNDS, valid=None):
    if tr.height > MAX_TRAIN_ROWS:
        tr = tr.sample(MAX_TRAIN_ROWS, seed=seed)
    X, y = _xy(tr, cols)
    del tr
    ds = lgb.Dataset(X, y, feature_name=cols, free_raw_data=True)
    params = dict(PARAMS, seed=seed)
    cb = [lgb.log_evaluation(100)]
    vs = []
    if valid is not None:
        Xv, yv = _xy(valid, cols)
        vs = [lgb.Dataset(Xv, yv, reference=ds)]
        cb.append(lgb.early_stopping(60, verbose=False))
    return lgb.train(params, ds, num_boost_round=rounds, valid_sets=vs, callbacks=cb)


def predict(model, df: pl.DataFrame, cols, chunk=2_000_000) -> np.ndarray:
    return np.concatenate([
        model.predict(df.slice(i, chunk).select(cols).to_numpy().astype(np.float32)) for i in range(0, df.height, chunk)
    ]).astype(np.float32)


def stage_match(force: bool = False):
    """2-fold cross-fitting on train (folds by S1 entity) -> out-of-fold probabilities."""
    p_oof = cache("train_oof.parquet")
    ids = id_maps("train")
    if not os.path.exists(p_oof) or force:
        lf = scan_feats("train").with_columns(fold_expr().alias("fold"))
        cols = feature_cols(lf)
        es = (pl.col("sid").hash(seed=7) % 20) == 0  # early-stopping slice, by S1 entity
        oof, iters = [], []
        for k in (0, 1):
            tr = lf.filter((pl.col("fold") != k) & ~es).collect()
            va = lf.filter((pl.col("fold") != k) & es).collect()
            m = fit(tr, cols, seed=k, valid=va)
            del tr, va
            iters.append(m.best_iteration)
            log(f"fold {k}: best_iter={m.best_iteration}")
            m.save_model(cache(f"matcher_fold{k}.txt"), num_iteration=m.best_iteration)
            for part in stage_features("train"):
                te = pl.read_parquet(part).with_columns(fold_expr().alias("fold")).filter(pl.col("fold") == k)
                oof.append(te.select("qid", "sid", "y").with_columns(pl.Series("p", predict(m, te, cols))))
            del m
        pl.concat(oof).write_parquet(p_oof)
        with open(cache("best_iters.txt"), "w") as f:
            f.write(" ".join(map(str, iters)))
    oof = pl.read_parquet(p_oof)
    evaluate(oof, ids)
    return oof


def evaluate(oof: pl.DataFrame, ids: pl.DataFrame, thrs=(0.5, 0.6, 0.7, 0.8, 0.9)):
    s1_ids = ids.filter(pl.col("src") == 1)["entity_id"]
    truth = load_gt_pairs()
    q = ids.select(pl.col("rid").alias("qid"), pl.col("entity_id").alias("m_id"))
    s = ids.select(pl.col("rid").alias("sid"), pl.col("entity_id").alias("s1_id"))
    res = {}
    for use_ef in (False, True):
        for thr in thrs:
            links = decide(oof.select("qid", "sid", "p"), thr=thr, use_ef=use_ef)
            pred = links.join(q, on="qid").join(s, on="sid").select("s1_id", "m_id")
            r = macro_f05(pred, truth, s1_ids)
            res[(use_ef, thr)] = r
            log(f"  ef={use_ef} thr={thr:.2f}  F0.5={r['macro_f05']:.5f}  P={r['micro_precision']:.4f} "
                f"R={r['micro_recall']:.4f} single={r['singleton_acc']:.4f}")
    return res


# ------------------------------------------------------------------------------------------
DECISION = dict(thr=0.5, use_ef=True)   # tuned on stage-2 OOF (expected-F0.5 rule; flat for thr 0.3-0.6)


def write_lists(pairs: pl.DataFrame, ids: pl.DataFrame, col: str, path: str):
    """pairs: (qid, sid).  One row per S1 entity (all of them), comma-joined S2/S3 ids."""
    s1 = ids.filter(pl.col("src") == 1).select(pl.col("rid").alias("sid"), pl.col("entity_id").alias("source1_entity_id"))
    q = ids.filter(pl.col("src") != 1).select(pl.col("rid").alias("qid"), pl.col("entity_id").alias("m"))
    lists = (
        pairs.select("qid", "sid").unique().join(q, on="qid").sort(["sid", "m"])
        .group_by("sid", maintain_order=True).agg(pl.col("m").str.join(",").alias(col))
    )
    out = s1.join(lists, on="sid", how="left").with_columns(pl.col(col).fill_null("")).select("source1_entity_id", col)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
    log(f"wrote {path}: {out.height:,} rows")


def predict_stage1_test(force: bool = False) -> pl.DataFrame:
    """Stage-1 test probabilities = mean of the two cross-fitted fold models."""
    path = cache("test_pred.parquet")
    if os.path.exists(path) and not force:
        return pl.read_parquet(path)
    models = [lgb.Booster(model_file=cache(f"matcher_fold{k}.txt")) for k in (0, 1)]
    cols = models[0].feature_name()
    preds = []
    for part in stage_features("test"):
        f = pl.read_parquet(part)
        p = np.mean([predict(m, f, cols) for m in models], axis=0)
        preds.append(f.select("qid", "sid").with_columns(pl.Series("p", p.astype(np.float32))))
    out = pl.concat(preds)
    out.write_parquet(path)
    return out


def collective(split: str, force: bool = False) -> pl.DataFrame:
    path = cache(f"{split}_coll.parquet")
    if os.path.exists(path) and not force:
        return pl.read_parquet(path)
    pred = pl.read_parquet(cache("train_oof.parquet")).select("qid", "sid", "p") if split == "train" else predict_stage1_test()
    coll = S2.collective_features(pred.select("qid", "sid", "p"), normalize_split(split, columns=S2.REC))
    coll.write_parquet(path)
    log(f"{split}: collective features {coll.shape}")
    return coll


def stage_match2(force: bool = False):
    """Stage 2: stage-1 features + collective features, 2-fold cross-fitted by S1 entity."""
    p_oof = cache("train_oof2.parquet")
    ids = id_maps("train")
    if not os.path.exists(p_oof) or force:
        coll = collective("train", force=force)
        lf = scan_feats("train").join(coll.lazy(), on=["qid", "sid"], how="left").with_columns(fold_expr().alias("fold"))
        cols = feature_cols(lf)
        es = (pl.col("sid").hash(seed=7) % 20) == 0
        oof, iters = [], []
        for k in (0, 1):
            tr = lf.filter((pl.col("fold") != k) & ~es).collect()
            va = lf.filter((pl.col("fold") != k) & es).collect()
            m = fit(tr, cols, seed=100 + k, valid=va)
            del tr, va
            iters.append(m.best_iteration)
            log(f"stage2 fold {k}: best_iter={m.best_iteration}")
            m.save_model(cache(f"matcher2_fold{k}.txt"), num_iteration=m.best_iteration)
            for part in stage_features("train"):
                te = (pl.read_parquet(part).join(coll, on=["qid", "sid"], how="left")
                      .with_columns(fold_expr().alias("fold")).filter(pl.col("fold") == k))
                oof.append(te.select("qid", "sid", "y").with_columns(pl.Series("p", predict(m, te, cols))))
            del m
        pl.concat(oof).write_parquet(p_oof)
    oof = pl.read_parquet(p_oof)
    evaluate(oof, ids)
    return oof


def stage_predict(force: bool = False, use_stage2: bool = True):
    """Final test probabilities (stage 2 if its models exist), decision rule, output files."""
    if use_stage2 and os.path.exists(cache("matcher2_fold1.txt")):
        coll = collective("test", force=force)
        models = [lgb.Booster(model_file=cache(f"matcher2_fold{k}.txt")) for k in (0, 1)]
        cols = models[0].feature_name()
        preds = []
        for part in stage_features("test"):
            f = pl.read_parquet(part).join(coll, on=["qid", "sid"], how="left")
            p = np.mean([predict(m, f, cols) for m in models], axis=0)
            preds.append(f.select("qid", "sid").with_columns(pl.Series("p", p.astype(np.float32))))
        feats = pl.concat(preds)
        feats.write_parquet(cache("test_pred2.parquet"))
        log("test: using stage-2 probabilities")
    else:
        feats = predict_stage1_test(force=force)
    links = decide(feats.select("qid", "sid", "p"), **DECISION)
    ids = id_maps("test")
    write_lists(links, ids, "matched_entity_ids", os.path.join(OUT, "matching_results.tsv"))
    write_lists(feats.select("qid", "sid"), ids, "candidate_entity_ids", os.path.join(OUT, "candidate_pairs.tsv"))
    n_s1 = ids.filter(pl.col("src") == 1).height
    log(f"test: {feats.height:,} candidate pairs ({feats.height / n_s1:.2f} per S1), {links.height:,} links")


def finalize(pred_file: str = "test_pred2.parquet"):
    """Re-apply the decision rule to saved test probabilities and rewrite the two output files."""
    feats = pl.read_parquet(cache(pred_file))
    links = decide(feats.select("qid", "sid", "p"), **DECISION)
    ids = id_maps("test")
    write_lists(links, ids, "matched_entity_ids", os.path.join(OUT, "matching_results.tsv"))
    write_lists(feats.select("qid", "sid"), ids, "candidate_entity_ids", os.path.join(OUT, "candidate_pairs.tsv"))
    log(f"test: {feats.height:,} candidate pairs, {links.height:,} links ({pred_file}, {DECISION})")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    force = "--force" in sys.argv
    if stage in ("prune", "all"):
        stage_prune("train", force=force)
        stage_prune("test", force=force)
    if stage in ("features", "all"):
        stage_features("train", force=force)
        stage_features("test", force=force)
    if stage in ("match", "all"):
        stage_match(force=force)
    if stage in ("match2", "all"):
        predict_stage1_test(force=force)
        stage_match2(force=force)
    if stage in ("predict", "all"):
        stage_predict(force=force)
