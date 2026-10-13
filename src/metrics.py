"""Macro F0.5 exactly as the organisers compute it (per S1 entity, singletons included)."""
import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5) -> dict:
    """pred / truth: DataFrames with columns (s1_id, m_id).  s1_ids: every S1 entity evaluated.
    An entity with no true and no predicted match scores 1; with exactly one side empty scores 0."""
    b2 = beta * beta
    base = pl.DataFrame({"s1_id": s1_ids.unique()})
    pred = pred.select("s1_id", "m_id").unique().join(base, on="s1_id", how="semi")
    truth = truth.select("s1_id", "m_id").unique().join(base, on="s1_id", how="semi")
    tp = pred.join(truth, on=["s1_id", "m_id"]).group_by("s1_id").len().rename({"len": "tp"})
    npred = pred.group_by("s1_id").len().rename({"len": "np"})
    ntrue = truth.group_by("s1_id").len().rename({"len": "nt"})
    d = (
        base.join(tp, on="s1_id", how="left")
        .join(npred, on="s1_id", how="left")
        .join(ntrue, on="s1_id", how="left")
        .fill_null(0)
    )
    d = d.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0))
        .then(1.0)
        .when((pl.col("np") == 0) | (pl.col("nt") == 0) | (pl.col("tp") == 0))
        .then(0.0)
        .otherwise(
            (1 + b2) * pl.col("tp") / ((1 + b2) * pl.col("tp") + b2 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp")))
        )
        .alias("f")
    )
    tp_all = d["tp"].sum()
    return {
        "macro_f05": d["f"].mean(),
        "micro_precision": tp_all / max(1, d["np"].sum()),
        "micro_recall": tp_all / max(1, d["nt"].sum()),
        "singleton_acc": d.filter(pl.col("nt") == 0).select((pl.col("np") == 0).mean()).item()
        if (d["nt"] == 0).any()
        else float("nan"),
        "n_entities": d.height,
    }
