"""Learned address-word aliases (training ground truth only).

Some address variations are real words, so the spelling corrector (which only touches
out-of-vocabulary words) cannot fix them: street-type typos that happen to be words ("dive",
"dove", "davie" for "drive"; "read", "rand" for "road"), abbreviations ("atl" for "atlanta"), and
the generator's city/suburb swaps ("dorchester" for "boston", "calcuta" for "kolkata").

From the training pairs we take every pair whose addresses differ by exactly one word on each side,
count the substitutions (S2/S3 word -> S1 word) and keep a mapping when it is frequent
(>= MIN_COUNT), dominant for that S2/S3 word (>= MIN_SHARE of its substitutions) and the S2/S3 word
is usually NOT matched literally (unmatched in >= MIN_UNMATCHED of the pairs where it occurs).
The map is applied to the S2/S3 side only (the S1 side is the reference); blocking keys and
features then see the S1 spelling.
"""
import json
import os
import sys

import polars as pl

from common import cache, load_gt_pairs, log

MIN_COUNT = 30
MIN_SHARE = 0.5
MIN_UNMATCHED = 0.3
PATH = cache("addr_alias.json")
META = cache("addr_alias_meta.json")  # countries the aliases were learned from


def learn():
    nrm = pl.read_parquet(cache("train_norm.parquet"), columns=["rid", "entity_id", "src", "a_words_c", "a_words"])
    q = nrm.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("m_id"), pl.col("a_words_c").alias("qw"))
    s = nrm.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1_id"), pl.col("a_words").alias("sw"))
    m = load_gt_pairs().join(q, on="m_id").join(s, on="s1_id")
    m = m.select(
        pl.col("qw").str.split(" ").list.unique().alias("qt"),
        pl.col("qw").str.split(" ").list.set_difference(pl.col("sw").str.split(" ")).alias("qo"),
        pl.col("sw").str.split(" ").list.set_difference(pl.col("qw").str.split(" ")).alias("so"),
    )
    appear = m.select(pl.col("qt").explode()).group_by("qt").len().rename({"len": "n_app"})
    unmatched = m.select(pl.col("qo").explode().alias("qt")).group_by("qt").len().rename({"len": "n_unm"})
    one = m.filter((pl.col("qo").list.len() == 1) & (pl.col("so").list.len() == 1)).select(
        pl.col("qo").list.first().alias("qt"), pl.col("so").list.first().alias("st"))
    sub = one.group_by("qt", "st").len().rename({"len": "n"})
    tot = sub.group_by("qt").agg(pl.col("n").sum().alias("n_sub"))
    best = sub.sort("n", descending=True).group_by("qt").head(1)
    best = best.join(tot, on="qt").join(unmatched, on="qt").join(appear, on="qt")
    keep = best.filter(
        (pl.col("n") >= MIN_COUNT) & (pl.col("n") / pl.col("n_sub") >= MIN_SHARE)
        & (pl.col("n_unm") / pl.col("n_app") >= MIN_UNMATCHED) & (pl.col("qt") != "") & (pl.col("st") != "")
    ).sort("n", descending=True)
    mp = dict(zip(keep["qt"].to_list(), keep["st"].to_list()))
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(mp, f)
    countries = sorted(nrm.join(pl.read_parquet(cache("train_norm.parquet"), columns=["rid", "country"]), on="rid")["country"].unique().to_list())
    with open(META, "w", encoding="utf-8") as f:
        json.dump({"countries": countries}, f)
    log(f"address aliases: {len(mp)} learned; top: {list(mp.items())[:25]}")
    return mp


def _map_tokens(col: pl.Expr, mp: dict) -> pl.Expr:
    return col.str.split(" ").list.eval(pl.element().replace(mp)).list.join(" ")


def apply(split: str):
    if not os.path.exists(PATH):
        return
    with open(PATH, encoding="utf-8") as f:
        mp = json.load(f)
    # aliases are learned from US/India training pairs.  Many are generic typo fixes ("avnue" ->
    # "ave", "bouleard" -> "blvd"), others are US word swaps ("maison" -> "madison", "cour" -> "ct")
    # that corrupt real words of another language.  Countries the map was learned on get the full
    # map; any other country only the entries whose source word is NOT in its S1 address vocabulary
    # and whose target IS (label-free: typo fixes pass, "cour"/"maison"/"pont" are left alone).
    countries = json.load(open(META, encoding="utf-8"))["countries"] if os.path.exists(META) else ["US", "India"]
    p = cache(f"{split}_norm.parquet")
    df = pl.read_parquet(p)
    parts = []
    for (country,), d in df.group_by("country", maintain_order=True):
        cmp = mp
        if country not in countries:
            vocab = set(d.filter(pl.col("src") == 1)["a_words"].str.split(" ").explode().drop_nulls().to_list())
            cmp = {k: v for k, v in mp.items() if k not in vocab and v in vocab}
            log(f"{split}/{country}: {len(cmp)} of {len(mp)} aliases pass the vocabulary gate")
        isq = pl.col("src") != 1
        parts.append(d.with_columns(
            pl.when(isq).then(_map_tokens(pl.col("a_words_c"), cmp)).otherwise(pl.col("a_words_c")).alias("a_words_c"),
            pl.when(isq).then(_map_tokens(pl.col("a_full_c"), cmp)).otherwise(pl.col("a_full_c")).alias("a_full_c"),
        ))
    df = pl.concat(parts).sort("rid")
    df.write_parquet(p)
    log(f"{split}: address aliases applied (full map for {countries}, vocabulary-gated elsewhere)")


if __name__ == "__main__":
    learn()
    for split in sys.argv[1:] or ["train", "test"]:
        apply(split)
