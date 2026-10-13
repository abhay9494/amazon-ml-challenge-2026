"""Learn a native-script -> Latin token dictionary from the TRAINING ground truth only.

About a quarter of Indian S2/S3 names are written in an Indic script (Devanagari, Tamil, Telugu,
Kannada, Bengali, Gujarati, Malayalam, Odia, Gurmukhi).  unidecode gives only a rough phonetic
romanisation ("सर्विसेज" -> "srvisej" for "services"), so we learn the exact mapping from the
training pairs: when a matched S2/S3 name and its S1 name have the same number of tokens, the
tokens are aligned by position and every native token votes for the Latin token at its position.
Mappings with support >= 2 and purity >= 0.5 are kept (the majority spelling beats unidecode).  The vocabulary is small (~1.5k native
types) and identical between train and test, so coverage on test is ~95%; the rest falls back to
unidecode.  No external data is used.
"""
import json
import re
from collections import Counter, defaultdict

import polars as pl

from common import cache, load_gt_pairs, log, read_tsv, source_path

ZW = re.compile(r"[​-‍⁠﻿]")
TOK = re.compile(r"[0-9A-Za-zÀ-ɏऀ-ॣ०-ൿ]+")
NATIVE = re.compile(r"[ऀ-ൿ]")


def toks(s: str):
    return TOK.findall(ZW.sub("", s or "").lower())


def build(min_support: int = 2, min_purity: float = 0.5) -> dict:
    s1 = read_tsv(source_path("train", 1)).select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s1n"))
    q = pl.concat([read_tsv(source_path("train", k)) for k in (2, 3)]).select(
        pl.col("entity_id").alias("m_id"), pl.col("business_name").alias("qn")
    )
    q = q.filter(pl.col("qn").str.contains(r"[ऀ-ൿ]"))
    pairs = load_gt_pairs().join(q, on="m_id").join(s1, on="s1_id")
    log(f"translit: {pairs.height:,} native-script training pairs")
    cnt = defaultdict(Counter)
    for qn, s1n in zip(pairs["qn"].to_list(), pairs["s1n"].to_list()):
        a, b = toks(qn), toks(s1n)
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            if NATIVE.search(x) and not NATIVE.search(y):
                cnt[x][y] += 1
    d = {}
    for x, c in cnt.items():
        y, k = c.most_common(1)[0]
        tot = sum(c.values())
        if tot >= min_support and k / tot >= min_purity:
            d[x] = y
    log(f"translit: {len(cnt):,} native types seen, {len(d):,} kept")
    with open(cache("translit_dict.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return d


if __name__ == "__main__":
    build()
