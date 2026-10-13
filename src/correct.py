"""Vocabulary-based spelling correction of S2/S3 name and address words.

S1 is the clean reference: its words form, per country, the vocabulary of correct spellings.
S2/S3 words that are NOT in that vocabulary (typos: "Deelopmett", "Albaany", "Trcay", leftover
transliteration variants ...) are mapped to the most frequent S1 word within a small
Damerau-Levenshtein distance (1; 2 for name words longer than 6 letters), found with a SymSpell-style
deletion index.  Only the split's own S1 records are used (label-free, so it also builds a French
vocabulary from the French test S1 records).

Blocking can only gain from it: an out-of-vocabulary token can never hit an S1 key, while its
correction can.  Features are computed on both the original and the corrected strings.

Adds to {split}_norm.parquet: n_core_c, a_words_c, a_full_c (corrected), n_corr, a_corr (counts).
"""
import os
import sys
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz.distance import DamerauLevenshtein

from common import cache, log

MIN_LEN = 4


def _deletes(w: str, depth: int):
    out = {w}
    frontier = {w}
    for _ in range(depth):
        nxt = set()
        for x in frontier:
            for i in range(len(x)):
                nxt.add(x[:i] + x[i + 1:])
        out |= nxt
        frontier = nxt
    return out


class Corrector:
    def __init__(self, vocab: Counter, long_dist: int = 2):
        self.freq = vocab
        self.long_dist = long_dist
        self.index = defaultdict(list)
        for w in vocab:
            if len(w) < MIN_LEN - 1:
                continue
            for d in _deletes(w, self._max_dist(len(w))):
                self.index[d].append(w)

    def _max_dist(self, n: int) -> int:
        return 1 if n <= 6 else self.long_dist

    def correct(self, t: str):
        md = self._max_dist(len(t))
        cands = set()
        for d in _deletes(t, md):
            cands.update(self.index.get(d, ()))
        best, best_key = None, None
        for c in cands:
            if abs(len(c) - len(t)) > md:
                continue
            dist = DamerauLevenshtein.distance(t, c, score_cutoff=md)
            if dist > md:
                continue
            key = (dist, -self.freq[c], c)
            if best_key is None or key < best_key:
                best, best_key = c, key
        return best


def _vocab(s1: pl.DataFrame, col: str) -> Counter:
    v = s1.select(pl.col(col).str.split(" ").alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= MIN_LEN - 1)
    vc = v.group_by("t").len()
    return Counter(dict(zip(vc["t"].to_list(), vc["len"].to_list())))


def _oov_types(q: pl.DataFrame, col: str, vocab: Counter) -> list:
    t = q.select(pl.col(col).str.split(" ").alias("t")).explode("t").unique()
    t = t.filter((pl.col("t").str.len_chars() >= MIN_LEN) & pl.col("t").str.contains(r"^[a-z]+$"))
    return [x for x in t["t"].to_list() if x not in vocab]


def _apply(df: pl.DataFrame, col: str, mapping: dict, out: str) -> pl.DataFrame:
    if not mapping:
        return df.with_columns(pl.col(col).alias(out))
    m = pl.DataFrame({"_k": list(mapping.keys()), "_v": list(mapping.values())})
    ex = df.select("rid", pl.col(col).str.split(" ").alias("_t")).with_row_index("_i").explode("_t")
    ex = ex.with_row_index("_j").join(m, left_on="_t", right_on="_k", how="left").sort("_j")
    ex = ex.with_columns(pl.coalesce("_v", "_t").alias("_t"))
    agg = ex.group_by("_i", maintain_order=True).agg(pl.col("_t").str.join(" ").alias(out))
    return df.with_row_index("_i").join(agg, on="_i", how="left", maintain_order="left").drop("_i")


def train_countries() -> set:
    """Countries with labelled training pairs (those the learned address aliases cover)."""
    p = cache("train_norm.parquet")
    return set(pl.scan_parquet(p).select(pl.col("country").unique()).collect()["country"].to_list()) if os.path.exists(p) else set()


def _street_full_forms(vocab: Counter) -> dict:
    """S1 addresses are canonicalised ("avenue" -> "ave"), so the vocabulary lacks the full
    street-type words and a typo of one ("aenue", "avene") is 'corrected' to an unrelated nearby
    word ("tenue", "lavene").  The full forms are added with the frequency of their canonical form;
    returns {full form: canonical}."""
    from normalize import STREET
    full = {k: v for k, v in STREET.items() if len(k) >= 5 and len(k) > len(v) and v in vocab and k not in vocab}
    for k, v in full.items():
        vocab[k] = vocab[v]
    return full


def correct_split(split: str, force: bool = False):
    p = cache(f"{split}_norm.parquet")
    df = pl.read_parquet(p)
    if "n_core_c" in df.columns and not force:
        return
    df = df.drop([c for c in ("n_core_c", "a_words_c", "a_full_c", "n_corr", "a_corr") if c in df.columns])
    trained = train_countries()
    parts = []
    for (country,), d in df.group_by("country", maintain_order=True):
        s1 = d.filter(pl.col("src") == 1)
        q = d.filter(pl.col("src") != 1)
        maps = {}
        for col in ("n_core", "a_words"):
            vocab = _vocab(s1, col)
            # street-type full forms only for countries without training labels (for US/India the
            # learned aliases cover these typos; their representation stays as validated)
            full = _street_full_forms(vocab) if col == "a_words" and trained and country not in trained else {}
            corr = Corrector(vocab, long_dist=2 if col == "n_core" else 1)
            oov = _oov_types(q, col, vocab)
            mp = {}
            for t in oov:
                c = corr.correct(t)
                c = full.get(c, c)
                if c is not None and c != t:
                    mp[t] = c
            maps[col] = mp
            log(f"  {split}/{country} {col}: vocab {len(vocab):,}, OOV types {len(oov):,}, corrected {len(mp):,}")
            del corr
        d = _apply(d, "n_core", maps["n_core"], "n_core_c")
        d = _apply(d, "a_words", maps["a_words"], "a_words_c")
        d = _apply(d, "a_full", maps["a_words"], "a_full_c")
        parts.append(d)
    out = pl.concat(parts).sort("rid")
    out = out.with_columns(
        (pl.col("n_core").str.split(" ").list.set_difference(pl.col("n_core_c").str.split(" ")).list.len()).cast(pl.Int8).alias("n_corr"),
        (pl.col("a_words").str.split(" ").list.set_difference(pl.col("a_words_c").str.split(" ")).list.len()).cast(pl.Int8).alias("a_corr"),
    )
    out.write_parquet(p)
    log(f"{split}: spelling correction written ({out['n_corr'].gt(0).sum():,} names, {out['a_corr'].gt(0).sum():,} addresses changed)")


if __name__ == "__main__":
    for split in sys.argv[1:] or ["train", "test"]:
        correct_split(split, force=True)
