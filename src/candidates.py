"""Stage 1 of candidate generation: inverted-index retrieval + cheap similarity scores.

Output (cache/{split}_retrieval/part_*.parquet): one row per (query qid, S1 sid) retrieved pair with
the blocking scores and a handful of cheap rapidfuzz similarities used by the pruning ranker.
"""
import glob
import os
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from blocking import S1Index
from common import cache, log
from normalize import normalize_split

Q_CHUNK = 150_000
RET_COLS = ["rid", "src", "country", "n_full", "n_core", "n_core_c", "n_alt", "n_ns", "n_alt_ns", "n_skel",
            "a_full", "a_full_c", "a_words_c", "a_nums", "a_hglue"]


def cheap_sims(pairs: pl.DataFrame, q: pl.DataFrame, s: pl.DataFrame) -> pl.DataFrame:
    """Adds cheap string similarities to (qid, sid) pairs (query side spelling-corrected)."""
    qs = q.select(pl.col("rid").alias("qid"), pl.col("n_full").alias("qn"), pl.col("a_full_c").alias("qa"),
                  pl.col("n_ns").alias("qns"), pl.col("n_alt_ns").alias("qans"), pl.col("n_core_c").alias("qcc"))
    ss = s.select(pl.col("rid").alias("sid"), pl.col("n_full").alias("sn_"), pl.col("a_full").alias("sa_"),
                  pl.col("n_ns").alias("sns"), pl.col("n_core").alias("scc"))
    p = pairs.join(qs, on="qid", how="left", maintain_order="left").join(ss, on="sid", how="left", maintain_order="left")
    qn, sn = p["qn"].to_list(), p["sn_"].to_list()
    qa, sa = p["qa"].to_list(), p["sa_"].to_list()
    qns, sns, qans = p["qns"].to_list(), p["sns"].to_list(), p["qans"].to_list()
    kw = dict(workers=-1, dtype=np.uint8)
    return pairs.with_columns(
        pl.Series("c_ntset", process.cpdist(qn, sn, scorer=fuzz.token_set_ratio, **kw)),
        pl.Series("c_nsort", process.cpdist(qn, sn, scorer=fuzz.token_sort_ratio, **kw)),
        pl.Series("c_nctset", process.cpdist(p["qcc"].to_list(), p["scc"].to_list(), scorer=fuzz.token_set_ratio, **kw)),
        pl.Series("c_atset", process.cpdist(qa, sa, scorer=fuzz.token_set_ratio, **kw)),
        pl.Series("c_asort", process.cpdist(qa, sa, scorer=fuzz.token_sort_ratio, **kw)),
        pl.Series("c_nsr", np.maximum(process.cpdist(qns, sns, scorer=fuzz.ratio, **kw),
                                      process.cpdist(qans, sns, scorer=fuzz.ratio, **kw))),
    )


def retrieval_dir(split: str) -> str:
    return cache(f"{split}_retrieval")


def retrieve_split(split: str, force: bool = False, resume: bool = False):
    """Runs retrieval for all queries of a split; each query chunk is written to its own parquet
    part (nothing accumulates in memory).  Returns the list of part paths."""
    d_out = retrieval_dir(split)
    done = os.path.join(d_out, "_DONE")
    if os.path.exists(done) and not (force or resume):
        return sorted(glob.glob(os.path.join(d_out, "part_*.parquet")))
    os.makedirs(d_out, exist_ok=True)
    if not resume:
        for f in glob.glob(os.path.join(d_out, "*")):
            os.remove(f)
    df = normalize_split(split, columns=RET_COLS)
    n_pairs, n_q, j = 0, 0, 0
    for (country,), d in df.group_by("country", maintain_order=True):
        s1 = d.filter(pl.col("src") == 1)
        q = d.filter(pl.col("src") != 1)
        n_chunks = (q.height + Q_CHUNK - 1) // Q_CHUNK
        marker = os.path.join(d_out, f"_DONE_{country}")
        if resume and os.path.exists(marker):  # this country's parts are complete
            j += n_chunks
            n_q += q.height
            log(f"{split}/{country}: already done, skipping")
            continue
        log(f"{split}/{country}: S1={s1.height:,} queries={q.height:,}")
        idx = S1Index(s1)
        s1_small = s1.select("rid", "n_full", "a_full", "n_ns", "n_core")
        for i in range(0, q.height, Q_CHUNK):
            t0 = time.time()
            qc = q.slice(i, Q_CHUNK)
            cand = idx.query(qc)
            t1 = time.time()
            cand = cheap_sims(cand, qc, s1_small)
            cand.write_parquet(os.path.join(d_out, f"part_{j:04d}.parquet"))
            n_pairs += cand.height
            j += 1
            log(f"  chunk {j}: {qc.height:,} queries -> {cand.height:,} pairs (query {t1 - t0:.1f}s, sims {time.time() - t1:.1f}s)")
        n_q += q.height
        open(marker, "w").close()
        del idx, s1, q, s1_small
    log(f"{split}: {n_pairs:,} retrieved pairs ({n_pairs / n_q:.1f} per query)")
    open(done, "w").close()
    return sorted(glob.glob(os.path.join(d_out, "part_*.parquet")))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for split in args or ["train", "test"]:
        retrieve_split(split, resume=("--resume" in sys.argv), force=("--resume" not in sys.argv))
