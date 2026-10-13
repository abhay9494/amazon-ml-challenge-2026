"""Shared paths, I/O helpers and logging for the entity-resolution pipeline."""
import os
import sys
import time

import polars as pl

ROOT = os.environ.get(
    "ER_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "Data Set", "student_resource"),
)
DATA = os.path.join(ROOT, "dataset")
CACHE = os.environ.get(
    "ER_CACHE", os.path.join(os.path.dirname(__file__), "..", "cache")
)
OUT = os.environ.get("ER_OUT", os.path.join(os.path.dirname(__file__), "..", "..", "..", "output"))
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

N_JOBS = max(1, (os.cpu_count() or 4) - 2)

_T0 = time.time()


def log(*a):
    print(f"[{time.time() - _T0:8.1f}s]", *a, flush=True)


def read_tsv(path: str) -> pl.DataFrame:
    """Read a challenge TSV: tab separated, NO quote handling, everything as string."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, has_header=True)


def source_path(split: str, src: int) -> str:
    return os.path.join(DATA, split, f"{split}_source{src}.tsv")


def gt_path() -> str:
    return os.path.join(DATA, "train", "train_ground_truth.tsv")


def cache(name: str) -> str:
    return os.path.join(CACHE, name)


def load_raw(split: str) -> pl.DataFrame:
    """All three sources of a split stacked, with a dense int32 row id `rid` and source number `src`."""
    parts = []
    for s in (1, 2, 3):
        df = read_tsv(source_path(split, s)).with_columns(pl.lit(s, dtype=pl.Int8).alias("src"))
        parts.append(df)
    df = pl.concat(parts)
    df = df.with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null("").str.strip_chars(),
    )
    return df.with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int32))


def load_gt_pairs() -> pl.DataFrame:
    """Ground truth as (s1_id, m_id) pairs."""
    gt = read_tsv(gt_path())
    return (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"})
    )


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
