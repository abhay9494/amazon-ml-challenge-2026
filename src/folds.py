"""Entity-family folds for leak-free validation (see cv.py for the full design).

  S1 record              -> crc32(entity_id)
  matched S2/S3 record   -> the family of its single true S1 owner
  unmatched S2/S3 record -> crc32(own entity_id)
fold = bit 0 of the family hash, inner = bit 1 (for nested cross-fitting), fam = the hash.
Every candidate row of a query inherits the query's fold.
"""
import os
import zlib

import polars as pl

from common import cache, load_gt_pairs
from normalize import normalize_split


def _crc(s: str) -> int:
    return zlib.crc32(s.encode())


def train_folds(force: bool = False) -> pl.DataFrame:
    path = cache("train_folds.parquet")
    if os.path.exists(path) and not force:
        return pl.read_parquet(path)
    ids = normalize_split("train", columns=["rid", "entity_id", "src"])
    owner = load_gt_pairs().rename({"m_id": "entity_id", "s1_id": "owner"})
    fam = ids.join(owner, on="entity_id", how="left").with_columns(pl.coalesce("owner", "entity_id").alias("family"))
    fam = fam.with_columns(pl.col("family").map_elements(_crc, return_dtype=pl.UInt32).alias("fam"))
    fam = fam.with_columns(
        (pl.col("fam") % 2).cast(pl.Int8).alias("fold"),
        ((pl.col("fam") // 2) % 2).cast(pl.Int8).alias("inner"),
    ).select("rid", "fold", "inner", "fam")
    fam.write_parquet(path)
    return fam


def query_folds() -> pl.DataFrame:
    return train_folds().select(pl.col("rid").alias("qid"), "fold", "inner", "fam")
