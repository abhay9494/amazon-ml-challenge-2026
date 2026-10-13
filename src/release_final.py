"""Final decision of the released configuration (v13c).

Test probability s = mean of re-scorers v5 and v6 (stack5.py, stack6.py); expected-F0.5 subset rule
per S1 entity with probability floor 0.6.  France (no training labels): links of the stage-1
ensemble rule that the re-scorer keeps, plus re-scorer additions whose record has an address and
exactly the S1's corrected core name; other French additions (decoy-word and category-swap
patterns, 0 % true in their Indian analogues) are not made.  Writes matching_results.tsv and
candidate_pairs.tsv to the output folder (common.OUT, i.e. ../../../output or $ER_OUT).
usage: python release_final.py [samecore|remove_only|full] [floor] [out_dir]"""
import os
import sys

import polars as pl

import pipeline as P
from common import OUT as DEFAULT_OUT, cache, load_raw, log
from decide import decide

FRANCE_MODE = sys.argv[1] if len(sys.argv) > 1 else "samecore"   # samecore | remove_only | full
FLOOR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
OUT = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT
st = pl.read_parquet(cache("stack5_test.parquet")).join(pl.read_parquet(cache("stack6_test.parquet")), on=["qid", "sid"]).with_columns(((pl.col("s5") + pl.col("s6")) / 2).alias("s")).select("qid", "sid", "p", "s")
assert st.height == st.select("qid", "sid").unique().height == 10_736_059
new = decide(st.select("qid", "sid", pl.col("s").alias("p")), thr=FLOOR, use_ef=True).select("qid", "sid")
old = decide(st.select("qid", "sid", "p"), thr=0.7, use_ef=True).select("qid", "sid")
country = load_raw("test").select(pl.col("rid").alias("qid"), "country")
if FRANCE_MODE == "samecore":
    # French links: v7c links the re-scorer keeps, plus re-scorer additions whose record has an address
    # and exactly the S1's corrected core name (decoy-word and category-swap additions are excluded)
    fr_q = country.filter(pl.col("country") == "France").select("qid")
    nrm = P.normalize_split("test", columns=["rid", "n_core_c", "n_core", "a_full"])
    fr_new = new.join(fr_q, on="qid")
    keep_old = fr_new.join(old, on=["qid", "sid"], how="semi")
    cand_add = fr_new.join(old, on=["qid", "sid"], how="anti")         .join(nrm.select(pl.col("rid").alias("qid"), pl.col("n_core_c").alias("qc"), pl.col("a_full").alias("qa")), on="qid")         .join(nrm.select(pl.col("rid").alias("sid"), pl.col("n_core").alias("sc")), on="sid")
    fr_add = cand_add.filter((pl.col("qc") == pl.col("sc")) & (pl.col("qa") != "")).select("qid", "sid")
    log(f"France samecore additions: {fr_add.height:,} of {cand_add.height:,}")
    new = pl.concat([new.join(fr_q, on="qid", how="anti"), keep_old, fr_add])
if FRANCE_MODE == "remove_only":
    fr_q = country.filter(pl.col("country") == "France").select("qid")
    new_fr = new.join(fr_q, on="qid").join(old, on=["qid", "sid"], how="semi")   # French links kept only if v7c had them
    new = pl.concat([new.join(fr_q, on="qid", how="anti"), new_fr])
assert new["qid"].n_unique() == new.height
added = new.join(old, on=["qid", "sid"], how="anti").join(country, on="qid")
removed = old.join(new, on=["qid", "sid"], how="anti").join(country, on="qid")
log(f"final ({FRANCE_MODE}): {new.height:,} links (v7c {old.height:,}); added {dict(added.group_by('country').len().iter_rows())}; "
    f"removed {dict(removed.group_by('country').len().iter_rows())}")
ids = P.id_maps("test")
os.makedirs(OUT, exist_ok=True)
P.write_lists(new, ids, "matched_entity_ids", os.path.join(OUT, "matching_results.tsv"))
P.write_lists(st.select("qid", "sid"), ids, "candidate_entity_ids", os.path.join(OUT, "candidate_pairs.tsv"))
added.write_parquet(cache("final_added_vs_stage1.parquet"))
removed.write_parquet(cache("final_removed_vs_stage1.parquet"))
