"""Raw-text flags for EVERY candidate pair (train OOF pairs / test pairs), in parallel.
Flags: doubled letters in name (dq, ds) and address (adq, ads), edit types (ins, dele, sub, trans,
roman), legal-form add/drop/diff, erased differences (accent_add, leet, dblspace, caps, punct_diff,
order).  Output: cache/rawflags_{split}.parquet (qid, sid, flags as Int8)."""
import sys
import time
from multiprocessing import Pool


NAMES = ["dq", "ds", "adq", "ads", "ins", "dele", "sub", "trans", "roman", "legal_add", "legal_drop", "legal_diff",
         "accent_add", "leet", "dblspace", "caps", "punct_diff", "order"]


def row(t):
    from dbl import dbl_feats
    from edits import flags as ef
    from erased import fl as rf
    qn, sn, qa, sa = t
    qn, sn, qa, sa = qn or "", sn or "", qa or "", sa or ""
    dq, ds = dbl_feats(qn, sn)
    adq, ads = dbl_feats(qa, sa)
    e = ef(qn, sn)
    r = rf(qn, sn)
    return (dq, ds, adq, ads, e["ins"], e["dele"], e["sub"], e["trans"], e["roman"], e["legal_add"], e["legal_drop"], e["legal_diff"],
            r["accent_add"], r["leet"], r["dblspace"], r["caps"], r["punct_diff"], r["order"])


def run(split):
    import polars as pl
    from common import cache, load_raw
    if split == "train":
        pairs = pl.read_parquet(cache("cv_oof1.parquet"), columns=["qid", "sid"])
    else:
        pairs = pl.read_parquet(cache("v7c_test_pred.parquet"), columns=["qid", "sid"])
    raw = load_raw(split).select("rid", "business_name", "business_address")
    q = raw.rename({"rid": "qid", "business_name": "qn", "business_address": "qa"})
    s = raw.rename({"rid": "sid", "business_name": "sn", "business_address": "sa"})
    out = []
    t0 = time.time()
    with Pool(14) as pool:
        for i in range(0, pairs.height, 1_000_000):
            sl = pairs.slice(i, 1_000_000).join(q, on="qid", how="left").join(s, on="sid", how="left")
            res = pool.map(row, list(zip(sl["qn"].to_list(), sl["sn"].to_list(), sl["qa"].to_list(), sl["sa"].to_list())), chunksize=4000)
            cols = list(zip(*res))
            out.append(sl.select("qid", "sid").with_columns(*[pl.Series(n, c, dtype=pl.Int8) for n, c in zip(NAMES, cols)]))
            print(f"{split}: {i + sl.height:,}/{pairs.height:,}  {time.time() - t0:.0f}s", flush=True)
    df = pl.concat(out)
    assert df.height == pairs.height
    df.write_parquet(cache(f"rawflags_{split}.parquet"))


if __name__ == "__main__":
    for split in sys.argv[1:]:
        run(split)
