"""Pair features for the matching model.

All features are per pair or per query record (no per-S1 counts, which shift between train and
test).  String similarities come from rapidfuzz.process.cpdist (C++, multi-threaded); set
features from polars list operations; IDF weights from the S1 records of the pair's country.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

# canonical legal forms -> coarse family (only used to detect type conflicts)
LEGAL_FAMILY = {
    "inc": "corp", "corp": "corp", "co": "corp",
    "llc": "llc", "pllc": "llc",
    "ltd": "ltd", "pvt": "ltd", "plc": "ltd",
    "llp": "llp", "lp": "lp", "pc": "pc", "pa": "pc",
    "sa": "sa", "sas": "sas", "sasu": "sas", "sarl": "sarl", "eurl": "sarl", "sci": "sci",
    "snc": "snc", "scop": "scop", "sca": "sca", "selarl": "selarl", "gie": "gie",
}

REC_COLS = ["rid", "country", "n_full", "n_core", "n_core_c", "n_alt", "n_ns", "n_alt_ns", "n_skel", "legal",
            "web", "dba", "native", "a_full", "a_full_c", "a_words", "a_words_c", "a_nums", "a_hglue",
            "addr_empty", "n_corr", "a_corr"]


def token_idf(s1: pl.DataFrame):
    """IDF tables (country, tok, idf) of S1 core-name tokens and address words."""
    n = s1.group_by("country").len().rename({"len": "n_c"})
    out = {}
    for col, name in (("n_core", "name"), ("a_words", "addr")):
        t = (
            s1.select("rid", "country", pl.col(col).str.split(" ").list.unique(maintain_order=True).alias("tok"))
            .explode("tok")
            .filter(pl.col("tok").str.len_chars() > 0)
            .group_by("country", "tok")
            .len()
            .join(n, on="country")
            .select("country", "tok", (pl.col("n_c") / pl.col("len")).log().cast(pl.Float32).alias("idf"))
        )
        out[name] = t
    return out


def extra_token_stats(cands: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame, prior: float = 0.3, k: float = 10.0) -> pl.DataFrame:
    """LABEL-FREE per-token statistic of the words a query name has *in addition* to its best S1
    candidate's name.  The generator adds two kinds of extra words: noise words on true matches
    ("Center", "Services", "Shri", "Dr" ...) and distinguishing words on sibling decoys ("Group",
    "Holdings", "Downtown" ...).  Siblings usually sit at a different house number of the same
    street, so among best-candidate pairs that differ by exactly ONE extra name word and share the
    street, the share with an equal house number (heq) separates the two kinds (corr. 0.93 with the
    true-match rate on train).  Computed on each split from its own records - no labels - so it
    also covers new vocabularies such as French in the test set."""
    best = cands.filter(pl.col("pr_rank") == 1).select("qid", "sid")
    qq = q.select(pl.col("rid").alias("qid"), "country", pl.col("n_core_c").alias("qc"), pl.col("a_nums").alias("qn"), pl.col("a_words_c").alias("qw"))
    ss = s1.select(pl.col("rid").alias("sid"), pl.col("n_core").alias("sc"), pl.col("a_nums").alias("sn"), pl.col("a_words").alias("sw"))
    p = best.join(qq, on="qid").join(ss, on="sid")
    p = p.with_columns(
        pl.col("qc").str.split(" ").list.set_difference(pl.col("sc").str.split(" ")).alias("qonly"),
        pl.col("sc").str.split(" ").list.set_difference(pl.col("qc").str.split(" ")).list.len().alias("ns"),
        (pl.col("qn").str.split(" ").list.first() == pl.col("sn").str.split(" ").list.first()).alias("heq"),
        (pl.col("qw").str.split(" ").list.set_intersection(pl.col("sw").str.split(" ")).list.len() >= 2).alias("street_ok"),
    )
    one = p.filter((pl.col("qonly").list.len() == 1) & (pl.col("ns") == 0) & pl.col("street_ok") & (pl.col("qn") != "") & (pl.col("sn") != ""))
    st = one.select("country", pl.col("qonly").list.first().alias("tok"), pl.col("heq").cast(pl.Float32)).group_by("country", "tok").agg(
        pl.len().alias("xt_n"), pl.col("heq").sum().alias("_h")
    )
    return st.with_columns(((pl.col("_h") + prior * k) / (pl.col("xt_n") + k)).cast(pl.Float32).alias("xt_heq")).drop("_h")


def name_uniqueness(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """How many S1 records (same country) carry exactly the query's core name / the S1 record's
    core name.  An exact name match is decisive when the name is unique, useless when it is shared
    by dozens of S1 businesses (and the query has no usable address)."""
    ndf = s1.group_by("country", "n_ns").len().rename({"len": "_df"})
    qd = q.select(pl.col("rid").alias("qid"), "country", "n_ns", "n_alt_ns", pl.col("n_core_c").str.replace_all(" ", "").alias("n_ns_c"))
    qd = qd.join(ndf, on=["country", "n_ns"], how="left").rename({"_df": "q_name_df"})
    qd = qd.join(ndf.rename({"n_ns": "n_ns_c"}), on=["country", "n_ns_c"], how="left").rename({"_df": "q_name_df_c"})
    qd = qd.join(ndf.rename({"n_ns": "n_alt_ns"}), on=["country", "n_alt_ns"], how="left").rename({"_df": "q_alt_name_df"})
    sd = s1.select(pl.col("rid").alias("sid"), "country", "n_ns").join(ndf, on=["country", "n_ns"], how="left").rename({"_df": "s_name_df"})
    out = pairs.select("qid", "sid").join(qd.select("qid", "q_name_df", "q_name_df_c", "q_alt_name_df"), on="qid", how="left", maintain_order="left").join(
        sd.select("sid", "s_name_df"), on="sid", how="left", maintain_order="left")
    return out.select(
        *[(pl.col(c).fill_null(0) + 1).log().cast(pl.Float32).alias(c) for c in ("q_name_df", "q_name_df_c", "q_alt_name_df", "s_name_df")]
    )


def _extra_token_feats(pid: pl.Series, country: pl.Series, qonly: pl.Series, xt: pl.DataFrame) -> pl.DataFrame:
    d = pl.DataFrame({"pid": pid, "country": country, "tok": qonly}).explode("tok").filter(
        pl.col("tok").is_not_null() & (pl.col("tok").str.len_chars() > 0)
    )
    d = d.join(xt, on=["country", "tok"], how="left").with_columns(pl.col("xt_n").fill_null(0))
    known = pl.col("xt_n") >= 30
    g = d.group_by("pid").agg(
        pl.col("xt_heq").filter(known).min().alias("xt_min"),
        pl.col("xt_heq").filter(known).max().alias("xt_max"),
        (known & (pl.col("xt_heq") < 0.4)).sum().alias("xt_n_decoy"),
        (known & (pl.col("xt_heq") >= 0.6)).sum().alias("xt_n_noise"),
        (~known).sum().alias("xt_n_unknown"),
    )
    out = pl.DataFrame({"pid": pid}).join(g, on="pid", how="left", maintain_order="left").with_columns(
        pl.col("xt_n_decoy").fill_null(0), pl.col("xt_n_noise").fill_null(0), pl.col("xt_n_unknown").fill_null(0),
        pl.col("xt_min").fill_null(-1.0), pl.col("xt_max").fill_null(-1.0),
    )
    return out.drop("pid").with_columns(pl.all().cast(pl.Float32))


def _idf_sum(pid: pl.Series, country: pl.Series, toks: pl.Series, idf: pl.DataFrame, default: float) -> np.ndarray:
    d = pl.DataFrame({"pid": pid, "country": country, "tok": toks}).explode("tok").filter(
        pl.col("tok").is_not_null() & (pl.col("tok").str.len_chars() > 0)
    )
    d = d.join(idf, on=["country", "tok"], how="left").with_columns(pl.col("idf").fill_null(default))
    s = d.group_by("pid").agg(pl.col("idf").sum())
    return (
        pl.DataFrame({"pid": pid}).join(s, on="pid", how="left", maintain_order="left")["idf"].fill_null(0.0).to_numpy().astype(np.float32)
    )


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _soft_match(pid: pl.Series, country: pl.Series, qt: pl.Series, st: pl.Series, idf: pl.DataFrame,
                cap: int, prefix: str) -> pl.DataFrame:
    """Monge-Elkan style soft token alignment.  For every S1 token the best fuzz.ratio against the
    query tokens (and vice versa): mean, minimum, IDF-weighted mean and number of tokens without a
    >= 80 counterpart.  Robust to typos that exact token sets miss."""
    d = pl.DataFrame({"pid": pid, "country": country, "st": st.list.head(cap), "qt": qt.list.head(cap)})
    x = d.explode("st").drop_nulls("st").explode("qt").with_columns(pl.col("qt").fill_null(""))
    x = x.with_columns(pl.Series("r", process.cpdist(x["st"].to_list(), x["qt"].to_list(), scorer=fuzz.ratio,
                                                     workers=-1, dtype=np.float32)))
    bs = x.group_by("pid", "country", "st").agg(pl.col("r").max())
    bs = bs.join(idf.rename({"tok": "st"}), on=["country", "st"], how="left").with_columns(pl.col("idf").fill_null(12.0))
    gs = bs.group_by("pid").agg(
        pl.col("r").mean().alias(f"{prefix}_me_s"), pl.col("r").min().alias(f"{prefix}_me_s_min"),
        ((pl.col("r") * pl.col("idf")).sum() / pl.col("idf").sum()).alias(f"{prefix}_me_s_idf"),
        (pl.col("r") < 80).sum().alias(f"{prefix}_me_s_n80"),
    )
    bq = x.filter(pl.col("qt") != "").group_by("pid", "qt").agg(pl.col("r").max())
    gq = bq.group_by("pid").agg(
        pl.col("r").mean().alias(f"{prefix}_me_q"), pl.col("r").min().alias(f"{prefix}_me_q_min"),
        (pl.col("r") < 80).sum().alias(f"{prefix}_me_q_n80"),
    )
    out = pl.DataFrame({"pid": pid}).join(gs, on="pid", how="left", maintain_order="left").join(
        gq, on="pid", how="left", maintain_order="left")
    return out.drop("pid").with_columns(pl.all().cast(pl.Float32).fill_null(-1.0))


def pair_features(pairs: pl.DataFrame, q: pl.DataFrame, s: pl.DataFrame, idf: dict, xt: pl.DataFrame) -> pl.DataFrame:
    """pairs: (qid, sid, + retrieval columns).  q / s: normalised record frames keyed by rid."""
    qd = q.select(REC_COLS).rename({c: "q_" + c for c in REC_COLS}).rename({"q_rid": "qid"})
    sd = s.select(REC_COLS).rename({c: "s_" + c for c in REC_COLS}).rename({"s_rid": "sid"})
    p = pairs.with_row_index("pid").join(qd, on="qid", how="left").join(sd, on="sid", how="left").sort("pid")

    L = lambda c: p[c].to_list()
    qn, sn = L("q_n_core"), L("s_n_core")
    qf, sf = L("q_n_full"), L("s_n_full")
    qalt = L("q_n_alt")
    qa, sa = L("q_a_full"), L("s_a_full")
    qw, sw = L("q_a_words"), L("s_a_words")
    f = {}
    # ---- name similarities
    f["n_ratio"] = _cp(qn, sn, fuzz.ratio)
    f["n_tsort"] = _cp(qn, sn, fuzz.token_sort_ratio)
    f["n_tset"] = _cp(qn, sn, fuzz.token_set_ratio)
    f["n_partial"] = _cp(qn, sn, fuzz.partial_ratio)
    f["n_jw"] = _cp(qn, sn, JaroWinkler.normalized_similarity) * 100
    f["n_full_tset"] = _cp(qf, sf, fuzz.token_set_ratio)
    f["n_full_ratio"] = _cp(qf, sf, fuzz.ratio)
    f["n_ns_ratio"] = _cp(L("q_n_ns"), L("s_n_ns"), fuzz.ratio)
    f["n_ns_partial"] = _cp(L("q_n_ns"), L("s_n_ns"), fuzz.partial_ratio)
    f["n_skel_tset"] = _cp(L("q_n_skel"), L("s_n_skel"), fuzz.token_set_ratio)
    alt_tset = _cp(qalt, sn, fuzz.token_set_ratio)
    alt_ratio = _cp(L("q_n_alt_ns"), L("s_n_ns"), fuzz.ratio)
    has_alt = np.array([len(x) > 0 for x in qalt])
    f["n_alt_tset"] = np.where(has_alt, alt_tset, -1).astype(np.float32)
    f["n_alt_ratio"] = np.where(has_alt, alt_ratio, -1).astype(np.float32)
    f["n_best_tset"] = np.maximum(f["n_tset"], np.where(has_alt, alt_tset, 0)).astype(np.float32)
    # ---- address similarities
    f["a_ratio"] = _cp(qa, sa, fuzz.ratio)
    f["a_tsort"] = _cp(qa, sa, fuzz.token_sort_ratio)
    f["a_tset"] = _cp(qa, sa, fuzz.token_set_ratio)
    f["a_ptset"] = _cp(qa, sa, fuzz.partial_token_set_ratio)
    f["a_partial"] = _cp(qa, sa, fuzz.partial_ratio)
    f["aw_tset"] = _cp(qw, sw, fuzz.token_set_ratio)
    f["aw_tsort"] = _cp(qw, sw, fuzz.token_sort_ratio)
    # all address digits glued together: robust to "7-16" vs "716", "A-76" vs "76 A"
    qd = [x.replace(" ", "") for x in L("q_a_nums")]
    sd = [x.replace(" ", "") for x in L("s_a_nums")]
    f["h_digits_ratio"] = _cp(qd, sd, fuzz.ratio)
    f["h_digits_partial"] = _cp(qd, sd, fuzz.partial_ratio)
    # spelling-corrected query strings (S1 side is the clean reference)
    qnc, qac = L("q_n_core_c"), L("q_a_full_c")
    f["nc_tset"] = _cp(qnc, sn, fuzz.token_set_ratio)
    f["nc_ratio"] = _cp(qnc, sn, fuzz.ratio)
    f["nc_tsort"] = _cp(qnc, sn, fuzz.token_sort_ratio)
    f["ac_tset"] = _cp(qac, sa, fuzz.token_set_ratio)
    f["ac_tsort"] = _cp(qac, sa, fuzz.token_sort_ratio)
    f["ac_ratio"] = _cp(qac, sa, fuzz.ratio)
    # acronym relation between the two core names
    qtok = p["q_n_core_c"].str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    stok = p["s_n_core"].str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    acro = pl.DataFrame({"qt": qtok, "st": stok}).select(
        (
            ((pl.col("qt").list.len() == 1) & (pl.col("st").list.len() >= 2)
             & (pl.col("qt").list.first() == pl.col("st").list.eval(pl.element().str.slice(0, 1)).list.join("")))
            | ((pl.col("st").list.len() == 1) & (pl.col("qt").list.len() >= 2)
               & (pl.col("st").list.first() == pl.col("qt").list.eval(pl.element().str.slice(0, 1)).list.join("")))
        ).fill_null(False).cast(pl.Int8).alias("a")
    )["a"]
    f["n_acronym"] = acro.to_numpy()
    f["q_n_corr"] = p["q_n_corr"].cast(pl.Float32).to_numpy()
    f["q_a_corr"] = p["q_a_corr"].cast(pl.Float32).to_numpy()

    # ---- token-set features (polars)
    tk = p.select(
        "pid",
        pl.col("q_country").alias("country"),
        pl.col("q_n_core_c").str.split(" ").list.unique(maintain_order=True).alias("qt"),
        pl.col("q_n_core").str.split(" ").list.unique(maintain_order=True).alias("qt_orig"),
        pl.col("s_n_core").str.split(" ").list.unique(maintain_order=True).alias("st"),
        pl.col("q_n_alt").str.split(" ").list.unique(maintain_order=True).alias("qat"),
        pl.col("q_n_skel").str.split(" ").list.unique(maintain_order=True).alias("qk"),
        pl.col("s_n_skel").str.split(" ").list.unique(maintain_order=True).alias("sk"),
        pl.col("q_a_words_c").str.split(" ").list.unique(maintain_order=True).alias("qw"),
        pl.col("s_a_words").str.split(" ").list.unique(maintain_order=True).alias("sw"),
        pl.col("q_a_nums").str.split(" ").alias("qnum"),
        pl.col("s_a_nums").str.split(" ").alias("snum"),
        pl.col("q_legal").str.split(" ").alias("ql"),
        pl.col("s_legal").str.split(" ").alias("sl"),
    ).with_columns(
        [pl.col(c).list.eval(pl.element().filter(pl.element() != "")) for c in ("qt", "qt_orig", "st", "qat", "qk", "sk", "qw", "sw", "qnum", "snum", "ql", "sl")]
    )
    tk = tk.with_columns(
        pl.col("qt").list.set_intersection("st").alias("nc"),
        pl.col("qt").list.set_difference("st").alias("nq_only"),
        pl.col("st").list.set_difference("qt").alias("ns_only"),
        pl.col("qat").list.set_intersection("st").alias("nac"),
        pl.col("qw").list.set_intersection("sw").alias("wc"),
        pl.col("qw").list.set_difference("sw").alias("wq_only"),
        pl.col("sw").list.set_difference("qw").alias("ws_only"),
    )
    g = tk.select(
        pl.col("qt").list.len().alias("n_qlen"),
        pl.col("st").list.len().alias("n_slen"),
        pl.col("nc").list.len().alias("n_common"),
        pl.col("nq_only").list.len().alias("n_qonly"),
        pl.col("ns_only").list.len().alias("n_sonly"),
        pl.col("nac").list.len().alias("n_alt_common"),
        (pl.col("qk").list.set_intersection("sk").list.len() / pl.col("qk").list.set_union("sk").list.len().clip(1)).alias("n_skel_jac"),
        pl.col("wc").list.len().alias("a_common"),
        pl.col("wq_only").list.len().alias("a_qonly"),
        pl.col("ws_only").list.len().alias("a_sonly"),
        pl.col("qw").list.len().alias("a_qlen"),
        pl.col("sw").list.len().alias("a_slen"),
        # house numbers
        pl.col("qnum").list.len().alias("h_qn"),
        pl.col("snum").list.len().alias("h_sn"),
        (pl.col("qnum").list.first() == pl.col("snum").list.first()).fill_null(False).alias("h_first_eq"),
        pl.col("snum").list.contains(pl.col("qnum").list.first()).fill_null(False).alias("h_qfirst_in_s"),
        pl.col("qnum").list.contains(pl.col("snum").list.first()).fill_null(False).alias("h_sfirst_in_q"),
        (pl.col("qnum").list.set_intersection("snum").list.len() / pl.col("qnum").list.set_union("snum").list.len().clip(1)).alias("h_jac"),
        pl.col("qnum").list.set_difference("snum").list.len().alias("h_qonly"),
        pl.col("snum").list.set_difference("qnum").list.len().alias("h_sonly"),
        (
            (pl.col("qnum").list.first().cast(pl.Float64, strict=False) - pl.col("snum").list.first().cast(pl.Float64, strict=False)).abs() + 1
        ).log10().alias("h_first_logdiff"),
        # one first number is a prefix / suffix of the other (dropped digits: 774 ~ 9774, 643 ~ 6437)
        (
            (pl.col("qnum").list.first() != pl.col("snum").list.first())
            & (
                pl.col("snum").list.first().str.starts_with(pl.col("qnum").list.first())
                | pl.col("snum").list.first().str.ends_with(pl.col("qnum").list.first())
                | pl.col("qnum").list.first().str.starts_with(pl.col("snum").list.first())
                | pl.col("qnum").list.first().str.ends_with(pl.col("snum").list.first())
            )
        ).fill_null(False).alias("h_affix"),
        # legal forms
        pl.col("ql").list.len().alias("l_qn"),
        pl.col("sl").list.len().alias("l_sn"),
        pl.col("ql").list.set_intersection("sl").list.len().alias("l_common"),
        pl.col("ql").list.eval(pl.element().replace_strict(LEGAL_FAMILY, default="x")).list.unique(maintain_order=True).alias("qfam"),
        pl.col("sl").list.eval(pl.element().replace_strict(LEGAL_FAMILY, default="x")).list.unique(maintain_order=True).alias("sfam"),
    ).with_columns(
        ((pl.col("qfam").list.len() > 0) & (pl.col("sfam").list.len() > 0) & (pl.col("qfam").list.set_intersection("sfam").list.len() == 0)).alias("l_conflict")
    ).drop("qfam", "sfam")

    cty = p["q_country"]
    pid = p["pid"]
    n_q_idf = _idf_sum(pid, cty, tk["qt"], idf["name"], 12.0)
    n_s_idf = _idf_sum(pid, cty, tk["st"], idf["name"], 12.0)
    n_c_idf = _idf_sum(pid, cty, tk["nc"], idf["name"], 12.0)
    n_ac_idf = _idf_sum(pid, cty, tk["nac"], idf["name"], 12.0)
    a_q_idf = _idf_sum(pid, cty, tk["qw"], idf["addr"], 12.0)
    a_s_idf = _idf_sum(pid, cty, tk["sw"], idf["addr"], 12.0)
    a_c_idf = _idf_sum(pid, cty, tk["wc"], idf["addr"], 12.0)
    f["n_idf_cov_s"] = n_c_idf / np.maximum(n_s_idf, 1e-3)
    f["n_idf_cov_q"] = n_c_idf / np.maximum(n_q_idf, 1e-3)
    f["n_idf_common"] = n_c_idf
    f["n_idf_qonly"] = n_q_idf - n_c_idf
    f["n_idf_sonly"] = n_s_idf - n_c_idf
    f["n_alt_idf_cov_s"] = n_ac_idf / np.maximum(n_s_idf, 1e-3)
    f["a_idf_cov_s"] = a_c_idf / np.maximum(a_s_idf, 1e-3)
    f["a_idf_cov_q"] = a_c_idf / np.maximum(a_q_idf, 1e-3)
    f["a_idf_common"] = a_c_idf
    f["a_idf_qonly"] = a_q_idf - a_c_idf
    f["a_idf_sonly"] = a_s_idf - a_c_idf

    # ---- record flags
    f["q_web"] = p["q_web"].cast(pl.Int8).to_numpy()
    f["q_dba"] = p["q_dba"].cast(pl.Int8).to_numpy()
    f["q_native"] = p["q_native"].cast(pl.Int8).to_numpy()
    f["q_addr_empty"] = p["q_addr_empty"].cast(pl.Int8).to_numpy()
    f["q_src3"] = (p["qid"].to_numpy() >= 0).astype(np.int8) * 0  # placeholder, filled by caller
    f["q_nlen_chars"] = p["q_n_full"].str.len_chars().to_numpy()
    f["s_nlen_chars"] = p["s_n_full"].str.len_chars().to_numpy()

    # out-of-vocabulary (w.r.t. the country's S1 names) original query tokens: alias / made-up names
    oov = (
        pl.DataFrame({"pid": pid, "country": cty, "tok": tk["qt_orig"]}).explode("tok").drop_nulls("tok")
        .join(idf["name"], on=["country", "tok"], how="left")
        .group_by("pid").agg(pl.col("idf").is_null().sum().alias("n_q_oov"), pl.len().alias("_n"))
    )
    oov = pl.DataFrame({"pid": pid}).join(oov, on="pid", how="left", maintain_order="left").fill_null(0)
    f["n_q_oov"] = oov["n_q_oov"].cast(pl.Float32).to_numpy()
    f["n_q_oov_frac"] = (oov["n_q_oov"] / oov["_n"].clip(1)).cast(pl.Float32).to_numpy()
    # glued house-number digits vs the other side's glued digits / first number
    qg, sg = p["q_a_hglue"], p["s_a_hglue"]
    qf1 = p["q_a_nums"].str.split(" ").list.first().fill_null("")
    sf1 = p["s_a_nums"].str.split(" ").list.first().fill_null("")
    f["h_glue_match"] = (((qg != "") & ((qg == sg) | (qg == sf1))) | ((sg != "") & (sg == qf1))).cast(pl.Int8).to_numpy()
    sm_n = _soft_match(pid, cty, tk["qt"], tk["st"], idf["name"], 8, "n")
    sm_a = _soft_match(pid, cty, tk["qw"], tk["sw"], idf["addr"], 6, "a")
    xf = _extra_token_feats(pid, cty, tk["nq_only"], xt)
    feats = pl.DataFrame({k: np.asarray(v) for k, v in f.items()})
    feats = pl.concat([p.select("pid", "qid", "sid"), feats, g.with_columns(pl.all().cast(pl.Float32)), xf, sm_n, sm_a], how="horizontal")
    return feats.drop("q_src3")
