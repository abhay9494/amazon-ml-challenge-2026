"""Scalable candidate generation (blocking).

Every record (S1 and S2/S3 alike) is turned into a bag of hashed *blocking keys*, built from the
normalised (and spelling-corrected) fields.  Keys are namespaced by country, so records only ever
meet records of their own country, whatever the country label is (open set of labels); the index
is also built one country at a time to bound memory.

Two independent channels:
  name channel  : core-name tokens, unordered pairs of name tokens, name token x house number,
                  the space-free core name (original, corrected, alias), phonetic skeletons (>= 5)
  address chan. : address words, adjacent address-word pairs, address word x house number

House-number variants (the sources perturb numbers): the first three numbers, the first number
without its last digit (6437 ~ 643) and without its first digit (9774 ~ 774), and the glued digits
of the first numbered component ("1-38" ~ 138).

An inverted index is built over the S1 keys only.  Keys shared by more than MAX_DF S1 records are
dropped, so each posting list is short and the cost is linear in the number of records.  Each query
(S2/S3 record) looks up only its RARE_K rarest keys per channel, accumulates IDF weights per S1
record and keeps the top candidates per channel.
"""
import polars as pl

from common import log

MAX_DF = 300          # keys held by more S1 records than this are not indexed
RARE_K = 8            # each query probes at most this many of its rarest keys per channel
TOP_PER_CHANNEL = 16  # candidates kept per channel
SEED = 1234567

_P32 = 4294967296
CH_NAME = 0
CH_ADDR = 1

KEY_COLS = ["rid", "country", "n_core_c", "n_alt", "n_ns", "n_alt_ns", "n_skel", "a_words_c", "a_nums", "a_hglue"]


def _key(prefix: str, *cols) -> pl.Expr:
    return pl.concat_str([pl.col("country"), pl.lit("\x1f" + prefix + "\x1f"), *cols])


def _house_numbers() -> pl.Expr:
    nums = pl.col("a_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    first = nums.list.first()
    n = first.str.len_chars()
    trunc = pl.when(n >= 3).then(first.str.slice(0, n - 1))
    drop1 = pl.when(n >= 3).then(first.str.slice(1).str.strip_chars_start("0"))
    glue = pl.when(pl.col("a_hglue") != "").then(pl.col("a_hglue"))
    extra = pl.concat_list([trunc, drop1, glue]).list.drop_nulls().list.eval(pl.element().filter(pl.element() != ""))
    return pl.concat_list([nums.list.head(3), extra]).list.unique(maintain_order=True)


def build_keys(df: pl.DataFrame, chunk: int = 200_000) -> pl.DataFrame:
    """df needs KEY_COLS.  Returns unique (rid, key:u64, ch:i8).  Chunked to bound memory."""
    df = df.select(KEY_COLS)
    return pl.concat([_build_keys(df.slice(i, chunk)) for i in range(0, df.height, chunk)])


def _toks(col: str, min_len: int = 2) -> pl.Expr:
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= min_len))


def _build_keys(df: pl.DataFrame) -> pl.DataFrame:
    base = df.select(
        "rid",
        "country",
        pl.concat_list([_toks("n_core_c"), _toks("n_alt")]).list.unique(maintain_order=True).alias("nt"),
        _toks("n_skel", 5).alias("sk"),
        _toks("a_words_c").alias("aw"),
        _house_numbers().alias("hs"),
        pl.col("n_ns"),
        pl.col("n_core_c").str.replace_all(" ", "").alias("n_ns_c"),
        pl.col("n_alt_ns"),
    )
    out = []
    lit_n = pl.lit(CH_NAME, pl.Int8).alias("ch")
    lit_a = pl.lit(CH_ADDR, pl.Int8).alias("ch")

    # name tokens
    nt = base.select("rid", "country", pl.col("nt").list.head(8)).explode("nt").drop_nulls("nt")
    out.append(nt.select("rid", _key("n1", pl.col("nt")).alias("k"), lit_n))
    # name token x house number variants (first 4 name tokens)
    nth = base.select("rid", "country", pl.col("nt").list.head(4), "hs").explode("nt").drop_nulls("nt").explode("hs").drop_nulls("hs")
    out.append(nth.select("rid", _key("nh", pl.col("nt"), pl.lit("|"), pl.col("hs")).alias("k"), lit_n))
    del nth
    # unordered pairs of name tokens
    nts = nt.with_columns(pl.int_range(pl.len()).over("rid").alias("i"))
    pr = nts.join(nts.select("rid", pl.col("nt").alias("nt2"), pl.col("i").alias("i2")), on="rid").filter(pl.col("i") < pl.col("i2"))
    out.append(pr.select("rid", _key("n2", pl.min_horizontal("nt", "nt2"), pl.lit("|"), pl.max_horizontal("nt", "nt2")).alias("k"), lit_n))
    del pr, nts
    # space-free names (original, corrected, alias)
    for c in ("n_ns", "n_ns_c", "n_alt_ns"):
        out.append(base.filter(pl.col(c).str.len_chars() >= 3).select("rid", _key("ns", pl.col(c)).alias("k"), lit_n))
    # acronyms ("BB" ~ "Bombay Business", "SAR" ~ "Styles All Realty LLC"): the initials of a
    # multi-word core name and a short single-token name meet in the same key space, combined with
    # a house number or one of the first two address words to stay rare
    toks = pl.col("n_core_c").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    ac = df.select(
        "rid", "country",
        pl.concat_list([
            pl.when(toks.list.len() >= 2).then(toks.list.eval(pl.element().str.slice(0, 1)).list.join("")),
            pl.when((toks.list.len() == 1) & toks.list.first().str.len_chars().is_between(2, 5)).then(toks.list.first()),
        ]).list.drop_nulls().alias("ac"),
    ).join(base.select("rid", "hs", pl.col("aw").list.head(6).alias("aw2")), on="rid").explode("ac").drop_nulls("ac")
    ach = ac.select("rid", "country", "ac", pl.concat_list(["hs", "aw2"]).alias("ctx")).explode("ctx").drop_nulls("ctx")
    out.append(ach.select("rid", _key("ac", pl.col("ac"), pl.lit("|"), pl.col("ctx")).alias("k"), lit_n))
    del ac, ach
    # name x locality ("First Food" + "Nashik"): the space-free corrected name with each of the
    # first 8 address words; retrieves same-name businesses by town/street when numbers differ
    nw = base.filter(pl.col("n_ns_c").str.len_chars() >= 3).select("rid", "country", "n_ns_c", pl.col("aw").list.head(8)).explode("aw").drop_nulls("aw")
    out.append(nw.select("rid", _key("nw", pl.col("n_ns_c"), pl.lit("|"), pl.col("aw")).alias("k"), lit_n))
    del nw
    # phonetic skeletons
    sk = base.select("rid", "country", "sk").explode("sk").drop_nulls("sk")
    out.append(sk.select("rid", _key("sk", pl.col("sk")).alias("k"), lit_n))

    # address words
    aw = base.select("rid", "country", pl.col("aw").list.head(12)).explode("aw").drop_nulls("aw")
    out.append(aw.select("rid", _key("a1", pl.col("aw")).alias("k"), lit_a))
    # address word x house number variants (first 6 address words)
    awh = base.select("rid", "country", pl.col("aw").list.head(6), "hs").explode("aw").drop_nulls("aw").explode("hs").drop_nulls("hs")
    out.append(awh.select("rid", _key("ah", pl.col("aw"), pl.lit("|"), pl.col("hs")).alias("k"), lit_a))
    del awh
    # adjacent address-word pairs
    aws = aw.with_columns(pl.int_range(pl.len()).over("rid").alias("i"))
    adj = aws.join(aws.select("rid", pl.col("aw").alias("aw2"), (pl.col("i") - 1).alias("i")), on=["rid", "i"])
    out.append(adj.select("rid", _key("a2", pl.col("aw"), pl.lit("|"), pl.col("aw2")).alias("k"), lit_a))

    keys = pl.concat(out)
    return keys.select("rid", pl.col("k").hash(seed=SEED).alias("key"), "ch").unique(["rid", "key"])


class S1Index:
    """Inverted index over the S1 blocking keys of ONE country."""

    def __init__(self, s1: pl.DataFrame):
        t = build_keys(s1)
        n_c = s1.height
        df = t.group_by("key").agg(pl.len().alias("df"))
        self.df_all = df.with_columns((pl.lit(float(n_c)) / pl.col("df")).log().cast(pl.Float32).alias("w"))
        t = t.join(self.df_all.filter(pl.col("df") <= MAX_DF), on="key")
        self.post = t.select("key", pl.col("rid").alias("sid"), "w", "ch")
        log(f"  S1 index: {n_c:,} records, {df.height:,} distinct keys, {self.post.height:,} postings (df<={MAX_DF})")

    def query(self, q: pl.DataFrame) -> pl.DataFrame:
        """Candidate (qid, sid) pairs with channel scores; union of the top per channel."""
        k = build_keys(q).rename({"rid": "qid"})
        k = k.join(self.df_all.filter(pl.col("df") <= MAX_DF).select("key", "df"), on="key")
        k = k.sort(["qid", "ch", "df", "key"]).group_by(["qid", "ch"], maintain_order=True).head(RARE_K)
        hits = k.select("qid", "key").join(self.post, on="key")
        hits = hits.select(
            (pl.col("qid").cast(pl.UInt64) * _P32 + pl.col("sid").cast(pl.UInt64)).alias("pk"),
            pl.when(pl.col("ch") == CH_NAME).then(pl.col("w")).otherwise(0.0).alias("wn"),
            pl.when(pl.col("ch") == CH_ADDR).then(pl.col("w")).otherwise(0.0).alias("wa"),
            (pl.col("ch") == CH_NAME).cast(pl.UInt8).alias("cn"),
            (pl.col("ch") == CH_ADDR).cast(pl.UInt8).alias("ca"),
        )
        agg = hits.group_by("pk").agg(
            pl.col("wn").sum().alias("sn"), pl.col("wa").sum().alias("sa"),
            pl.col("cn").sum().alias("kn"), pl.col("ca").sum().alias("ka"),
        ).with_columns(
            (pl.col("pk") // _P32).cast(pl.Int32).alias("qid"),
            (pl.col("pk") % _P32).cast(pl.Int32).alias("sid"),
            (pl.col("sn") + pl.col("sa")).alias("st"),
        )
        agg = agg.sort(["qid", "sid"])  # deterministic tie-breaking of the per-channel ranks
        keep = pl.lit(False)
        for c in ("st", "sn", "sa"):
            keep = keep | ((pl.col(c) > 0) & (pl.col(c).rank("ordinal", descending=True).over("qid") <= TOP_PER_CHANNEL))
        return agg.filter(keep).drop("pk")
