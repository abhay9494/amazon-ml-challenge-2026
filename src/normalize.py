"""Name / address normalisation.

Both sides of every comparison go through exactly the same functions.  The output of
`normalize_split` is one parquet per split with, per record:

  n_full   normalised full name (legal forms canonicalised)
  n_core   name without legal forms / articles (space separated tokens)
  n_alt    core of the alias part of a DBA-style name ("X f/k/a Y" -> Y), else ""
  n_ns     n_core without spaces   (websites, split/merged words)
  n_skel   phonetic consonant skeletons of the core tokens
  legal    sorted canonical legal-form tokens (e.g. "ltd pvt")
  a_full   normalised address tokens, state / country / placeholders / unit words removed
  a_words  the non-numeric address tokens
  a_nums   the numeric address tokens (leading zeros stripped) in order
  flags    web, dba, native (non-latin script), addr_empty
"""
import json
import os
import re
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

from common import N_JOBS, cache, load_raw, log
from regions import NATIVE_STATES

# --------------------------------------------------------------------------------------------
# transliteration dictionary (learned from training pairs by translit.py; optional)
# --------------------------------------------------------------------------------------------
TRANSLIT: dict = {}


def load_translit():
    global TRANSLIT
    p = cache("translit_dict.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            TRANSLIT = json.load(f)


_ZW = re.compile(r"[​-‍⁠﻿]")
_NATIVE_RUN = re.compile(r"[ऀ-ॣ०-ൿ]+")


def _native(m) -> str:
    t = m.group(0)
    return " " + (TRANSLIT.get(t) or unidecode(t)) + " "


def to_ascii(s: str) -> str:
    """Indic-script words -> learned Latin spelling (fallback unidecode); accents etc. -> unidecode."""
    if s.isascii():
        return s
    s = _ZW.sub("", s)
    for k, v in NATIVE_STATES.items():
        if k in s:
            s = s.replace(k, v)
    s = _NATIVE_RUN.sub(_native, s)
    return unidecode(s)


# --------------------------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------------------------
_REPEAT = re.compile(r"([a-z])\1+")
# Leetspeak digits actually used by the data generator (measured on aligned training pairs:
# 1->l 11.9k vs 1->i 12; 0->o 16.2k; 5->s 6.2k; 6->g 2.3k; 8->b 2.9k; 3/4/7/9 are never leet).
_LEET = str.maketrans({"0": "o", "1": "l", "5": "s", "6": "g", "8": "b"})
_LEET_DIGITS = set("01568")
_HAS_ALPHA = re.compile(r"[a-z]")
_HAS_DIGIT = re.compile(r"\d")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")
_APOS = re.compile(r"['’‘`´]")


_POSS = re.compile(r"['’]s\b", re.I)
_ELISION = re.compile(r"\b([ldLD])['’‘`´]")      # l'Alma -> l Alma
_GLUED = re.compile(r"\b([ld])(?=[A-Z][a-z])")                # lAlma  -> l Alma (case-sensitive)


def elide(s: str) -> str:
    """Apostrophes: possessive 's is glued ("ronny's" -> "ronnys"); French elisions are split so
    that the article becomes a stop word ("l'alma" / "lAlma" -> "l alma"); others are removed."""
    s = _POSS.sub("s", s)
    s = _ELISION.sub(r"\1 ", s)
    return _GLUED.sub(r"\1 ", s)


def collapse(s: str) -> str:
    return _REPEAT.sub(r"\1", s)


def fix_leet_token(t: str) -> str:
    if _HAS_ALPHA.search(t) and _HAS_DIGIT.search(t) and all(c in _LEET_DIGITS for c in t if c.isdigit()):
        return t.translate(_LEET)
    return t


# --------------------------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------------------------
LEGAL = {
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co",
    "llc": "llc", "pllc": "llc", "lc": "llc",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "llp": "llp", "lp": "lp", "pc": "pc", "plc": "plc", "pa": "pa",
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "sca": "sca", "selarl": "selarl", "gie": "gie",
}
# Name-word synonyms -> one canonical spelling (applied after legal forms).  French abbreviations
# that the sources use interchangeably with the full word.
NAME_SYN = {
    "etablissements": "ets", "etablissement": "ets", "etab": "ets",
    "compagnie": "co", "cie": "co",
    "association": "assoc", "asso": "assoc", "asociation": "assoc", "assoc": "assoc",
    "saint": "st", "sainte": "ste",
    "societe": "soc",
    # transliteration / spelling variants measured on aligned training pairs
    # (jay->jai 5.4k, laxmi->lakshmi 5.7k, shre/sre->shri 5.7k, centre->center 1.8k)
    "jay": "jai", "laxmi": "lakshmi", "lakshmi": "lakshmi", "shree": "shri", "shre": "shri", "sree": "shri",
    "sre": "shri", "sri": "shri", "centre": "center",
}
NAME_SYN.update({collapse(k): v for k, v in list(NAME_SYN.items())})  # collapsed spellings
STOP_NAME = {"the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d", "a", "an", "en"}

_WEB = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+(?:\.[a-z0-9\-]+)*?)\.(?:com|net|org|in|co\.in|co|fr|biz|info|io|us)\b")
_DBA = re.compile(
    r"\b(?:f/k/a|fka|a/k/a|aka|d/b/a|dba|t/a|n/k/a|nka|formerly known as|formerly|also known as|"
    r"doing business as|trading as|trading under|nee|operating as|now known as)\b"
)
_TAG = re.compile(r"#\s*\d+|\b\d{5,}\b")


def _name_tokens(s: str):
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = _APOS.sub("", s)
    s = _TAG.sub(" ", s)
    toks = []
    for t in _NONALNUM.sub(" ", s).split():
        if t in LEGAL:  # before collapsing, so that llp / lp and pllc / plc stay distinct
            toks.append(LEGAL[t])
            continue
        if not t.isalpha() and not t.isdigit():
            t = fix_leet_token(t)
        t = collapse(t)
        t = NAME_SYN.get(t, t)
        toks.append(LEGAL.get(t, t))
    return toks


_SK_ASP = re.compile(r"(?<=[bdgkpt])h")
_SK_VOW = re.compile(r"[aeiouyh]")


def skeleton(t: str) -> str:
    """Consonant skeleton: first letter + remaining consonants, with transliteration-prone
    letters merged (ph=f, sh=s, c/q=k, z=s, j=g, w=v, aspirated consonants lose the h)."""
    t = t.replace("ph", "f").replace("sh", "s").replace("ck", "k").replace("q", "k").replace("x", "ks")
    t = t.replace("c", "k").replace("z", "s").replace("j", "g").replace("w", "v")
    t = _SK_ASP.sub("", t)
    if len(t) > 1:
        t = t[0] + _SK_VOW.sub("", t[1:])
    return collapse(t)


def norm_name(raw: str):
    s = elide(to_ascii(raw)).lower().strip()
    native = not raw.isascii() and any(ord(c) > 0x2FF for c in raw)
    web = False
    m = _WEB.match(s.strip())
    if m and " " not in s.strip():
        web = True
        s = m.group(1).replace(".", " ")
    alt = ""
    dm = _DBA.search(s)
    dba = dm is not None
    main = s
    if dba:
        main, alt = s[: dm.start()], s[dm.end():]
    toks = _name_tokens(main)
    legal = sorted({t for t in toks if t in _LEGAL_CANON})
    core = [t for t in toks if t not in _LEGAL_CANON and t not in STOP_NAME]
    alt_core = []
    if alt:
        alt_core = [t for t in _name_tokens(alt) if t not in _LEGAL_CANON and t not in STOP_NAME]
        if not legal:
            legal = sorted({t for t in _name_tokens(alt) if t in _LEGAL_CANON})
    full = " ".join(toks)
    return {
        "n_full": full,
        "n_core": " ".join(core),
        "n_alt": " ".join(alt_core),
        "n_ns": "".join(core),
        "n_alt_ns": "".join(alt_core),
        "n_skel": " ".join(skeleton(t) for t in core),
        "legal": " ".join(legal),
        "web": web,
        "dba": dba,
        "native": native,
    }


_LEGAL_CANON = set(LEGAL.values())

# --------------------------------------------------------------------------------------------
# addresses
# --------------------------------------------------------------------------------------------
STREET = {
    "street": "st", "st": "st", "str": "st", "saint": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd", "bvd": "blvd",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "court": "ct", "ct": "ct", "crt": "ct",
    "circle": "cir", "cir": "cir", "circ": "cir",
    "lane": "ln", "ln": "ln",
    "place": "pl", "pl": "pl", "plz": "plz", "plaza": "plz",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "trail": "trl", "trl": "trl",
    "terrace": "ter", "ter": "ter", "terr": "ter",
    "square": "sq", "sq": "sq",
    "way": "way", "wy": "way",
    "expressway": "expy", "expy": "expy",
    "freeway": "fwy", "fwy": "fwy",
    "point": "pt", "pt": "pt",
    "crossing": "xing", "xing": "xing",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "mount": "mt", "mt": "mt", "mountain": "mtn", "mtn": "mtn",
    "heights": "hts", "hts": "hts",
    "junction": "jct", "jct": "jct",
    "loop": "loop", "run": "run", "pass": "pass", "passage": "pass",
    "cove": "cv", "cv": "cv", "creek": "crk", "crk": "crk",
    "extension": "ext", "extn": "ext", "ext": "ext",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # French
    "rue": "rue", "r": "rue",
    "allee": "allee", "all": "allee",
    "impasse": "imp", "imp": "imp",
    "chemin": "ch", "ch": "ch", "che": "ch",
    "route": "rte", "rte": "rte",
    "faubourg": "fbg", "fbg": "fbg",
    "cours": "crs", "crs": "crs",
    "quai": "quai", "qu": "quai",
    "residence": "res", "res": "res",
    "chem": "ch", "fg": "fbg", "esplanade": "espl", "espl": "espl", "passage_": "pass", "psg": "pass",
    "promenade": "prom", "prom": "prom", "lotissement": "lot", "lot": "lot", "chaussee": "chee", "chee": "chee",
    "hameau": "ham", "ham": "ham", "allees": "allee", "q": "quai", "rond": "rond", "rpt": "rpt",
    "sainte": "ste", "ste": "ste",
    "bis": "b",
    # India
    "nagar": "ngr", "ngr": "ngr",
    "sector": "sec", "sec": "sec",
    "block": "blk", "blk": "blk",
    "marg": "marg", "mg": "marg",
    "colony": "col", "col": "col",
    "ft": "fort", "fort": "fort", "tpke": "tpke", "turnpike": "tpke", "rdg": "ridge", "ridge": "ridge",
}
ORD_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
}
UNIT = {
    "unit", "apt", "apartment", "appartement", "appt", "suite", "no", "nos", "number", "num",
    "hno", "house", "door", "flat", "plot", "pmb", "po", "box", "bp", "cs", "cedex", "floor", "flr",
    "bldg", "building", "khasra", "kh", "survey", "sy",
    "h", "hn", "fl",  # leftovers of "H.No." / "HN" / "Fl 1" (unmatched in thousands of training pairs)
}
PLACEHOLDER = {"null", "none", "nan", "n/a", "na", "", "-", "nil", "unknown", "not available"}
ADDR_STOP = {"the", "of", "de", "du", "des", "la", "le", "les", "l", "d", "a", "and", "et",
             "near", "nr", "opp", "opposite", "behind", "beside", "next", "to", "cdp", "via",
             "city", "township", "twp"}  # "ENID CITY" / "X Township" vs "X"


def _with_collapsed(vocab):
    """Address text is collapsed ('floor' -> 'flor') before lookup: add collapsed keys."""
    if isinstance(vocab, dict):
        extra = {collapse(k): v for k, v in vocab.items() if collapse(k) not in vocab}
        vocab.update(extra)
    else:
        vocab |= {collapse(k) for k in vocab}
    return vocab


STREET = _with_collapsed(STREET)
ORD_WORDS = _with_collapsed(ORD_WORDS)
UNIT = _with_collapsed(UNIT)
PLACEHOLDER = _with_collapsed(PLACEHOLDER)
ADDR_STOP = _with_collapsed(ADDR_STOP)

REGIONS: set = set()  # filled by load_regions(): canonical state / region / country component strings


def load_regions():
    from regions import REGION_STRINGS

    REGIONS.update(REGION_STRINGS)


_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|ere|eme|e|me)\b")
_LEAD0 = re.compile(r"\b0+(\d)")
_SPLIT_AD = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_COMP = re.compile(r"[,;|]")
_DIGITS = re.compile(r"\d+")
_NUMBER_SIGN = re.compile(r"\bn[o°º]\s*\.?\s*(?=\d)|[#°º№]")


def _addr_mixed(t: str) -> str:
    """A token mixing letters and digits: leetspeak word ('6urgaon') -> letters, otherwise
    (house numbers like '35b', 'a76', '8b12') split into letter / digit runs."""
    nd = sum(c.isdigit() for c in t)
    if len(t) - nd >= 4 and nd <= 2 and all(c in _LEET_DIGITS for c in t if c.isdigit()):
        return t.translate(_LEET)
    return _SPLIT_AD.sub(" ", t)


def _comp_key(c: str) -> str:
    return _NONALNUM.sub(" ", c).strip()


_NUMERO_RAW = re.compile(r"[Nn]\s*[°º№]\s*|[°º№]")  # N°28, Nº 41, №5 -> " 28"


def norm_addr(raw: str):
    s = elide(to_ascii(_NUMERO_RAW.sub(" ", raw))).lower()
    s = _APOS.sub("", s)
    comps = []
    for c in _COMP.split(s):
        k = _comp_key(c)
        if k in PLACEHOLDER or collapse(k) in REGIONS or k in REGIONS:
            continue
        comps.append(c)
    s = " , ".join(comps)
    # glued digits of the first component holding several digit runs ("#1-38" -> "138",
    # "E-57-A-191" -> "57191"): the sources disagree on separators inside house numbers
    hglue = ""
    for c in comps:
        runs = _DIGITS.findall(c)
        if runs:
            if len(runs) >= 2:
                g = "".join(runs).lstrip("0")
                if 2 <= len(g) <= 8:
                    hglue = g
            break
    s = _NUMBER_SIGN.sub(" ", s)
    s = _ORD.sub(r"\1", s)
    s = _NONALNUM.sub(" ", s)
    s = " ".join(_addr_mixed(t) if not (t.isalpha() or t.isdigit()) else t for t in s.split())
    s = collapse(s)
    toks = []
    nums = []
    words = []
    for t in s.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
            toks.append(t)
            nums.append(t)
            continue
        if t in PLACEHOLDER:
            continue
        t = ORD_WORDS.get(t, t)
        if t.isdigit():
            toks.append(t)
            nums.append(t)
            continue
        if t in UNIT or t in ADDR_STOP:
            continue
        t = STREET.get(t, t)
        toks.append(t)
        words.append(t)
    return {
        "a_full": " ".join(toks),
        "a_words": " ".join(words),
        "a_nums": " ".join(nums),
        "a_hglue": hglue,
        "addr_empty": len(toks) == 0,
    }


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------
def _init_worker():
    load_translit()
    load_regions()


_NAME_COLS = ["n_full", "n_core", "n_alt", "n_ns", "n_alt_ns", "n_skel", "legal", "web", "dba", "native"]
_ADDR_COLS = ["a_full", "a_words", "a_nums", "a_hglue", "addr_empty"]


def _work(args):
    names, addrs = args
    cols = {c: [] for c in _NAME_COLS + _ADDR_COLS}
    for name, addr in zip(names, addrs):
        d = norm_name(name)
        for c in _NAME_COLS:
            cols[c].append(d[c])
        d = norm_addr(addr)
        for c in _ADDR_COLS:
            cols[c].append(d[c])
    return pl.DataFrame(cols)


def normalize_frame(df: pl.DataFrame, n_jobs: int = N_JOBS, chunk: int = 25000) -> pl.DataFrame:
    def batch(i0, i1):
        return [
            (df["business_name"].slice(i, chunk).to_list(), df["business_address"].slice(i, chunk).to_list())
            for i in range(i0, min(i1, df.height), chunk)
        ]

    parts = []
    step = chunk * n_jobs * 2  # bounded amount of work in flight
    with Pool(n_jobs, initializer=_init_worker) as pool:
        for i0 in range(0, df.height, step):
            parts.extend(pool.map(_work, batch(i0, i0 + step), chunksize=1))
    norm = pl.concat(parts)
    return pl.concat([df, norm], how="horizontal_extend")


def normalize_split(split: str, force: bool = False, columns=None) -> pl.DataFrame:
    p = cache(f"{split}_norm.parquet")
    if os.path.exists(p) and not force:
        return pl.read_parquet(p, columns=columns)
    raw = load_raw(split)
    log(f"normalising {split}: {raw.height:,} records")
    df = normalize_frame(raw)
    df.write_parquet(p)
    log(f"wrote {p}")
    return df.select(columns) if columns else df


if __name__ == "__main__":
    import sys

    _init_worker()
    for split in sys.argv[1:] or ["train", "test"]:
        normalize_split(split, force=True)
