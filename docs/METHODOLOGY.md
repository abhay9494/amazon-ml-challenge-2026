# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Doer


**Submission Date:** 2026-09-28

---

## 1. Executive Summary

A fully CPU-based pipeline (no GPU, no pretrained language model, no external data):

(i) symmetric normalisation of names and addresses, learned Indic-script transliteration, a
SymSpell-style spelling corrector against each country's own Source-1 vocabulary and learned
address aliases (applied in full only to the countries they were learned on); 

(ii) scalable candidate generation - a per-country inverted index over rare blocking keys (names, name pairs,
acronyms, name × locality, house-number variants, address words) followed by a cross-fitted
LightGBM pruner with **adaptive retention** (≈6.20 candidates per Source-1 entity on test); 

(iii) an ensemble of cross-fitted stage-1 LightGBM matchers on ~140 pair features (A (511 leaves, lr 0.05) + B (255 leaves, lr 0.08) + C (1023 leaves, feature fraction 0.5)); 

(iv) an exclusivity-aware expected-F0.5 decision rule tuned for macro F0.5.

Validation is **family-grouped and cross-fitted**: folds are built from entity families (an S1
entity and all its true S2/S3 records), every candidate row of a record stays in the record's fold,
the pruner is cross-fitted, and a collective stage 2 was evaluated nested. Residual upstream
dependence remains and the local numbers are therefore somewhat optimistic: the pruner that selected
and scored fold B's matcher-training rows was trained with fold A's labels, and the transliteration
and alias dictionaries are learned from all training pairs. Out-of-fold macro F0.5 of the released
configuration (v13c) on the 2.2 M training entities: **0.98894** (v7c 0.98640, US 0.98708, India
0.98539). Observed public-leaderboard scores: v1 0.977, v4 0.982256, **v7c 0.98299**, **v8c 0.983617** (validated gain +0.00062, observed +0.00063), **v11a 0.985089** (validated +0.00242 over v7c, observed +0.0021), **v13c 0.985246** (the submitted version). France has no labels; its handling is
checked with label-free diagnostics only.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA findings (train unless stated):

* 2.21 M S1 entities, 10.3 M S2/S3 records, 7.64 M true links. Every S2/S3 record is linked to at
  most one S1; ~26 % of S2/S3 records are decoys. 5.6 % of S1 entities are singletons; mean 3.46
  links per S1. No link crosses countries.
* **Test differs**: 5.76 S2/S3 records per S1 (train 4.67) and a new country (France, 15 % of S1).
  For US/India the predicted links per S1 and the empty-row rate on test are identical to train
  (the extra test records behave like extra decoys); France is similar.
* **France (test only, label-free analysis):** the generator's French decoy-sibling words are
  *groupe, développement, participations, holding, international, distribution, france* (as an
  extra name word they share the S1 house number in only 0.7–6.7 % of cases, like US
  *holdings/group* ≈ 1 %); French noise words are *fils, services, associés* (≈ 0.87, like US
  *center/services*). French S2/S3 addresses carry the same street-type typos as US ones
  ("aveue", "aenue", "avnue", "bouleard"). "EI" (entreprise individuelle) differences behave like
  the US legal-type sibling decoys (linked 98 % when the house number agrees, rejected otherwise).
  French records are more often uncertain: best-candidate p in [0.1, 0.9) for 6.6 % of French
  queries vs 3.9 % (US) and 2.8 % (India).
* **Scripts**: ~23 % of Indian S2 names are written in an Indic script; the native vocabulary is tiny
  (≈1.5 k word types) and identical in train and test. Native script in addresses occurs only as the
  state component.
* **Name noise**: legal-form variants and moves, dotted acronyms, typos and repeated letters,
  generator leetspeak (measured on aligned training pairs: 1→l 11.9 k vs 1→i 12; 0→o, 5→s, 6→g, 8→b;
  3/4/7/9 never), websites, DBA/alias forms ("X f/k/a Y", "née", "trading as"), unrelated
  pseudo-word alias names ("Avilyra", "Synjax") matched only through the address, **acronyms**
  ("BB" = "Bombay Business", "SAR" = "Styles All Realty LLC"), transliteration variants
  (jai/jay, lakshmi/laxmi, shri/shree/sri), honorifics.
* **Address noise**: component re-ordering, dropped components, state written in full/abbreviated/
  native, "NULL" placeholders, street-type abbreviations and **street-type typos that are real words**
  ("dive"/"davie" for drive, "read" for road), city/suburb swaps ("Dorchester" for Boston,
  "Calcuta" for Kolkata), "X Township"/"X City", ordinals, number formats ("H.NO", "1-38" vs "138",
  "A-76"), **perturbed house numbers** (leading digit dropped 9774→774, last digit dropped
  6437→643, 922 vs 924), French "N°28" (unidecode turns "°" into "deg").
* **Decoys**: sibling businesses - same base name + one extra distinguishing word at the same
  street with a nearby house number; different company type (LLC vs Inc, SARL vs SCI); different
  businesses at the same premises; decoys whose name and address are identical to an S1 record.
  When a candidate's name has one extra *vocabulary* word the pair is almost never a match
  (0/9002 for "group"), whereas noise words ("center", "services", "shri") match ~30 % of the time.

### 2.2 Solution Strategy

**Approach Type:** blocking (inverted index) + learned pruning + gradient-boosted pair classifier +
collective (stacked) classifier + exclusivity-aware decision rule.

**Core Innovation:** learned, train-only text resources (transliteration, spelling correction
against the S1 vocabulary, address aliases); label-free extra-word decoy statistics that transfer to
an unseen country; adaptive candidate retention; a collective second stage; and an end-to-end
leak-free (nested) validation protocol used for every model and threshold decision.

---

## 3. Candidate Generation (Blocking)

Two stages; `candidate_pairs.tsv` is the output of the second one (the exact pairs the matcher
scores, nothing is added afterwards).

**Stage 1 - per-country inverted index over rare keys** (`blocking.py`, `candidates.py`). 

Every record is turned into hashed keys, namespaced by its country label (open label set):

* *name channel*: spelling-corrected core-name tokens (+ alias part of DBA names); unordered pairs of
  name tokens; name token × house-number variant; space-free original, corrected and alias names;
  phonetic consonant skeletons (≥ 5 letters); **acronym keys** (initials of multi-word names and
  short single-token names, × house number or first six address words); **name × locality keys**
  (space-free corrected name × each of the first 8 address words - retrieves a same-name business
  by town/street when the house numbers disagree);
* *address channel*: corrected address words; adjacent word pairs; word × house-number variant.
* *country gating of label-derived resources*: the address aliases are learned from US/India
  pairs and include US-specific word swaps ("maison" → "madison", "cour" → "ct", "pont" → "pt");
  applied to French text they corrupted ≈ 23 k real French words in 1.7 % of French queries (bug
  found by an external review). From v7 they apply in full only to US/India; other countries get
  only entries whose source word is absent from that country's S1 address vocabulary and whose
  target is present (typo fixes such as "avnue" → "ave" pass; 197 of 645). For countries without
  labels the spelling corrector also knows the full street-type words ("aenue" → avenue, formerly
  "tenue").
* *house-number variants*: first three numbers, first number without its last digit, without its
  first digit, and the glued digits of the first numbered component.

Keys held by more than 300 S1 records are not indexed; each query probes its 8 rarest keys per
channel and keeps the union of the top-16 S1 by total, name and address IDF score (≈24 per record);
cheap rapidfuzz similarities are attached. Cost is linear in the number of records.

**Stage 2 - cross-fitted learned pruning with adaptive retention** (`prune.py`). 

A LightGBM model on blocking scores, cheap similarities and within-record context scores every retrieved pair (on
train the pruner that scores a record never saw that record's family). Retention: if the record's
best candidate is confident (p ≥ 0.9) keep it plus near-ties (top-2, ≥ 0.2 × best); otherwise keep
up to 8 alternatives with p ≥ 0.002 (confidence cut 0.95). Chosen on the clean pruner scores (v5
scores; the released v6 rule is the "conf 0.95, k8, 0.002" variant):

| retention rule | candidates / S1 (train) | link recall | candidate-oracle macro F0.5 |
|---|---|---|---|
| fixed top-3 (≥ 0.2 × best, ≥ 0.01) | 4.43 | 97.99 % | 0.99368 |
| top-5 (≥ 0.05 × best, ≥ 0.003) | 4.74 | 98.26 % | 0.99453 |
| adaptive 0.9 / 6 / 0.003 (v5) | 4.82 | 98.31 % | 0.99468 |
| **adaptive 0.95 / 8 / 0.002 (v6)** | **4.97** | **98.38 %** | **0.99492** |
| everything ≥ 0.001 | 5.33 | 98.49 % | 0.99524 |

* **Candidate pairs generated:** 10,736,117 on test (6.20 per S1 entity);
  11,062,999 on train (5.01 per S1). Reduction ratio vs. all cross-source pairs > 99.9999 %.
* **How true matches are protected:** two independent channels, typo-tolerant corrected tokens,
  house-number variants, acronym, alias and name × locality keys; recall measured at every stage
  on train - raw retrieval 98.67 %, after pruning 98.51 % of the 7.64 M true pairs (candidate-oracle
  macro F0.5 0.99594 and 0.99542).

---

## 4. Matching Model

**Features used** (`features.py`, `stage2.py`, ~140 per pair):

* Name: rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler on original and
  spelling-corrected core names, full names, space-free names and phonetic skeletons; alias (DBA)
  name; **Monge-Elkan soft token alignment** (best fuzzy match per token in both directions: mean,
  minimum, IDF-weighted, count below 80); IDF-weighted token coverage; legal-form overlap and
  **legal-type conflict**; **label-free extra-word statistics**; **name uniqueness** (how many S1
  records carry exactly this name); **acronym relation**; out-of-vocabulary share (pseudo-word
  aliases); number of spelling corrections.
* Address: ratio / token-sort / token-set / partial-token-set on original and corrected addresses and
  on their words; soft token alignment; IDF-weighted word coverage; house numbers - first equal,
  first-in-other, Jaccard, unmatched numbers, log difference, **prefix/suffix relation** (dropped
  digits), **glued-digit match**.
* Context: blocking scores, cheap similarities, candidate rank / count / gap, pruner score and
  margin to the best competing S1, best similarity among the record's other candidates, source flag,
  record flags (native script, website, DBA, empty address).
* **Stage 2 (collective)**: the other records whose best stage-1 candidate is the same S1 with
  p ≥ 0.5 ("partners", excluding the record itself): count (all / same source), best partner
  probability, best and most-probable-partner name/address similarity, share of partners with the
  record's / the S1's house number (only among partners that HAVE a number), number of number
  conflicts, exact name+address duplicate, name-only duplicate; plus the stage-1 probability, its
  rank and gap.

*Label-free extra-word statistic.* Per split, among each record's best candidate differing by one
extra name word on the record side with agreeing streets, the share with an equal house number per
extra word: noise words ≈0.75, decoy words ≈0.05 (Pearson 0.93 with the true match rate on train);
no labels, so French words get their own values from the French test records.

**Model type:** an ensemble of LightGBM binary classifiers, each cross-fitted on the family folds
with early stopping on a held-out slice of training families: A (511 leaves, lr 0.05) + B (255 leaves, lr 0.08) + C (1023 leaves, feature fraction 0.5). A alone (511 leaves beat
255 leaves by +0.00006…+0.00010) gives 0.98631; A+B 0.98636 (+0.00005, 95 % CI [+0.00003,
+0.00007]); ABC 0.98640 (vs A+B +0.00004, 95 % CI [+0.00003, +0.00006]).
Test probability = mean over members of the mean of each member's two fold models.

**Threshold selection method:** each S2/S3 record is assigned to its highest-probability candidate
(ties broken deterministically) and kept if p ≥ threshold; the threshold (and optionally an
expected-F0.5 subset rule per S1) is chosen by clean out-of-fold macro F0.5 - released policy (v6 onwards):
stage-1 ensemble, expected-F0.5 subset rule per S1 with probability floor 0.7.

---

## 5. Validation protocol (leak-free)

* **Family folds** (`folds.py`): an S1 entity and all its true S2/S3 records form a family;
  unmatched records are their own family; fold = bit 0 of crc32(family id), inner fold = bit 1.
  Every candidate row of a record inherits the record's fold - no labelled row of a held-out record
  is ever fitted, even when its candidates point to S1 records of both folds.
* **Pruner**: cross-fitted - records of fold k are pruned by the pruner trained on fold 1 − k.
* **Stage 1**: fold k is predicted by the model trained on fold 1 − k.
* **Stage 2 (nested)**: to evaluate fold k, every stage-1 probability it consumes (its own and its
  partners') comes from models that never saw fold-k labels - fold-k rows from the outer model
  trained on fold 1 − k, fold 1 − k rows from inner cross-fitting within fold 1 − k - and the
  stage-2 model itself is fitted on fold 1 − k only.
* Label-derived dictionaries (transliteration, address aliases) are learned once from all training
  pairs; learning the transliteration dictionary from one fold only changes the mapping of 0.17 % of
  held-out native-token occurrences (negligible second-order effect).
* Unlabelled information used across the boundary: all records of the split (retrieval bank, IDF,
  S1 vocabulary for spelling correction, name frequencies, label-free extra-word statistics).
* Uncertainty: paired bootstrap over S1 entities (200 resamples).

---

## 6. Results & Error Analysis

* **Released v13c = mean of re-scorers v5 and v6 (v6: same inputs, different sampling / learning
  rate; +0.00003 [+0.00001, +0.00004] over v5), same France rule as v12c.** Out-of-fold 0.98894.
* **v12c = v12a + French additions whose record has an address and exactly the S1's corrected
  core name** (1,827 links; the same kind of addition is 95 % true in US and India). Other French
  re-scorer additions (decoy-word and category-swap patterns, 0 % true in their Indian analogues) are
  not made.
* **v12a = re-scorer v5 - out-of-fold macro F0.5 0.98891** (+0.00009 [+0.00007,
  +0.00012] over v11a). v5 adds raw name identity / ambiguity inputs: exact casefolded name and
  address equality, how many S1 records and how many S2/S3 records share the record's exact name,
  how many S2/S3 records carry the S1's exact name, core-name frequency among records, and exact
  core-name ties among the record's candidates. Test: +49,058 links (US 26,240, India 22,818),
  −20,673 (US 6,834, India 9,384, France 4,455); France removals only (v12b with French additions
  kept separately).
* **v11a = mean of re-scorers v3 and v4 - out-of-fold macro F0.5 0.98882** (+0.00006
  [+0.00004, +0.00008] over v10a; v4 adds partner features at the strict partner threshold p ≥ 0.8).
* **v10a = stage-2 re-scorer over all candidate pairs - out-of-fold macro F0.5 0.98876**
  (+0.00236 over v7c; EF rule floor 0.6). The veto/adder of v8b/v8c showed that re-scoring the
  first-stage probabilities with information the pair model cannot see transfers to the leaderboard
  one-to-one (+0.00062 validated, +0.00063 observed). v9/v10 generalise it: a LightGBM (127 leaves,
  cross-fitted on the family folds, trained on the out-of-fold ensemble probabilities) re-scores every
  candidate pair from (i) the three member probabilities, their mean and spread; (ii) **ownership
  competition** - rank, gap to the best other candidate of the record, the record's probability mass,
  the S1's candidate count, the best and summed probability of the other records claiming the same S1,
  and the S1's first-pass links; (iii) the 18 raw-text flags (doubled letters in name and address,
  edit types, legal-form changes, normalisation-erased differences); (iv) 52 main-model features; (v)
  collective partner features (similarity of the record to the S1's other confident records: name,
  address, house number, same source, exact duplicates). The gap to the runner-up is the most
  important input. v9 (without (v) and the extra main features): +0.00178; v10 adds +0.00065 on top
  ([+0.00062, +0.00068]). A second pass re-scoring the re-scored probabilities was worse (−0.0005) and
  was rejected. Test changes vs v7c: +48,098 links (US 26,043, India 22,055), −19,860 (US 6,259, India
  9,463, France 4,138). **France: removals only** - additions there cannot be validated without French
  labels (a variant with French additions, v10b, is kept separately).
* **v8c = v8b + raw-evidence link adder (US/India) - out-of-fold macro F0.5 0.98702**
  (+0.00017, 95 % CI [+0.00016, +0.00019] over v8b; +0.00062 over v7c). The mirror of the veto: each
  query's best candidate with p in [0.2, 0.7) that is still unlinked is re-scored by a cross-fitted
  LightGBM from p, the gap to the runner-up, the candidate count, the raw-text flags, address
  agreement and S1 support; candidates scored ≥ 0.75 are linked (12,288 on train, 81 % true;
  10,039 on test: US 4,080, India 5,959). France is excluded: without French labels its addition
  rate (1.33 % per S1 vs 0.56 % on train) could not be validated, and a sample showed decoy-word
  sibling replacements ("Lille SARL France" → "Lille Club SARL") among them.
* **v8b = v7c + general link veto - out-of-fold macro F0.5 0.98685** (+0.00045, 95 % CI
  [+0.00042, +0.00047] over v7c, paired bootstrap over S1 entities). A LightGBM (63 leaves,
  cross-fitted on the family folds) re-scores every released link from the ensemble p plus raw-text
  evidence the main model cannot see (edit types, normalisation-erased differences, doubled letters
  in name and address, address agreement, S1 support); links scored below 0.5 are removed: 4,921 on
  train (81 % of them false merges, 73 % doubled-letter cases), 4,248 on test (US 1,756, India 2,250,
  France 242; 0.25 % per S1 vs 0.22 % on train). The doubled-letter-only veto (v8) gave +0.00034.
  Finding: the most
  confident false merges are unmatched decoys whose name *doubles a letter* of the S1 name ("Solliq
  Metal Works" vs "Soliq Metal Works", "Novvora" vs "Novora", "DGRR" vs "DGR"). The normalisation
  collapses repeated letters, so these become exact matches with p ≈ 1. Released links whose query
  has such a doubled letter are only 55 % true (US 62 %, India 45 %; ≈ 3,700 of the 21,794 false
  merges), while every other normalisation-erased difference (added accents, case, double spaces,
  punctuation, leetspeak digits, word order, inserted/deleted/substituted/transposed letters, legal
  forms, doubled letters in addresses) is ≥ 94 % true. A small LightGBM (15 leaves, cross-fitted on
  the family folds; inputs: ensemble p, raw-edit and erased-difference flags, address agreement, S1
  support) scores these links; links below 0.7 are removed: 4,187 on train (78 % of them false
  merges), 3,281 on test (US 1,227, India 1,966, France 88 - the same rate per S1 as on train).
  Candidates are unchanged.
* **Measured and rejected in the same session (training labels):** linking missing-address records
  through same-name siblings (a unique sibling owner is right only 19–43 % of the time - the
  generator does not reuse family-specific spellings); rescuing S1 entities left without a link
  (−0.0001 … −0.0046: they are mostly true singletons); choosing among same-name owners by family
  size (43 % correct); noise marks as evidence of a true copy (no lift); French same-address
  category-word swaps ("X Club" vs "X Fetes") are decoy siblings, as their Indian analogues
  ("solutions", "trading", "technologies" swaps) are 0 % true, so leaving them unlinked is correct.
* **Released v7c - clean out-of-fold macro F0.5 0.98640** (US 0.98708, India
  0.98539), members ABC: the train representation is that of v6
  (unchanged), member C is cross-fitted on the same family folds; the
  test side changes only the test representation of France
  (alias gating, street-type words). Effect on test: 699 French S1 entities change (265 links
  removed, ≈ 470 added); US/India change only through 0.005 % of queries whose near-tied retrieval
  scores are ordered non-deterministically (156 / 82 S1 rows). This France change cannot be
  measured without French labels - it removes a demonstrable text corruption.
* **v6 single model - clean (leak-free) out-of-fold macro F0.5 on all 2.2 M training entities:
  0.98631** (US 0.98701, India 0.98526); 22,385 false-positive links, 267,384 missed links, 1,885
  singleton entities falsely merged. v6 − v5 (released configurations, paired bootstrap over S1
  entities): **+0.00037 [95 % CI +0.00032, +0.00042]**.
* **Candidate oracle**: raw retrieval 0.99594 (link recall 98.67 %); after pruning = final scored
  set 0.99542 (98.51 %).
* **Error budget** (sequential convention, total 1 − 0.98631 = 0.01369): candidates absent 0.00458;
  candidates present but not accepted 0.00633; false positives 0.00278.
* **Stage 2 (collective, nested):** v5 +0.00015 [+0.00010, +0.00019] over stage 1 (released in
  v5); v6 −0.00045 [−0.00049, −0.00040] (not released). The nested protocol feeds stage 2 with
  weaker quarter-data stage-1 probabilities during training than at evaluation/test time; with the
  stronger v6 stage 1 this mismatch outweighs the collective evidence. The former leaky protocol had
  credited stage 2 with +0.0011 - most of that was leakage.
* **Subgroups (v6, link recall candidates → final):** empty address 78.3 % → 51.1 % (337 k links;
  mostly names shared by several S1 entities), with address 99.4 % → 98.6 %; native-script names
  99.5 % → 98.9 %; S2 96.7 % vs S3 96.3 % final; query name absent from S1 (typos/aliases)
  97.1 % → 94.8 %, unique names 99.8 % → 99.2 %; S1 entities with exactly one true match 0.961,
  larger families 0.984–0.990, true singletons 0.985.
* **Ideas tested and rejected on clean validation:** separate threshold for empty-address records
  (no change); stricter stage-2 partner seed p ≥ 0.8 (−0.00004); sibling-based candidate rescue
  (≈5 k recoverable pairs, not built); a transformer reranker (no GPU; residual errors dominated by
  missing information).
* **Regression cases** (from two external reviews, observed, never tuned on): the empty-address
  "Harris Companies Inc." with two identical S1 names is left unlinked (was a wrong merge); a
  same-premises decoy and an unlabelled near-duplicate stay unlinked; "MT" (= Mccloud's Trading) is
  retrieved since v6 but scored ≈ 0.44 (unlinked); "Service Liberty Inc, 1199 School St" (true
  match of "Liberty Owl Inc, 1201 School St") and "Legacy Piedmont" (no house number) have no
  candidate; "Traders Services, D/2 Ahmedabad" (family of "Traders Farms") is linked to a wrong
  S1 - all generator-hard cases that look like decoys.
* **Common false positives (wrong merges):** sibling decoys with the same (often typo'd) name at a
  nearby house number; pseudo-word alias names at an address shared by several businesses; decoys
  identical to their S1 record after normalisation; records with no address and a name shared by
  several S1 entities.
* **Common false negatives (missed matches):** records with an empty address and a name shared by
  many S1 entities (information-theoretically ambiguous); pseudo-word aliases combined with a
  perturbed house number; true matches whose house number was perturbed exactly like a sibling
  decoy's.
* **Measured but not adopted:** sibling-based candidate rescue (only ≈ 5 k missed records are clearly
  closer to a confidently linked sibling than to their S1 record); a transformer reranker (no GPU;
  residual errors are dominated by missing information rather than text understanding).

---

## 7. Conclusion

Measuring recall and the candidate oracle at every stage, learning text resources from the
training pairs, and validating every decision with a nested, family-grouped protocol turned a
0.977 leaderboard baseline (v4: 0.982256, v7c: 0.98299 on the leaderboard) into 0.98894 out-of-fold
macro F0.5 for v13c (with the residual upstream dependence described in §1). Reading the raw text
of the most confident errors found one decoy mechanism the normalisation had erased (doubled
letters). The remaining gap to the
candidate oracle and to the information-theoretic ceiling is dominated by records that carry no
distinguishing information (empty address + shared name) and by decoys that are identical to real
matches.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` - `cd src && python run_all.py && python run_release.py`
regenerates both output files from the provided data (see its README; every step runs in its own
process and caches its output). Base pipeline: `translit.py`, `normalize.py`, `regions.py`,
`correct.py`, `aliases.py`, `blocking.py`, `candidates.py`, `prune.py`, `features.py`, `stage2.py`,
`folds.py`, `cv.py`, `decide.py`, `metrics.py`, `report.py`, `pipeline.py`, `run_all.py`. Release
chain (`run_release.py`): `ens_b.py`, `ens_test_ab.py`, `ens_c.py` (stage-1 members B and C),
`dbl.py`, `edits.py`, `erased.py`, `rawflags_all.py` (raw-text evidence), `stack.py`, `stack4.py`,
`stack5.py`, `stack6.py` (stage-2 re-scorers), `release_final.py` (final decision).
Dependencies (pinned in `requirements.txt`): polars, numpy, LightGBM (MIT), rapidfuzz (MIT),
Unidecode (GPL-2.0; character tables only, no model). No pretrained model, no external data, no
external lookup, geocoding or entity-resolution service.

### B. Additional Results

**Iterations.** v1–v4 were validated with the earlier, leaky protocol (folds by candidate S1,
in-sample pruner, stage 2 on the same folds) - their numbers are optimistic and not comparable with
v5's clean numbers.

| version | main change | local macro F0.5 | protocol | train candidate recall | test candidates / S1 |
|---|---|---|---|---|---|
| v1 | baseline (public leaderboard **0.977**) | 0.9803 | leaky | 96.73 % | 5.34 |
| v2 | multi-number keys, name uniqueness, glued digits | 0.9842 | leaky | 97.62 % | 5.21 |
| v3 | French fixes (N°, Cie, Ets …), test side only | 0.9842 | leaky | 97.62 % | 5.20 |
| v4 | leetspeak 1→l, SymSpell correction, soft token alignment, stage 2 | 0.9862 | leaky | 97.90 % | 5.49 |
| v5 | aliases, acronyms, name variants, bug fixes, adaptive retention, **leak-free nested CV**, stage 2 | 0.98594 | clean | 98.31 % | 5.91 |
| v6 | name × locality keys, wider acronym keys, retention 0.95/8/0.002, 511-leaf models, stage 1 released | 0.98631 | clean | 98.51 % | 6.20 |
| v6b | + model B (255 leaves) ensemble | 0.98636 | clean | 98.51 % | 6.20 |
| v7c | France: country-gated aliases, street-type words in the corrector; members ABC (test side only); leaderboard 0.98299 | 0.98640 | family-grouped | 98.51 % | 6.20 |
| v8 | + doubled-letter decoy veto (removes 3,281 test links) | 0.98674 | family-grouped | 98.51 % | 6.20 |
| v8b | + general raw-evidence link veto (removes 4,248 test links) | 0.98685 | family-grouped | 98.51 % | 6.20 |
| v8c | + raw-evidence link adder, US/India (adds 10,039 test links); leaderboard 0.983617 | 0.98702 | family-grouped | 98.51 % | 6.20 |
| v9a | stage-2 re-scorer (competition + raw flags), France removals only | 0.98812 | family-grouped | 98.51 % | 6.20 |
| v10a | + collective partner features and 25 more main features | 0.98876 | family-grouped | 98.51 % | 6.20 |
| v11a | mean of re-scorers v3 and v4 (strict-partner features) | 0.98882 | family-grouped | 98.51 % | 6.20 |
| v12a | re-scorer v5: + raw name identity / ambiguity features | 0.98891 | family-grouped | 98.51 % | 6.20 |
| **v13c** | mean of re-scorers v5 and v6; France: removals + identical-core additions with address; **leaderboard 0.985246** | **0.98894** | **family-grouped** | **98.51 %** | **6.20** |

**Label-free test diagnostics (v7c predictions, members ABC)**

| country | S1 | S2+S3 | candidates / S1 | links / S1 | S1 without link | links with different house no. |
|---|---|---|---|---|---|---|
| train US (clean OOF) | 1,323,633 | 6,186,873 | 4.94 | 3.353 | 5.73 % | 12.0 % |
| train India (clean OOF) | 883,188 | 4,133,346 | 5.12 | 3.346 | 5.74 % | 24.7 % |
| test US | 663,106 | 3,817,031 | 5.86 | 3.348 | 5.78 % | 12.2 % |
| test India | 809,986 | 4,717,565 | 6.33 | 3.344 | 5.78 % | 24.8 % |
| test France | 259,452 | 1,434,993 | 6.65 | 3.289 | 5.85 % | 1.6 % |

These are sanity checks, not accuracy measurements for France (no French labels exist).

**Compute.** One CPU machine (16 threads, 16 GB RAM, no GPU): normalisation + correction + aliases
~25 min, retrieval ~70 min, cross-fitted pruning ~50 min, features ~35 min, clean stage-1 (2 outer +
4 inner fits) ~110 min, nested stage 2 ~45 min, release ~20 min. v7 test-side rebuild (correction →
prediction) ~65 min; each extra ensemble member ~15–30 min of training plus ~10 min of prediction.
