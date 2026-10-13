# Business Entity Resolution - team Doer (submitted configuration v13c)

Links every Source-2 / Source-3 business record to at most one Source-1 entity.

The pipeline runs these stages in order:

1. Normalisation and spelling correction.
2. Blocking with a per-country inverted index.
3. Learned pruning; its output is the candidate set.
4. Pair features.
5. An ensemble of three stage-1 LightGBM matchers.
6. A stage-2 re-scorer over all candidate pairs: ownership competition, raw-text evidence, collective partner evidence and name ambiguity.
7. The expected-F0.5 decision per Source-1 entity.

Every choice was validated on cross-fitted entity-family folds. Only the provided data is used: no external data, APIs, geocoding or pretrained models.

The two files in `output/` were produced by exactly the commands below; `matching_results.tsv` is the file uploaded to the leaderboard. Its public score for v13c is 0.985246.

## Environment

* Python 3.12. Tested with 3.12.10 on Windows 11, 16 GB RAM, 16 logical CPUs, no GPU.
* Install the pinned dependencies: `pip install -r requirements.txt` (polars, numpy, lightgbm, rapidfuzz, Unidecode).
* Disk: about 40 GB for the intermediate cache.

## Data location

By default the code expects the organisers' `student_resource` folder at `../../Data Set/student_resource`, relative to this folder, with `dataset/train/*.tsv` and `dataset/test/*.tsv` inside it. Override with environment variables:

```
ER_DATA_ROOT=/path/to/student_resource   # folder that contains dataset/
ER_CACHE=/path/to/cache                  # intermediate files (default: ./cache next to src/)
ER_OUT=/path/to/output                   # where the two TSVs are written (default: ../../output = the zip's output/)
```

## Reproduce both output files end to end

```
cd src
python run_all.py        # base pipeline + stage-1 member A + validation     (~7 h)
python run_release.py    # members B, C + re-scorers + final decision         (~3.5 h)
```

Then validate with the organisers' checker:

```
python "<student_resource>/utils/validate_submission.py" --matching ../../../output/matching_results.tsv \
       --candidate ../../../output/candidate_pairs.tsv --test-dir "<student_resource>/dataset/test" --check-ids
```

Both drivers run every step in its own process and cache its result, so an interrupted run can be restarted:

* `python run_all.py --from <stage>` restarts `run_all.py` at a stage; `python run_all.py --resume` resumes an interrupted retrieval.
* `python run_release.py --from <step>` restarts the release chain at a step.

### `run_all.py` stages

| stage | module | what it does |
|---|---|---|
| translit | `translit.py` | learns the Indic-script → Latin token dictionary from training pairs |
| normalize | `normalize.py` | name / address normalisation of all records |
| correct | `correct.py` | per-country S1 vocabulary; S2/S3 out-of-vocabulary words → nearest S1 word (SymSpell); countries without training labels also get the full street-type words |
| aliases | `aliases.py` | learns S2/S3→S1 address-word aliases from training pairs; full map for the countries it was learned on, vocabulary-gated elsewhere |
| retrieval | `candidates.py`, `blocking.py` | per-country inverted index over rare keys + cheap similarities |
| prune | `pipeline.py`, `prune.py` | cross-fitted LightGBM pruner with adaptive retention → **candidate set** (= `candidate_pairs.tsv`) |
| features | `pipeline.py`, `features.py` | ~140 pair features, streamed to parquet parts |
| match | `cv.py`, `folds.py` | stage-1 member A (511 leaves), cross-fitted on entity-family folds → out-of-fold probabilities |
| match2 | `cv.py`, `stage2.py` | nested stage-2 collective model (validated, not used in the release) |
| predict | `cv.py`, `decide.py` | member-A test probabilities and an intermediate single-model output (overwritten by `run_release.py`) |

### `run_release.py` steps

| step | module | what it does |
|---|---|---|
| ens_b | `ens_b.py` | stage-1 member B (255 leaves, lr 0.08), cross-fitted on the same folds |
| ens_test_ab | `ens_test_ab.py` | test probabilities of members A and B |
| ens_c | `ens_c.py` | members C (1023 leaves) and D (extra-trees); A+B+C adopted by paired bootstrap, D rejected → A+B+C test probabilities |
| rawflags | `rawflags_all.py` (+ `dbl.py`, `edits.py`, `erased.py`) | raw-text flags for every candidate pair (doubled letters in name/address, edit types, legal-form changes, normalisation-erased differences) |
| stack5 | `stack5.py` (+ `stack.py`, `stack4.py`, `stage2.py`) | re-scorer v5 over all candidate pairs, cross-fitted; inputs: member probabilities and spread, ownership competition (rank, gap to the best other candidate, S1 claims), raw-text flags, 52 main features, collective partner features (partner thresholds 0.5 and 0.8), raw name identity / ambiguity |
| stack5_test | `stack5.py test` | test scores of re-scorer v5 |
| stack6 | `stack6.py` | re-scorer v6 (same inputs, other sampling / learning rate) + test scores |
| final | `release_final.py` | s = mean(v5, v6); expected-F0.5 rule with floor 0.6; France: removals plus additions only when the record has an address and exactly the S1's core name → `output/` |

## Expected outputs

| file | rows | SHA-256 of the submitted file |
|---|---|---|
| `matching_results.tsv` | 1,732,544 (5,811,358 links) | `a792eab1e054259eb68e312e60e48487f70ef6e62e86ab13aa7a2908841a3dce` |
| `candidate_pairs.tsv` | 1,732,544 (10,736,059 pairs) | `57e9b460d13255f176c224aad42f4193e83acf14ab3c439599b22e3866f315e5` |

A fresh run should reproduce these counts. The hashes may differ in a handful of rows, because summed floating-point key weights are ranked in a non-deterministic order during retrieval (about 0.005 % of queries). LightGBM multi-threading can also change the last bits of scores.

## Source layout (`src/`)

| file | role |
|---|---|
| `common.py` | paths (environment-configurable), TSV reader (tab separated, **no quote handling**), logging |
| `translit.py`, `regions.py`, `normalize.py` | transliteration dictionary, region vocabularies, name & address normalisation |
| `correct.py`, `aliases.py` | spelling correction against each country's S1 vocabulary; learned address aliases |
| `blocking.py`, `candidates.py`, `prune.py` | blocking keys and inverted index, retrieval, learned pruning (the candidate set) |
| `features.py`, `stage2.py` | pair features; collective partner features |
| `folds.py`, `cv.py`, `metrics.py`, `report.py` | entity-family folds, cross-fitted training and validation, the official macro F0.5, error analysis |
| `decide.py` | exclusivity + expected-F0.5 subset selection per S1 |
| `pipeline.py`, `run_all.py` | stage orchestration of the base pipeline |
| `ens_b.py`, `ens_test_ab.py`, `ens_c.py` | stage-1 ensemble members B and C |
| `dbl.py`, `edits.py`, `erased.py`, `rawflags_all.py` | raw-text evidence erased by normalisation |
| `stack.py`, `stack4.py`, `stack5.py`, `stack6.py` | stage-2 re-scorer feature builders and models |
| `release_final.py`, `run_release.py` | final decision and the release-chain driver |
