# Amazon ML Challenge 2026: Business Entity Resolution

**Top 1,000 of 32,000+ teams** (89,000+ registrations) · public leaderboard **macro F0.5 0.985** · peak public rank **416** · CPU only, no external data, no pretrained models.

The task: link every Source-2 / Source-3 business record (noisy, multilingual, full of typos) to at most one Source-1 entity, across the US, India and France. The data has 2.2M entities, 10.3M records to link, and about 26% decoys that must stay unlinked.

## Approach

1. **Normalisation.**
   - Names and addresses are normalised symmetrically. Indic-script names are transliterated with a dictionary learned from training pairs.
   - Legal forms, leetspeak, street types and house-number variants are canonicalised.
   - A SymSpell-style spelling corrector fixes typos against each country's own vocabulary.
2. **Blocking.** A per-country inverted index over rare keys (name tokens and pairs, acronyms, name × locality, house numbers, address words) retrieves about 260M raw candidate pairs.
3. **Learned pruning.** A cross-fitted LightGBM pruner with adaptive retention keeps **about 6.2 candidates per entity at 98.5% recall**.
4. **Matching.** An ensemble of **three LightGBM models** scores each pair on about 140 features: RapidFuzz similarities, soft token alignment, numbers and name uniqueness.
5. **Stage-2 re-scorer.** It adds ownership competition, raw-text evidence and collective "partner" features from the other records linked to the same entity.
6. **Decision.** An exclusivity-aware **expected-F0.5 rule** is applied per entity.

Every choice was validated on **family-grouped, cross-fitted folds**, so no record leaks between folds. Over 13 iterated versions, out-of-fold macro F0.5 reached **0.989**.

| version | main change | public leaderboard |
|---|---|---|
| v1 | baseline | 0.977 |
| v4 | SymSpell correction, leetspeak fix, soft token alignment, stage 2 | 0.982256 |
| v7c | country-gated aliases (France), 3-model ensemble | 0.98299 |
| v8c | raw-evidence link veto and adder | 0.983617 |
| v11a | collective partner features, re-scorer ensemble | 0.985089 |
| **v13c** | final re-scorer ensemble (**submitted**) | **0.985246** |

The full write-up (EDA, features, validation protocol, every change and what it gained) is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Run it

```bash
pip install -r requirements.txt          # Python 3.12: polars, numpy, lightgbm, rapidfuzz, Unidecode
export ER_DATA_ROOT=/path/to/student_resource   # the organisers' data folder (contains dataset/)
export ER_CACHE=/path/to/cache                  # about 40 GB of intermediate files
export ER_OUT=/path/to/output                   # where matching_results.tsv and candidate_pairs.tsv go
cd src
python run_all.py        # normalisation, blocking, pruning, features, stage-1 model A, validation (~7 h)
python run_release.py    # models B and C, re-scorers, final decision (~3.5 h)
```

Every step caches its result, so an interrupted run restarts where it stopped. Stage-by-stage details are in [docs/RUNNING.md](docs/RUNNING.md).

- **Developed on:** Windows 11 with 16 GB RAM, 16 logical CPUs and no GPU. AWS (EC2, S3, SageMaker) was also used during development.
- **Data:** the competition data belongs to the organisers and is **not included** in this repository.

## Author

Abhay Prasad · team **Doer** · Amazon ML Challenge 2026

## License

MIT (see [LICENSE](LICENSE)).
