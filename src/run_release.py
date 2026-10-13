"""Release chain after run_all.py -> the submitted v13c output files.  Each step runs in its own
process (bounded memory) and caches its results under ../cache.

    python run_release.py                 # all steps
    python run_release.py --from stack5   # restart at a step (ens_b, ens_test_ab, ens_c, rawflags, stack5, stack5_test, stack6, final)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("ens_b", ["ens_b.py"]),                        # stage-1 member B (255 leaves), cross-fitted on the family folds
    ("ens_test_ab", ["ens_test_ab.py"]),            # test probabilities of members A and B
    ("ens_c", ["ens_c.py"]),                        # member C (1023 leaves) [+ D, rejected]; A+B+C test probabilities
    ("rawflags", ["rawflags_all.py", "train", "test"]),  # raw-text flags for every candidate pair
    ("stack5", ["stack5.py", "train"]),             # re-scorer v5, cross-fitted, out-of-fold evaluation
    ("stack5_test", ["stack5.py", "test"]),
    ("stack6", ["stack6.py"]),                      # re-scorer v6 (other sampling), evaluation + test
    ("final", ["release_final.py"]),                # final decision -> output/matching_results.tsv, candidate_pairs.tsv
]
start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else STEPS[0][0]
names = [n for n, _ in STEPS]
for name, cmd in STEPS[names.index(start):]:
    print(f"=== {name}: python {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"step {name} failed with code {r.returncode}")
print("RELEASE DONE", flush=True)
