"""Run the complete pipeline end to end (data -> blocking -> matching -> output files).

    python run_all.py                    # everything, recomputing every stage
    python run_all.py --from retrieval   # start at a later stage, reusing cached earlier ones
    python run_all.py --resume           # resume an interrupted retrieval, then continue

Stages: translit, normalize, correct, aliases, retrieval, prune, features, match, match2, predict.
Every stage runs in its own subprocess so that its memory is returned to the OS before the next
one starts (the machine this was developed on has 16 GB RAM).
"""
import subprocess
import sys

STAGES = ["translit", "normalize", "correct", "aliases", "retrieval", "prune", "features", "match", "match2", "predict"]


def run_stage(stage: str, resume: bool = False):
    if stage == "translit":
        import translit
        translit.build()
    elif stage == "normalize":
        import normalize
        normalize.load_translit()
        normalize.load_regions()
        for split in ("train", "test"):
            normalize.normalize_split(split, force=True)
    elif stage == "correct":
        import correct
        for split in ("train", "test"):
            correct.correct_split(split, force=True)
    elif stage == "aliases":
        import aliases
        aliases.learn()
        for split in ("train", "test"):
            aliases.apply(split)
    elif stage == "retrieval":
        import candidates
        for split in ("train", "test"):
            candidates.retrieve_split(split, resume=resume, force=not resume)
    else:
        import pipeline as P
        if stage == "prune":
            for split in ("train", "test"):
                P.stage_prune(split, force=True)
        elif stage == "features":
            for split in ("train", "test"):
                P.stage_features(split, force=True)
        elif stage == "match":  # clean validation: family folds, outer + inner stage-1 fits
            import cv
            cv.train_folds(force=True)
            cv.outer_stage1(force=True)
            cv.inner_stage1(force=True)
        elif stage == "match2":  # nested stage 2
            import cv
            cv.nested_stage2(force=True)
        elif stage == "predict":  # choose stage + policy on clean OOF, predict test, manifest
            import cv
            cv.release()


def main():
    if "--stage" in sys.argv:  # child process
        run_stage(sys.argv[sys.argv.index("--stage") + 1], resume="--resume" in sys.argv)
        return
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "translit"
    resume = "--resume" in sys.argv
    if resume:
        start = "retrieval"
    for stage in STAGES[STAGES.index(start):]:
        cmd = [sys.executable, __file__, "--stage", stage] + (["--resume"] if resume and stage == "retrieval" else [])
        print(f"=== stage {stage}", flush=True)
        subprocess.run(cmd, check=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
