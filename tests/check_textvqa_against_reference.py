"""
Differential test: scripts/run_experiment.py TextVQA scorer vs MMF's reference
(EvalAIAnswerProcessor + TextVQAAccuracyEvaluator).

Usage:
    curl -o /tmp/m4c_evaluators.py https://raw.githubusercontent.com/facebookresearch/mmf/main/mmf/utils/m4c_evaluators.py
    python3 tests/check_textvqa_against_reference.py /tmp/m4c_evaluators.py [n_cases]

Exit code 0 = zero mismatches (to 1e-9) over hand-picked vocabulary cases AND
random punctuation/digit/article fuzz cases.
"""
import importlib.util
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_experiment as rx


def load_reference(path):
    spec = importlib.util.spec_from_file_location("m4c_evaluators", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TextVQAAccuracyEvaluator()


VOCAB = ["stop", "Stop", "STOP.", "the stop sign", "a dog", "dog", "dog's", "dog 's", "dogs", "two", "2",
         "ten", "10", "coca-cola", "coca cola", "it's", "its", "don't", "dont", "3.5", "1,000", "1000",
         "yes", "no", "none", "0", "o'clock", "oclock", "new york", "st. louis", "(red)", "red!",
         "an apple", "apple", "foo,bar", "foobar", "foo, bar", "what?", "what", "", " spaced out ",
         "tab\there", "line\nbreak", "A.M.", "10:30", "$5", "50%", "e-mail", "email"]
ALPHABET = list("abcdeSTOP 0123456789") + list(";/[]\"{}()=+\\_-><@`,?!.'") + [" ", " "]


def fuzz_string(rnd):
    return "".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(0, 14)))


def make_cases(n, seed=0):
    rnd = random.Random(seed)
    cases = []
    for _ in range(n):
        pool_fn = (lambda: rnd.choice(VOCAB)) if rnd.random() < 0.6 else (lambda: fuzz_string(rnd))
        mode = rnd.random()
        if mode < 0.25:
            a = pool_fn()
            answers = [a] * 10                                   # unanimous humans
        elif mode < 0.6:
            pool = [pool_fn() for _ in range(rnd.randint(2, 4))]
            answers = [rnd.choice(pool) for _ in range(10)]
        else:
            answers = [pool_fn() for _ in range(10)]
        pred = rnd.choice(answers) if rnd.random() < 0.6 else pool_fn()
        if rnd.random() < 0.3:                                    # perturb case / punctuation of the prediction
            pred = rnd.choice([pred.upper(), pred + ".", pred.title(), " " + pred, pred + "?"])
        cases.append((pred, answers))
    return cases


def main():
    ev = load_reference(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    bad = 0
    for pred, answers in make_cases(n):
        want = ev.eval_pred_list([{"pred_answer": pred, "gt_answers": answers}])
        got = rx.textvqa_score(pred, answers)
        if abs(want - got) > 1e-9:
            bad += 1
            if bad <= 10:
                print("MISMATCH", repr(pred), answers[:3], "ref", want, "mine", got)
    print(f"{n} cases, {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
