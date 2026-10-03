"""
CMPE 249 — Experiment Runner  (Phase 2: vision-encoder quantization extension)
Launches llama-server for one quantization config, runs the selected benchmark
datasets against it, scores the outputs, and writes JSONL + a summary report.

Usage (on Jetson, inside sjsujetsontool shell):
    export LD_LIBRARY_PATH=/opt/llama-cpp-new/build/bin:$LD_LIBRARY_PATH
    cd /Developer/ias-vlm-quantization-edge
    python3 scripts/run_experiment.py                      # interactive menu
    python3 scripts/run_experiment.py --config-num 5 --datasets scienceqa --limit 20
    python3 scripts/run_experiment.py --config-num 9 --datasets blink_depth,blink_spatial,cv_bench,scienceqa --resume

Flags:
    --config-num N     Run config N without the menu (menu is the fallback)
    --datasets A,B     Comma-separated dataset names (default: every dataset in
                       DATASET_SPECS whose folder exists under --data-dir)
    --limit N          Smoke test: first N records per dataset. Outputs go to
                       results/smoke/ so real results are never touched.
    --resume           Continue an interrupted run. Skips completed records only
                       after verifying config/prompt/parser/data fingerprint.
                       A missing output file simply starts fresh.
    --retry-errors     With --resume: also retry records that ended in a
                       request error (default keeps them as errors).
    --overwrite        Replace existing outputs (old files are moved to
                       results/_overwritten/<timestamp>/, never deleted).
    --yes              Skip the "Start experiment? (y/n)" prompt
    --dry-run          Run all pre-flight checks and print the plan, no server
    --data-dir / --results-dir / --port

Without --resume or --overwrite the script REFUSES to touch an existing result
file. This protects the Config 1-8 BLINK / CV-Bench results.

Exit codes: 0 = every selected dataset fully attempted; 3 = incomplete (server
failure or datasets not run); 130 = interrupted; 1 = pre-flight/model-file
failure; 2 = usage error. "Complete" means every record was ATTEMPTED; records
that ended in a request error still count as attempted and appear as errors.

Adding a dataset
    1. Put prepared data in benchmarks/<name>/data.jsonl (+ images/).
    2. If its answer format matches an existing one, add one line to
       DATASET_SPECS. If not, add a new entry to FORMATS (prompt/parse/score)
       and reference it. Unknown formats raise an error; "raw_only" (save raw
       text, score=None) must be requested explicitly.
    3. Edit the loop only if the dataset needs something the registry cannot
       express (extra context, different image handling, a new metric).

Scoring protocol (versioned, see PROTOCOL_VERSION)
    blink_depth / blink_spatial / cv_bench : legacy prompt + legacy parser_v1
        (first character of the reply) so Configs 9-11 stay comparable with
        the Config 2/5/8 results, which saved no raw text. parser_v2 is also
        computed and saved for every MCQ record, so equivalence can be shown.
    scienceqa : context-aware MCQ prompt + parser_v2 (handles "(B)", "B.",
        "Answer: B"; rejects words that merely start with a letter).
    textvqa   : short-answer prompt + TextVQA accuracy, a port of MMF's
        EvalAIAnswerProcessor + TextVQAAccuracyEvaluator (always normalizes;
        leave-one-out over exactly 10 human answers).
    New datasets (scienceqa, textvqa) are validated strictly: unique ids,
    ground truth inside the choices, exactly 10 TextVQA answers.

History
    v2: RESTART_EVERY 75 -> 50, retry after server restart, RAM cleanup,
        raw vs answered accuracy.
    v3 (Sep 23): Pacific timestamps, error diagnostics, container-safe cache
        drop, poison-record tracking.
    v5 (Oct 1, after audit; protocol p2.1): TextVQA scorer switched to MMF's
        evaluator; resume repairs a missing final newline and rejects duplicate
        rec_no; unattempted records are "pending", never "complete"; strict
        validation for new datasets; model checksums + runner revision
        recorded; exit codes.
    v4 (Oct 1, Phase 2): Configs 9-11, runtime file sizes, per-config notes,
        dataset/format registry, MIME + relative paths, raw prediction
        storage, per-dataset metrics (no pooled TOTAL), safe resume/overwrite,
        --config-num/--datasets/--dry-run.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Time (Pacific, DST-aware when tzdata exists; falls back to fixed PDT) ───

try:
    from zoneinfo import ZoneInfo
    _PACIFIC = ZoneInfo("America/Los_Angeles")
except Exception:
    _PACIFIC = timezone(timedelta(hours=-7))   # PDT; becomes -8 after Nov 1


def now_pacific():
    return datetime.now(_PACIFIC)

# ── Constants ────────────────────────────────────────────────────────────────

PROTOCOL_VERSION = "p2.1"

MODEL_DIR    = Path("/Developer/models/cosmos-reason2-2b")
LLAMA_SRV    = Path("/opt/llama-cpp-new/build/bin/llama-server")
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # /Developer/ias-vlm-quantization-edge

SERVER_CTX    = 2048    # unchanged from Configs 1-8
SERVER_NGL    = 999
TEMPERATURE   = 0.0
RESTART_EVERY = 50      # restart server every N records to avoid memory leak
REQUEST_TIMEOUT = 60

# Runtime settings, filled in by configure()
DATA_DIR    = PROJECT_ROOT / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR     = RESULTS_DIR
PORT        = 8080
BASE_URL    = f"http://localhost:{PORT}"

# ── Config table ─────────────────────────────────────────────────────────────
# Config numbers are permanent: old result filenames and provenance depend on them.

_NOTE_Q4 = ("Unofficial Q4_0 mmproj (llama-quantize from the F16 mmproj). "
            "Output quality may be lower than standard configs.")
_NOTE_K = ("Experimental K-quant mmproj generated Oct 1 with llama-quantize "
           "from the F16 mmproj; passed load + one-image gate.")

CONFIGS = [
    {
        "num":      1,
        "label":    "F16 vision  +  F16 language  [BASELINE — highest quality, slowest]",
        "short":    "F16v_F16l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-F16.gguf",
        "model":    "Cosmos-Reason2-2B-F16.gguf",
        "official": True,
        "note":     "Documented OOM on the 8 GB Jetson; do not rerun just to fill a cell.",
    },
    {
        "num":      2,
        "label":    "F16 vision  +  Q8_0 language",
        "short":    "F16v_Q8l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-F16.gguf",
        "model":    "Cosmos-Reason2-2B-Q8_0.gguf",
        "official": True,
        "note":     "",
    },
    {
        "num":      3,
        "label":    "F16 vision  +  Q4_K_M language",
        "short":    "F16v_Q4l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-F16.gguf",
        "model":    "Cosmos-Reason2-2B-Q4_K_M.gguf",
        "official": True,
        "note":     "",
    },
    {
        "num":      4,
        "label":    "Q8_0 vision  +  F16 language",
        "short":    "Q8v_F16l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q8_0.gguf",
        "model":    "Cosmos-Reason2-2B-F16.gguf",
        "official": True,
        "note":     "",
    },
    {
        "num":      5,
        "label":    "Q8_0 vision  +  Q8_0 language",
        "short":    "Q8v_Q8l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q8_0.gguf",
        "model":    "Cosmos-Reason2-2B-Q8_0.gguf",
        "official": True,
        "note":     "",
    },
    {
        "num":      6,
        "label":    "Q8_0 vision  +  Q4_K_M language  [fastest confirmed config]",
        "short":    "Q8v_Q4l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q8_0.gguf",
        "model":    "Cosmos-Reason2-2B-Q4_K_M.gguf",
        "official": True,
        "note":     "",
    },
    {
        "num":      7,
        "label":    "Q4_0 vision  +  Q4_K_M language  [STRETCH — unofficial Q4 mmproj]",
        "short":    "Q4v_Q4l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q4_0.gguf",
        "model":    "Cosmos-Reason2-2B-Q4_K_M.gguf",
        "official": False,
        "note":     _NOTE_Q4,
    },
    {
        "num":      8,
        "label":    "Q4_0 vision  +  Q8_0 language  [Q4 vision isolation test]",
        "short":    "Q4v_Q8l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q4_0.gguf",
        "model":    "Cosmos-Reason2-2B-Q8_0.gguf",
        "official": False,
        "note":     _NOTE_Q4,
    },
    {
        "num":      9,
        "label":    "Q6_K vision  +  Q8_0 language  [PRIMARY — experimental mmproj]",
        "short":    "Q6v_Q8l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q6_K.gguf",
        "model":    "Cosmos-Reason2-2B-Q8_0.gguf",
        "official": False,
        "note":     _NOTE_K,
    },
    {
        "num":      10,
        "label":    "Q5_K_M vision  +  Q8_0 language  [PRIMARY — experimental mmproj]",
        "short":    "Q5v_Q8l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q5_K_M.gguf",
        "model":    "Cosmos-Reason2-2B-Q8_0.gguf",
        "official": False,
        "note":     _NOTE_K,
    },
    {
        "num":      11,
        "label":    "Q3_K_M vision  +  Q8_0 language  [OPTIONAL STRESS TEST]",
        "short":    "Q3v_Q8l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q3_K_M.gguf",
        "model":    "Cosmos-Reason2-2B-Q8_0.gguf",
        "official": False,
        "note":     _NOTE_K + " Optional; drop first if the schedule is tight.",
    },
]


def get_config(num):
    for c in CONFIGS:
        if c["num"] == num:
            return c
    return None


def file_size_bytes(path):
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def fmt_size(nbytes):
    """Human size from a byte count; 'MISSING' when the file is absent."""
    if nbytes is None:
        return "MISSING"
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024 ** 3:.1f} GB"
    return f"{nbytes / 1024 ** 2:.0f} MB"


def model_size_str(filename):
    return fmt_size(file_size_bytes(MODEL_DIR / filename))

# ── Image helpers ────────────────────────────────────────────────────────────

_SUFFIX_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def detect_mime(data, path=""):
    """MIME type from magic bytes; falls back to the suffix, then JPEG."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    return _SUFFIX_MIME.get(Path(str(path)).suffix.lower(), "image/jpeg")


def image_data_uri(path):
    with open(path, "rb") as f:
        data = f.read()
    return f"data:{detect_mime(data, path)};base64,{base64.b64encode(data).decode('utf-8')}"


def resolve_image_paths(record, dataset_dir):
    """Relative paths resolve against the dataset folder; absolute ones
    (the legacy BLINK / CV-Bench files) are used as-is."""
    out = []
    for p in record["image_paths"]:
        pp = Path(p)
        out.append(pp if pp.is_absolute() else Path(dataset_dir) / pp)
    return out

# ── Prompts ──────────────────────────────────────────────────────────────────

_MCQ_INSTRUCTION = "Answer with only the letter of the correct choice. Do not explain."


def _choice_block(choices):
    letters = [chr(ord("A") + i) for i in range(26)]
    return "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))


def mcq_prompt_legacy(record):
    """EXACT prompt used for Configs 1-8 on BLINK / CV-Bench. Do not edit."""
    return (
        f"{record['question']}\n\n{_choice_block(record['choices'])}\n\n"
        "Answer with only the letter of the correct choice. "
        "Do not explain."
    )


def mcq_prompt_context(record):
    """ScienceQA: optional hint first, then question, choices, instruction.
    Never includes lecture / solution (they leak reasoning or the answer)."""
    ctx = (record.get("context") or "").strip()
    head = f"Context: {ctx}\n\n" if ctx else ""
    return f"{head}{record['question']}\n\n{_choice_block(record['choices'])}\n\n{_MCQ_INSTRUCTION}"


def vqa_prompt_short(record):
    return f"{record['question']}\n\nAnswer with a single word or short phrase. Do not explain."


def plain_prompt(record):
    return str(record["question"])

# ── Parsers (raw model text -> prediction) ───────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def parse_letter_v1(text, record=None):
    """LEGACY parser: first character, uppercased. Used for the protocol that
    produced Configs 1-8. Known weakness: 'Apple' -> 'A'. Kept for
    comparability, never 'fixed'."""
    text = (text or "").strip()
    return text[0].upper() if text else ""


def parse_letter_v2(text, record=None):
    """Stricter parser: accepts 'B', '(B)', 'B.', 'B)', 'B:', 'Answer: B',
    'The answer is (B)', and a lone letter on the first line. Rejects words
    that merely start with a letter. Letters beyond the choice count -> ''."""
    n = len((record or {}).get("choices", [])) or 26
    valid = {chr(ord("A") + i) for i in range(min(n, 26))}
    t = _THINK_RE.sub("", text or "")
    t = t.replace("*", "").replace("`", "").strip()
    if not t:
        return ""

    candidates = []
    first_line = t.splitlines()[0].strip()
    if len(first_line) == 1 and first_line.isalpha():
        candidates.append(first_line.upper())
    m = re.match(r"^\(([A-Za-z])\)", t)
    if m:
        candidates.append(m.group(1).upper())
    m = re.match(r"^([A-Z])[.):](?:\s|$)", t)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"(?i:\banswer(?:\s+is)?)\s*[:\-]?\s*\(?([A-Z])\)?(?![A-Za-z])", t)
    if m:
        candidates.append(m.group(1))

    for c in candidates:
        if c in valid:
            return c
    return ""


def parse_short(text, record=None):
    """Keep the COMPLETE short answer (first non-empty line, 'Answer:' prefix
    and think blocks removed). Normalization happens at scoring time only."""
    t = _THINK_RE.sub("", text or "").strip()
    if not t:
        return ""
    first = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
    first = re.sub(r"^(?i:answer)\s*[:\-]\s*", "", first)
    return " ".join(first.split())


def parse_raw(text, record=None):
    return (text or "").strip()

# ── TextVQA normalization + scoring (port of MMF's TextVQA evaluator) ────────
# Reference: facebookresearch/mmf  mmf/utils/m4c_evaluators.py
#   EvalAIAnswerProcessor  +  TextVQAAccuracyEvaluator
# This is TextVQA's own metric. It differs from the general VQA evaluator
# (GT-Vision-Lab vqaEval.py) in two ways: it ALWAYS normalizes both the
# prediction and the human answers (even when all ten agree), and it has an
# extra tokenization step (lowercase, drop ',' and '?', split "'s").
# The contraction table is identical in both evaluators and was extracted
# programmatically, not retyped. tests/check_vqa_against_reference.py compares
# this port against the MMF module on thousands of generated cases.

_TV_CONTRACTIONS = {
    'aint': "ain't", 'arent': "aren't", 'cant': "can't",
    'couldve': "could've", 'couldnt': "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", 'didnt': "didn't", 'doesnt': "doesn't",
    'dont': "don't", 'hadnt': "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", 'hasnt': "hasn't", 'havent': "haven't",
    'hed': "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    'hes': "he's", 'howd': "how'd", 'howll': "how'll",
    'hows': "how's", "Id've": "I'd've", "I'dve": "I'd've",
    'Im': "I'm", 'Ive': "I've", 'isnt': "isn't",
    'itd': "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    'itll': "it'll", "let's": "let's", 'maam': "ma'am",
    'mightnt': "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    'mightve': "might've", 'mustnt': "mustn't", 'mustve': "must've",
    'neednt': "needn't", 'notve': "not've", 'oclock': "o'clock",
    'oughtnt': "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", 'shant': "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", 'shouldve': "should've",
    'shouldnt': "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": 'somebodyd', "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    'somebodyll': "somebody'll", 'somebodys': "somebody's", 'someoned': "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", 'someonell': "someone'll",
    'someones': "someone's", 'somethingd': "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", 'somethingll': "something'll", 'thats': "that's",
    'thered': "there'd", "thered've": "there'd've", "there'dve": "there'd've",
    'therere': "there're", 'theres': "there's", 'theyd': "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", 'theyll': "they'll",
    'theyre': "they're", 'theyve': "they've", 'twas': "'twas",
    'wasnt': "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    'weve': "we've", 'werent': "weren't", 'whatll': "what'll",
    'whatre': "what're", 'whats': "what's", 'whatve': "what've",
    'whens': "when's", 'whered': "where'd", 'wheres': "where's",
    'whereve': "where've", 'whod': "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", 'wholl': "who'll", 'whos': "who's",
    'whove': "who've", 'whyll': "why'll", 'whyre': "why're",
    'whys': "why's", 'wont': "won't", 'wouldve': "would've",
    'wouldnt': "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    'yall': "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    'youd': "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    'youll': "you'll", 'youre': "you're", 'youve': "you've",
}
_TV_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_TV_ARTICLES = ["a", "an", "the"]
_TV_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_TV_COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
_TV_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
             ">", "<", "@", "`", ",", "?", "!"]


def textvqa_word_tokenize(word):
    word = word.lower()
    word = word.replace(",", "").replace("?", "").replace("'s", " 's")
    return word.strip()


def textvqa_process_punctuation(in_text):
    out_text = in_text
    for p in _TV_PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (re.search(_TV_COMMA_STRIP, in_text) is not None):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    # MMF passes re.UNICODE (== 32) in the *count* slot of sub(); reproduced as-is.
    return _TV_PERIOD_STRIP.sub("", out_text, count=32)


def textvqa_process_digit_article(in_text):
    words = []
    for w in in_text.lower().split():
        w = _TV_NUMBER_MAP.get(w, w)
        if w not in _TV_ARTICLES:
            words.append(w)
    words = [_TV_CONTRACTIONS.get(w, w) for w in words]
    return " ".join(words)


def textvqa_normalize(item):
    item = textvqa_word_tokenize(str(item))
    item = item.replace("\n", " ").replace("\t", " ").strip()
    item = textvqa_process_punctuation(item)
    item = textvqa_process_digit_article(item)
    return item


def textvqa_score(pred, answers):
    """TextVQA accuracy for one question: average over the 10 leave-one-out
    comparisons of min(1, matches / 3) after normalizing everything.
    Exactly ten human answers are required (as in the reference)."""
    if len(answers) != 10:
        raise ValueError(f"TextVQA scoring needs exactly 10 human answers, got {len(answers)}")
    gts = [textvqa_normalize(a) for a in answers]
    res = textvqa_normalize(pred)
    accs = []
    for i in range(10):
        matches = sum(1 for j, g in enumerate(gts) if j != i and g == res)
        accs.append(min(1.0, matches / 3.0))
    return sum(accs) / 10.0

# ── Scorers ──────────────────────────────────────────────────────────────────


def normalize_gt_letter(answer):
    gt = str(answer).strip().upper().strip("()")
    if gt.isdigit():
        gt = chr(ord("A") + int(gt))
    return gt


def score_mcq(pred, record):
    gt = normalize_gt_letter(record["answer"])
    ok = (pred == gt)
    return {"score": 1.0 if ok else 0.0, "correct": ok, "gt_answer": gt}


def score_textvqa(pred, record):
    return {"score": textvqa_score(pred, record["answers"]), "correct": None, "gt_answer": None}

# ── Format / dataset registry ────────────────────────────────────────────────

FORMATS = {
    "mcq": {
        "prompts":   {"legacy": mcq_prompt_legacy, "context": mcq_prompt_context},
        "parsers":   {"v1": parse_letter_v1, "v2": parse_letter_v2},
        "scorer":    score_mcq,
        "scorer_id": "mcq_exact_v1",
        "normalize": lambda s: s,
        "metric":    "accuracy",
        "required":  ["question", "choices", "answer"],
    },
    "open_vqa": {
        "prompts":   {"short": vqa_prompt_short},
        "parsers":   {"short": parse_short},
        "scorer":    score_textvqa,
        "scorer_id": "textvqa_mmf_v1",
        "normalize": textvqa_normalize,
        "metric":    "TextVQA accuracy",
        "required":  ["question", "answers"],
    },
    # Opt-in only: stores raw text, score=None, summary marked unscored.
    "raw_only": {
        "prompts":   {"plain": plain_prompt},
        "parsers":   {"raw": parse_raw},
        "scorer":    None,
        "scorer_id": "none",
        "normalize": lambda s: s,
        "metric":    "unscored",
        "required":  ["question"],
    },
}

DATASET_SPECS = {
    # "strict": True = new datasets get hard validation (unique ids, ground truth
    # inside the choices). Legacy datasets only warn so existing data keeps working.
    # Legacy protocol: identical prompt + first-character parser as Configs 1-8
    "blink_depth":   {"format": "mcq", "prompt": "legacy",  "parser": "v1", "max_tokens": 16},
    "blink_spatial": {"format": "mcq", "prompt": "legacy",  "parser": "v1", "max_tokens": 16},
    "cv_bench":      {"format": "mcq", "prompt": "legacy",  "parser": "v1", "max_tokens": 16},
    # New datasets
    "scienceqa":     {"format": "mcq", "prompt": "context", "parser": "v2", "max_tokens": 16, "strict": True},
    "textvqa":       {"format": "open_vqa", "prompt": "short", "parser": "short", "max_tokens": 32, "strict": True},
    "textvqa_1500":  {"format": "open_vqa", "prompt": "short", "parser": "short", "max_tokens": 32, "strict": True},
}


def validate_registry(specs=None, formats=None):
    """Unknown formats / prompts / parsers are errors, never silent fallbacks."""
    specs = DATASET_SPECS if specs is None else specs
    formats = FORMATS if formats is None else formats
    for name, spec in specs.items():
        fmt = spec.get("format")
        if fmt not in formats:
            raise ValueError(f"dataset '{name}': unknown format {fmt!r} "
                             f"(known: {', '.join(formats)})")
        if spec.get("prompt") not in formats[fmt]["prompts"]:
            raise ValueError(f"dataset '{name}': unknown prompt {spec.get('prompt')!r} for format {fmt}")
        if spec.get("parser") not in formats[fmt]["parsers"]:
            raise ValueError(f"dataset '{name}': unknown parser {spec.get('parser')!r} for format {fmt}")
        if not isinstance(spec.get("max_tokens"), int) or spec["max_tokens"] <= 0:
            raise ValueError(f"dataset '{name}': max_tokens must be a positive int")


def parse_dataset_list(arg, specs=None):
    specs = DATASET_SPECS if specs is None else specs
    names = [d.strip() for d in arg.split(",") if d.strip()]
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)} "
                         f"(known: {', '.join(specs)})")
    if len(set(names)) != len(names):
        raise ValueError("duplicate dataset names in --datasets")
    return names


def mcq_gt_in_range(rec):
    """True if the ground-truth letter is one of the record's choices."""
    gt = normalize_gt_letter(rec["answer"])
    n = len(rec["choices"])
    return len(gt) == 1 and "A" <= gt <= chr(ord("A") + min(n, 26) - 1)


def validate_record(spec, rec, strict=False):
    """Return an error string, or None if the record is usable."""
    fmt = FORMATS[spec["format"]]
    for key in ["idx", "image_paths"] + fmt["required"]:
        if key not in rec:
            return f"missing field '{key}'"
    if not rec["image_paths"]:
        return "empty image_paths"
    if spec["format"] == "mcq" and not rec["choices"]:
        return "empty choices"
    if spec["format"] == "open_vqa":
        ans = rec["answers"]
        if not (isinstance(ans, list) and len(ans) == 10 and all(isinstance(a, str) for a in ans)):
            return "answers must be a list of exactly 10 strings"
    if rec.get("format") not in (None, spec["format"]):
        return f"record format {rec.get('format')!r} != dataset format {spec['format']!r}"
    if strict and spec["format"] == "mcq" and not mcq_gt_in_range(rec):
        return f"ground truth {rec['answer']!r} is outside the {len(rec['choices'])} choices"
    return None

# ── Server ────────────────────────────────────────────────────────────────────

def kill_stale_servers():
    """Kill any leftover llama-server processes."""
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(1)


def start_server(config):
    kill_stale_servers()
    cmd = [
        str(LLAMA_SRV),
        "--model",        str(MODEL_DIR / config["model"]),
        "--mmproj",       str(MODEL_DIR / config["mmproj"]),
        "--ctx-size",     str(SERVER_CTX),
        "--n-gpu-layers", str(SERVER_NGL),
        "--port",         str(PORT),
        "--log-disable",
    ]
    print("\n  Starting llama-server ...")
    print(f"  Vision encoder  : {config['mmproj']}  ({model_size_str(config['mmproj'])})")
    print(f"  Language decoder: {config['model']}  ({model_size_str(config['model'])})\n")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_server(timeout=120):
    print("  Waiting for server", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"  ready in {time.time()-start:.1f}s  [{mem_str()}]")
                    return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    print("\n  ERROR: server not ready in time.")
    return False


def stop_server(proc):
    print("\n  Stopping llama-server ...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    kill_stale_servers()   # kill any stragglers
    print("  Server stopped.")


def cleanup_ram():
    """Release GPU and system memory after experiment ends."""
    print("\n  Cleaning up memory ...")
    kill_stale_servers()
    # Try to drop kernel page caches — may fail in containers with read-only /proc
    try:
        subprocess.run(["sync"], timeout=5)
        result = subprocess.run(
            ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            timeout=5, capture_output=True
        )
        if result.returncode == 0:
            print("  Page caches dropped.")
        else:
            print("  Cache drop skipped (read-only /proc — normal in containers).")
    except Exception as e:
        print(f"  Cache drop skipped: {e}")
    print("  Cleanup complete.")

# ── RAM monitoring ────────────────────────────────────────────────────────────

def get_mem_mb():
    """Read available memory from /proc/meminfo. Returns (avail_MB, total_MB) or (None, None)."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:"):
                    info[parts[0]] = int(parts[1])  # kB
            total = info.get("MemTotal:", 0) // 1024
            avail = info.get("MemAvailable:", 0) // 1024
            return avail, total
    except Exception:
        return None, None


def mem_str():
    """Short string like 'mem=1842/7764MB' for log output."""
    avail, total = get_mem_mb()
    if avail is None:
        return "mem=N/A"
    return f"mem={avail}/{total}MB"

# ── Request / response ────────────────────────────────────────────────────────

def get_image_sizes(image_paths):
    """Return list of (path, file_size_KB) for error diagnostics."""
    sizes = []
    for p in image_paths:
        try:
            sizes.append((str(p), round(os.path.getsize(p) / 1024, 1)))
        except OSError:
            sizes.append((str(p), -1))
    return sizes


def call_server(image_paths, prompt, max_tokens):
    try:
        content = [{"type": "image_url", "image_url": {"url": image_data_uri(p)}}
                   for p in image_paths]
        content.append({"type": "text", "text": prompt})
        payload = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
        return {"error": str(e), "error_type": type(e).__name__}


def extract_text(response):
    """Full raw model text ('' when absent)."""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def extract_finish_reason(response):
    try:
        return response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def extract_tok_s(response):
    try:
        return response["timings"]["predicted_per_second"]
    except (KeyError, TypeError):
        return 0.0

# ── Result lines ──────────────────────────────────────────────────────────────

def build_line(config, ds_name, spec, record, rec_no, response, retried, mem_avail,
               error_msg="", error_type=None):
    """One JSONL result line. Always keeps the raw model text."""
    fmt = FORMATS[spec["format"]]
    parse_fn = fmt["parsers"][spec["parser"]]
    failed = bool(error_msg)

    raw = "" if failed else extract_text(response)
    predicted = "" if failed else parse_fn(raw, record)

    line = {
        "idx":                   record["idx"],
        "rec_no":                rec_no,
        "dataset":               ds_name,
        "config":                config["short"],
        "format":                spec["format"],
        "task":                  record.get("task", ""),
        "question":              record["question"],
    }
    if "choices" in record:
        line["choices"] = record["choices"]
    if record.get("context"):
        line["context"] = record["context"]
    if "answers" in record:
        line["answers"] = record["answers"]

    line.update({
        "predicted":             predicted,
        "prediction_raw":        raw,
        "prediction_normalized": "" if failed else fmt["normalize"](predicted),
        "parser":                spec["parser"],
        "score":                 None,
        "correct":               None,
        "gt_answer":             None,
        "tok_s":                 0.0 if failed else extract_tok_s(response),
        "finish_reason":         None if failed else extract_finish_reason(response),
        "retried":               retried,
        "error":                 error_msg if failed else None,
        "error_type":            error_type if failed else None,
        "mem_avail_mb":          mem_avail,
        "image_paths":           record["image_paths"],
        "protocol_version":      PROTOCOL_VERSION,
    })

    if spec["format"] == "mcq":
        line["pred_parser_v1"] = "" if failed else parse_letter_v1(raw, record)
        line["pred_parser_v2"] = "" if failed else parse_letter_v2(raw, record)
        line["gt_answer"] = normalize_gt_letter(record["answer"])

    if not failed and fmt["scorer"] is not None:
        line.update(fmt["scorer"](predicted, record))
    return line

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(lines, fmt_name, target=None):
    """Single source of truth for every reported number: derived from the
    JSONL lines, so resumed runs report the whole dataset, not one session.

    Lines with error_type == 'ServerUnavailable' are PLACEHOLDERS for records
    that were never attempted. They are counted as pending, never as errors
    or answers, and they keep the dataset from being reported as complete."""
    fmt = FORMATS[fmt_name]
    pending_lines = [l for l in lines if l.get("error_type") == "ServerUnavailable"]
    tried = [l for l in lines if l.get("error_type") != "ServerUnavailable"]
    errs = [l for l in tried if l.get("error")]
    ok = [l for l in tried if not l.get("error")]
    done = len(tried)
    answered = len(ok)
    target = target if target is not None else len(lines)
    is_scored = fmt["scorer"] is not None
    score_sum = sum(l["score"] for l in ok if l.get("score") is not None)
    toks = [l["tok_s"] for l in ok if l.get("tok_s", 0) > 0]
    complete = done >= target
    m = {
        "metric":       fmt["metric"],
        "target":       target,
        "total":        done,                              # records actually attempted
        "pending":      max(target - done, 0),             # never attempted (server failure, interrupt)
        "placeholders": len(pending_lines),
        "complete":     complete,
        "status":       "complete" if complete else ("not attempted" if done == 0 else "partial"),
        "answered":     answered,
        "errors":       len(errs),
        "retries":      sum(1 for l in tried if l.get("retried")),
        "unparsed":     sum(1 for l in ok if not l.get("predicted")),
        "correct":      sum(1 for l in ok if l.get("correct") is True) if fmt_name == "mcq" else None,
        "score_sum":    round(score_sum, 4) if is_scored else None,
        "score_raw":    (score_sum / done) if (is_scored and done) else None,
        "score_answered": (score_sum / answered) if (is_scored and answered) else None,
        "avg_tok_s":    round(sum(toks) / len(toks), 1) if toks else 0.0,
        "parser_disagreements": None,
    }
    if fmt_name == "mcq":
        m["parser_disagreements"] = sum(
            1 for l in ok if l.get("pred_parser_v1") != l.get("pred_parser_v2"))
    return m

# ── Data loading, fingerprints, resume ───────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# SHA-256 of every GGUF as frozen in phase0/phase1 manifests (Oct 1). A mismatch is
# reported loudly but does not block a run: you may legitimately regenerate a file.
FROZEN_SHA256 = {
    "Cosmos-Reason2-2B-F16.gguf": "e8a8ed8335f011e64afd2b0a6b4376c78e05d1653c539e35139285ebc4cdbe32",
    "Cosmos-Reason2-2B-Q8_0.gguf": "2a36ee7a3cb47d28794d2da3686e31e81502b76ec9a2b22cce709de553bb8c82",
    "Cosmos-Reason2-2B-Q4_K_M.gguf": "3ed011641891e81fe654328071de251718832e7deafef579033485d33082e4fd",
    "mmproj-Cosmos-Reason2-2B-F16.gguf": "8d3284c340d6a9c9237d56f2fc42f2df50d3761f539f3f01be964cefb0b73916",
    "mmproj-Cosmos-Reason2-2B-Q8_0.gguf": "b03075492e73a39309d758bc33e0eb051af0d5976fda4984fa1dd2be3e5b1c07",
    "mmproj-Cosmos-Reason2-2B-Q4_0.gguf": "ea7e9a9f637dabdf386875353fdfb0d4718bdb909e02c894163c04fc642a92b6",
    "mmproj-Cosmos-Reason2-2B-Q6_K.gguf": "039b9b1c097f8f300664b69387d1ad0eaf9735eebd13ff5791ae65bf231262e8",
    "mmproj-Cosmos-Reason2-2B-Q5_K_M.gguf": "7cb96c2aa395974a8d296912f2b896a3b8e594aa4f5ff420848e9b183e242b06",
    "mmproj-Cosmos-Reason2-2B-Q3_K_M.gguf": "cd7ccd53e47c25fcfacd27f6cd3a1129cc437a73d4b9446a05dc02e59553cca9"
}


def file_sha256_cached(path, cache_path=None):
    """SHA-256 of a (large) model file, cached by path + size + mtime so the
    ~3 GB hash is paid once, not on every launch."""
    path = Path(path)
    st = path.stat()
    key = str(path.resolve())
    cache = read_json(cache_path) if cache_path else None
    cache = cache if isinstance(cache, dict) else {}
    hit = cache.get(key)
    if hit and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns:
        return hit["sha256"]
    digest = sha256_file(path)
    if cache_path:
        cache[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        try:
            write_json(cache_path, cache)
        except OSError:
            pass
    return digest


def compute_model_hashes(config, cache_path=None):
    out = {}
    for field, fname in (("mmproj", config["mmproj"]), ("model", config["model"])):
        digest = file_sha256_cached(MODEL_DIR / fname, cache_path)
        frozen = FROZEN_SHA256.get(fname)
        out[f"{field}_sha256"] = digest
        out[f"{field}_frozen_match"] = None if frozen is None else (digest == frozen)
    return out


_RUNNER_REV = None


def runner_revision():
    """Which code produced a result. Recorded for provenance; deliberately NOT
    part of the resume fingerprint (you will keep editing this file between
    sessions). Anything that changes results must bump PROTOCOL_VERSION."""
    global _RUNNER_REV
    if _RUNNER_REV is None:
        rev = {"protocol_version": PROTOCOL_VERSION, "script_sha256": None, "git_commit": None}
        try:
            rev["script_sha256"] = sha256_file(__file__)[:16]
        except OSError:
            pass
        try:
            r = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                rev["git_commit"] = r.stdout.strip() or None
        except Exception:
            pass
        _RUNNER_REV = rev
    return _RUNNER_REV


def build_fingerprint(config, ds_name, spec, data_sha, manifest_sha, model_hashes=None):
    fmt = FORMATS[spec["format"]]
    mh = model_hashes or {}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "config_num":       config["num"],
        "config_short":     config["short"],
        "mmproj":           config["mmproj"],
        "model":            config["model"],
        "mmproj_sha256":    mh.get("mmproj_sha256"),
        "model_sha256":     mh.get("model_sha256"),
        "dataset":          ds_name,
        "format":           spec["format"],
        "prompt":           spec["prompt"],
        "parser":           spec["parser"],
        "scorer":           fmt["scorer_id"],
        "max_tokens":       spec["max_tokens"],
        "temperature":      TEMPERATURE,
        "ctx_size":         SERVER_CTX,
        "n_gpu_layers":     SERVER_NGL,
        "data_sha256":      data_sha,
        "manifest_sha256":  manifest_sha,
    }


def fingerprint_diff(old, new):
    keys = sorted(set(old) | set(new))
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in keys if old.get(k) != new.get(k)]


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_result_lines(path):
    """All lines of a result file ([] if missing). STRICT: a malformed line or a
    duplicate rec_no raises instead of being silently skipped, because skipped
    lines silently change every reported number."""
    if not Path(path).exists():
        return []
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                raise ValueError(f"{path}: line {n} is not valid JSON; the result file is damaged "
                                 f"(rerun with --resume to repair a torn final line)")
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {n} is not a JSON object")
            rn = obj.get("rec_no")
            if rn is not None:
                if rn in seen:
                    raise ValueError(f"{path}: duplicate rec_no {rn} (line {n})")
                seen.add(rn)
            out.append(obj)
    return out


def load_resume_state(path, retry_errors=False):
    """Validate a saved result file for resuming. Returns (kept, dropped, needs_rewrite).
    - kept: records to keep (each has a unique integer rec_no).
    - dropped: lines to redo: never-attempted (ServerUnavailable) lines always,
      other errors only with retry_errors.
    - needs_rewrite: the file must be atomically rewritten from `kept` before
      appending (dropped lines, a torn final line, or a missing final newline).
    A corrupt line that is not the last one, a duplicate rec_no, or a line
    without an integer rec_no raises ValueError: resume refuses."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    raw_lines = [l for l in text.split("\n") if l.strip()]
    needs_rewrite = bool(text) and not text.endswith("\n")
    kept, dropped, seen = [], 0, set()
    for n, raw in enumerate(raw_lines):
        is_last = (n == len(raw_lines) - 1)
        try:
            obj = json.loads(raw)
        except ValueError:
            if is_last:
                needs_rewrite = True        # torn final line from a killed run
                continue
            raise ValueError(f"{path}: corrupt line {n + 1} (not the last line); refusing to resume")
        if not isinstance(obj, dict):
            if is_last:
                needs_rewrite = True
                continue
            raise ValueError(f"{path}: line {n + 1} is not a JSON object; refusing to resume")
        rn = obj.get("rec_no")
        if not isinstance(rn, int) or isinstance(rn, bool):
            raise ValueError(f"{path}: line {n + 1} has no integer rec_no; refusing to resume")
        if rn in seen:
            raise ValueError(f"{path}: duplicate rec_no {rn} (line {n + 1}); refusing to resume")
        seen.add(rn)
        if obj.get("error") and (retry_errors or obj.get("error_type") == "ServerUnavailable"):
            dropped += 1
            needs_rewrite = True
            continue
        kept.append(obj)
    return kept, dropped, needs_rewrite


def rewrite_lines(path, lines):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")
    os.replace(tmp, path)

# ── Pre-flight planning ───────────────────────────────────────────────────────

def output_paths(config, ds_name, out_dir):
    stem = f"config{config['num']}_{config['short']}__{ds_name}"
    out_dir = Path(out_dir)
    return (out_dir / f"{stem}.jsonl", out_dir / f"{stem}.meta.json", out_dir / f"{stem}_poison.jsonl")


def plan_dataset(config, ds_name, data_dir, out_dir, opts):
    """Read-only checks for one dataset. Returns (plan, problems)."""
    problems = []
    spec = DATASET_SPECS[ds_name]
    ds_dir = Path(data_dir) / ds_name
    jsonl = ds_dir / "data.jsonl"
    if not jsonl.exists():
        return None, [f"{ds_name}: {jsonl} not found"]

    try:
        records = load_jsonl(jsonl)
    except ValueError as e:
        return None, [f"{ds_name}: data.jsonl is not valid JSONL ({e})"]
    if opts["limit"]:
        records = records[:opts["limit"]]
    if not records:
        return None, [f"{ds_name}: no records"]

    strict = bool(spec.get("strict"))
    warnings = []
    bad = [f"record {i}: {msg}" for i, r in enumerate(records)
           if (msg := validate_record(spec, r, strict))]
    if bad:
        problems.append(f"{ds_name}: {len(bad)} invalid record(s), e.g. " + "; ".join(bad[:3]))

    missing = []
    for i, r in enumerate(records):
        if "image_paths" not in r or not r["image_paths"]:
            continue
        for p in resolve_image_paths(r, ds_dir):
            sz = file_size_bytes(p)
            if not sz:
                missing.append(f"record {i}: {p}")
    if missing:
        problems.append(f"{ds_name}: {len(missing)} image file(s) missing/empty, e.g. " + "; ".join(missing[:3]))

    idxs = [str(r.get("idx")) for r in records]
    if len(set(idxs)) != len(idxs):
        n_dup = len(idxs) - len(set(idxs))
        if strict:
            problems.append(f"{ds_name}: {n_dup} duplicate idx value(s); ids must be unique in new datasets")
        else:
            warnings.append(f"{n_dup} duplicate idx value(s) in data.jsonl (legacy dataset; results are keyed by position)")
    if not strict and spec["format"] == "mcq":
        n_oor = sum(1 for r in records
                    if "answer" in r and r.get("choices") and not mcq_gt_in_range(r))
        if n_oor:
            warnings.append(f"{n_oor} record(s) have a ground-truth letter outside their choices "
                            f"(legacy dataset; kept as-is, they can never score)")

    manifest = ds_dir / "manifest.json"
    manifest_sha = sha256_file(manifest) if manifest.exists() else None
    fp = build_fingerprint(config, ds_name, spec, sha256_file(jsonl), manifest_sha,
                           opts.get("model_hashes"))
    out_path, meta_path, poison_path = output_paths(config, ds_name, out_dir)

    plan = {
        "dataset": ds_name, "spec": spec, "dataset_dir": ds_dir, "records": records,
        "fingerprint": fp, "out_path": out_path, "meta_path": meta_path,
        "poison_path": poison_path, "kept": [], "dropped": 0, "needs_rewrite": False,
        "warnings": warnings, "action": "fresh",
        "model_hashes": opts.get("model_hashes"), "runner": runner_revision(),
    }

    exists = out_path.exists()
    if opts["smoke"]:
        plan["action"] = "smoke"          # disposable; always rewritten
    elif not exists:
        plan["action"] = "fresh"
    elif opts["overwrite"]:
        plan["action"] = "overwrite"
    elif opts["resume"]:
        meta = read_json(meta_path)
        if not meta or "fingerprint" not in meta:
            problems.append(f"{ds_name}: {out_path.name} exists but has no .meta.json "
                            f"(legacy/unversioned file). Resume refused; it is protected. "
                            f"Use --overwrite only if you really want to replace it.")
        else:
            diffs = fingerprint_diff(meta["fingerprint"], fp)
            if diffs:
                problems.append(f"{ds_name}: cannot resume, protocol/data changed: " + " | ".join(diffs))
            else:
                try:
                    kept, dropped, rewrite = load_resume_state(out_path, opts["retry_errors"])
                except ValueError as e:
                    problems.append(f"{ds_name}: {e}")
                else:
                    misaligned = [l for l in kept
                                  if not (isinstance(l.get("rec_no"), int)
                                          and 0 <= l["rec_no"] < len(records)
                                          and str(records[l["rec_no"]]["idx"]) == str(l.get("idx")))]
                    if misaligned:
                        problems.append(f"{ds_name}: {len(misaligned)} saved line(s) do not match data.jsonl; refusing to resume")
                    else:
                        plan.update(action="resume", kept=kept, dropped=dropped, needs_rewrite=rewrite)
    else:
        problems.append(f"{ds_name}: {out_path.name} already exists. Use --resume to continue "
                        f"or --overwrite to replace (old file is backed up).")
    return plan, problems


def describe_plan(plan):
    a = plan["action"]
    n = len(plan["records"])
    if a == "resume":
        return f"resume: {len(plan['kept'])}/{n} already done, {n - len(plan['kept'])} to run"
    if a == "overwrite":
        return f"OVERWRITE (old files backed up), {n} records"
    if a == "smoke":
        return f"smoke test, {n} records -> {plan['out_path'].parent}"
    return f"fresh, {n} records"

# ── Inference loop ────────────────────────────────────────────────────────────

def backup_existing(plan, ts):
    dest = RESULTS_DIR / "_overwritten" / ts
    dest.mkdir(parents=True, exist_ok=True)
    for p in (plan["out_path"], plan["meta_path"], plan["poison_path"]):
        if Path(p).exists():
            shutil.move(str(p), str(dest / Path(p).name))
    print(f"    Old files moved to {dest}")


def run_dataset(config, state, plan):
    ds_name, spec, records = plan["dataset"], plan["spec"], plan["records"]
    fmt = FORMATS[spec["format"]]
    prompt_fn = fmt["prompts"][spec["prompt"]]
    out_path, meta_path, poison_path = plan["out_path"], plan["meta_path"], plan["poison_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # prepare output files
    if plan["action"] == "overwrite":
        backup_existing(plan, now_pacific().strftime("%Y-%m-%d_%H-%M-%S"))
    resume = plan["action"] == "resume"
    kept = plan["kept"] if resume else []
    if resume and plan["needs_rewrite"]:
        rewrite_lines(out_path, kept)         # atomic; validated lines, one per line, newline-terminated
    meta = read_json(meta_path) if resume else None
    if not meta:
        meta = {"fingerprint": plan["fingerprint"], "created": now_pacific().isoformat(), "sessions": []}
    meta["model_checks"] = plan.get("model_hashes")
    session = {"start": now_pacific().isoformat(), "processed": 0, "elapsed_s": 0.0, "completed": False,
               "runner": plan.get("runner")}
    meta["sessions"].append(session)
    write_json(meta_path, meta)

    done = {l["rec_no"] for l in kept}
    todo = [(i, r) for i, r in enumerate(records) if i not in done]
    print(f"\n  [{ds_name}]  {len(records)} records, {len(todo)} to run  "
          f"({spec['format']}, prompt={spec['prompt']}, parser={spec['parser']}, "
          f"max_tokens={spec['max_tokens']})  →  {out_path.name}")

    # running counters for progress lines (final numbers come from the file)
    run = {"n": len(kept), "err": sum(1 for l in kept if l.get("error")),
           "score": sum((l.get("score") or 0.0) for l in kept if not l.get("error")),
           "retries": sum(1 for l in kept if l.get("retried")),
           "toks": [l["tok_s"] for l in kept if l.get("tok_s", 0) > 0]}
    poison_records = []
    since_restart = 0
    retries_cfg = 0
    ds_start = time.time()
    scored = fmt["scorer"] is not None
    ds_dir = plan["dataset_dir"]

    def mark_unavailable(out_f, start_k):
        for (j, rj) in todo[start_k:]:
            ln = build_line(config, ds_name, spec, rj, j, {}, False, get_mem_mb()[0],
                            error_msg="server unavailable (restart failed)",
                            error_type="ServerUnavailable")
            out_f.write(json.dumps(ln) + "\n")
            run["n"] += 1
            run["err"] += 1
        out_f.flush()

    try:
        with open(out_path, "a" if resume else "w", encoding="utf-8") as out_f:
            for k, (i, record) in enumerate(todo):
                img_paths = resolve_image_paths(record, ds_dir)
                prompt = prompt_fn(record)
                response = call_server(img_paths, prompt, spec["max_tokens"])

                # ── Retry logic: on error, restart server and try once more ──
                retried = False
                if "error" in response:
                    retried = True
                    retries_cfg += 1
                    size_str = ", ".join(f"{s[1]}KB" for s in get_image_sizes(img_paths))
                    print(f"    [!] Record {i+1} failed ({size_str}) {mem_str()}, restarting for retry ...")
                    stop_server(state["proc"])
                    state["proc"] = start_server(config)
                    if not wait_for_server():
                        state["proc"].kill()
                        state["dead"] = True
                        mark_unavailable(out_f, k)
                        break
                    since_restart = 0
                    response = call_server(img_paths, prompt, spec["max_tokens"])

                error_msg, error_type = "", None
                if "error" in response:
                    error_msg = response.get("error", "unknown")
                    error_type = response.get("error_type", "Error")
                    avail_mb, total_mb = get_mem_mb()
                    poison_records.append({
                        "idx": record["idx"], "record_num": i + 1, "error": error_msg,
                        "image_sizes": get_image_sizes(img_paths),
                        "mem_avail_mb": avail_mb, "mem_total_mb": total_mb})

                line = build_line(config, ds_name, spec, record, i, response, retried,
                                  get_mem_mb()[0], error_msg, error_type)
                out_f.write(json.dumps(line) + "\n")
                out_f.flush()
                session["processed"] += 1
                since_restart += 1
                run["n"] += 1
                if error_msg:
                    run["err"] += 1
                else:
                    run["score"] += line.get("score") or 0.0
                    if line["tok_s"] > 0:
                        run["toks"].append(line["tok_s"])
                if retried:
                    run["retries"] += 1

                # ── Scheduled restart ──
                if RESTART_EVERY and since_restart >= RESTART_EVERY and (k + 1) < len(todo):
                    stop_server(state["proc"])
                    state["proc"] = start_server(config)
                    if not wait_for_server():
                        state["proc"].kill()
                        state["dead"] = True
                        mark_unavailable(out_f, k + 1)
                        break
                    since_restart = 0

                # ── Progress ──
                if (k + 1) % 25 == 0 or (k + 1) == len(todo):
                    answered = run["n"] - run["err"]
                    avg_tok = sum(run["toks"]) / len(run["toks"]) if run["toks"] else 0
                    if scored:
                        sc = (f"{fmt['metric']}={run['score']/run['n']*100:.1f}% "
                              f"({(run['score']/answered*100 if answered else 0):.1f}% of answered)")
                    else:
                        sc = "unscored"
                    print(f"    [{i+1:4d}/{len(records)}]  {sc}  avg_tok/s={avg_tok:.1f}  "
                          f"elapsed={time.time()-ds_start:.0f}s  errors={run['err']} "
                          f"retries={run['retries']}  {mem_str()}")
            else:
                session["completed"] = True
            if not todo:
                session["completed"] = True
    finally:
        session["elapsed_s"] = round(time.time() - ds_start, 1)
        session["end"] = now_pacific().isoformat()
        write_json(meta_path, meta)

    if poison_records:
        with open(poison_path, "a" if resume else "w", encoding="utf-8") as pf:
            for pr in poison_records:
                pf.write(json.dumps(pr) + "\n")
        print(f"    Poison records ({len(poison_records)}) saved → {poison_path.name}")

    return summarize_dataset(plan, meta)


def summarize_dataset(plan, meta=None):
    lines = load_result_lines(plan["out_path"])
    m = compute_metrics(lines, plan["spec"]["format"], target=len(plan["records"]))
    meta = meta or read_json(plan["meta_path"]) or {"sessions": []}
    m["elapsed_s"] = round(sum(s.get("elapsed_s", 0) for s in meta["sessions"]), 1)
    m["poison_records"] = m["errors"]
    m["output_file"] = str(plan["out_path"])
    m["prompt"], m["parser"] = plan["spec"]["prompt"], plan["spec"]["parser"]
    m["scorer"] = FORMATS[plan["spec"]["format"]]["scorer_id"]
    m["max_tokens"] = plan["spec"]["max_tokens"]
    m["format"] = plan["spec"]["format"]
    return m


def run_inference(config, state, plans, opts):
    summary = {
        "config_num":    config["num"],
        "config_label":  config["label"],
        "config_short":  config["short"],
        "mmproj":        config["mmproj"],
        "model":         config["model"],
        "note":          config.get("note", ""),
        "timestamp":     now_pacific().strftime("%Y-%m-%d %H:%M:%S"),
        "limit":         opts["limit"],
        "protocol_version": PROTOCOL_VERSION,
        "interrupted":   False,
        "server_failed": False,
        "not_run":       [],
        "complete":      False,
        "runner":        runner_revision(),
        "model_hashes":  opts.get("model_hashes"),
        "datasets":      {},
    }
    for plan in plans:
        if state.get("dead"):
            print(f"\n  [SKIP] {plan['dataset']} — server is unavailable")
            break
        try:
            summary["datasets"][plan["dataset"]] = run_dataset(config, state, plan)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            summary["interrupted"] = True
            summary["datasets"][plan["dataset"]] = summarize_dataset(plan)
            break
    summary["server_failed"] = bool(state.get("dead"))
    summary["not_run"] = [pl["dataset"] for pl in plans if pl["dataset"] not in summary["datasets"]]
    summary["complete"] = bool(
        not summary["interrupted"] and not summary["server_failed"] and not summary["not_run"]
        and summary["datasets"] and all(d["complete"] for d in summary["datasets"].values()))
    return summary

# ── Summary ───────────────────────────────────────────────────────────────────

def _pct(x):
    return "—" if x is None else f"{x * 100:.2f}%"


def save_summary(summary, out_dir):
    ts = now_pacific().strftime("%Y-%m-%d_%H-%M")
    stem = f"config{summary['config_num']}_{summary['config_short']}_{ts}"
    path = Path(out_dir) / f"{stem}.md"
    out_dir = Path(out_dir)

    lines = [
        "# CMPE 249 — Experiment Summary",
        "",
        "## Config",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Config number | {summary['config_num']} |",
        f"| Config label | {summary['config_label']} |",
        f"| Vision encoder | {summary['mmproj']} ({model_size_str(summary['mmproj'])}) |",
        f"| Language decoder | {summary['model']} ({model_size_str(summary['model'])}) |",
        f"| Timestamp | {summary['timestamp']} |",
        f"| Record limit | {summary['limit'] or 'all (full run)'} |",
        f"| Server restart interval | every {RESTART_EVERY} records |",
        f"| Server args | ctx-size {SERVER_CTX}, n-gpu-layers {SERVER_NGL}, temperature {TEMPERATURE} |",
        f"| Protocol version | {summary['protocol_version']} |",
    ]
    if summary.get("note"):
        lines.append(f"| Note | {summary['note']} |")
    if summary.get("interrupted"):
        lines += ["", "> **INTERRUPTED** — partial results; rerun with `--resume` to finish."]
    if summary.get("server_failed"):
        lines += ["", "> **INCOMPLETE — llama-server could not be restarted.** Records never attempted are "
                      "marked pending, not counted as answers or errors. Rerun with `--resume` to finish."]
    if summary.get("not_run"):
        lines += ["", f"> **Not run:** {', '.join(summary['not_run'])}"]
    rev, mh = summary.get("runner") or {}, summary.get("model_hashes") or {}
    lines += ["", "## Provenance", "", "| Item | Value |", "|------|-------|",
              f"| Runner | protocol {rev.get('protocol_version')}, script sha256 {rev.get('script_sha256')}, "
              f"git {rev.get('git_commit') or 'n/a'} |"]
    for field, fname in (("mmproj", summary["mmproj"]), ("model", summary["model"])):
        digest = mh.get(f"{field}_sha256")
        fm = mh.get(f"{field}_frozen_match")
        mark = "matches frozen manifest" if fm else ("**DIFFERS from frozen manifest**" if fm is False else "not in manifest")
        lines.append(f"| {fname} | sha256 {digest} ({mark}) |")
    lines += [
        "",
        "## Results by Dataset",
        "",
        "Each dataset is reported with its own metric. There is deliberately **no pooled TOTAL**: "
        "datasets differ in size and metric (MCQ accuracy vs TextVQA accuracy).",
        "",
        "| Dataset | Metric | Done/Target | Answered | Errors | Retries | Unparsed | Score (raw) | Score (answered) | Avg tok/s | Time |",
        "|---------|--------|-------------|----------|--------|---------|----------|-------------|------------------|-----------|------|",
    ]
    answered_scores = []
    for name, d in summary["datasets"].items():
        flag = "" if d["complete"] else f" ({d['status']}, {d['pending']} pending)"
        lines.append(
            f"| {name} | {d['metric']} | {d['total']}/{d['target']}{flag} | {d['answered']} | "
            f"{d['errors']} | {d['retries']} | {d['unparsed']} | {_pct(d['score_raw'])} | "
            f"{_pct(d['score_answered'])} | {d['avg_tok_s']} | {d['elapsed_s']}s |")
        if d["score_answered"] is not None and d["complete"]:
            answered_scores.append(d["score_answered"])

    lines += [
        "",
        "**Done/Target** counts records actually attempted; **pending** = never attempted.  ",
        "**Score (raw)** = score sum / records attempted (request errors count as 0)  ",
        "**Score (answered)** = score sum / answered (errors excluded) — use this for cross-config comparison  ",
        "**Unparsed** = answered records whose reply produced no usable prediction",
        "",
    ]
    if len(answered_scores) > 1:
        lines += [
            f"*Secondary only:* unweighted macro-average of per-dataset answered scores = "
            f"{sum(answered_scores)/len(answered_scores)*100:.2f}% over {len(answered_scores)} complete datasets. "
            f"It mixes MCQ accuracy and TextVQA accuracy; do not use it as a headline number.",
            "",
        ]

    diag = []
    for name, d in summary["datasets"].items():
        if d.get("parser_disagreements") is not None:
            diag.append(f"- {name}: scored with parser_{d['parser']}; parser_v1 and parser_v2 disagree on "
                        f"{d['parser_disagreements']} of {d['answered']} answered records")
    if diag:
        lines += ["## Scoring diagnostics", ""] + diag + [""]

    lines += ["## Protocol", "",
              "| Dataset | Format | Prompt | Parser | Scorer | max_tokens |",
              "|---------|--------|--------|--------|--------|------------|"]
    for name, d in summary["datasets"].items():
        lines.append(f"| {name} | {d['format']} | {d['prompt']} | {d['parser']} | {d['scorer']} | {d['max_tokens']} |")

    lines += ["", "## Output Files", ""]
    for name, d in summary["datasets"].items():
        lines.append(f"- `{Path(d['output_file']).name}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / f"{stem}.summary.json", summary)
    print(f"\n  Summary saved → {path}")
    return path

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CMPE 249 experiment runner (see module docstring for details)")
    p.add_argument("--data-dir",    default=str(PROJECT_ROOT / "benchmarks"))
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--limit",       default=None, type=int, help="smoke test: first N records per dataset")
    p.add_argument("--port",        default=8080, type=int)
    p.add_argument("--config-num",  default=None, type=int, help="run this config without the menu")
    p.add_argument("--datasets",    default=None, help="comma-separated dataset names")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume",    action="store_true", help="continue an interrupted run")
    g.add_argument("--overwrite", action="store_true", help="replace existing outputs (backed up)")
    p.add_argument("--retry-errors", action="store_true", help="with --resume: retry request-error records")
    p.add_argument("--yes", "-y",    action="store_true", help="skip the confirmation prompt")
    p.add_argument("--dry-run",      action="store_true", help="pre-flight checks only; do not start the server")
    args = p.parse_args(argv)
    if args.retry_errors and not args.resume:
        p.error("--retry-errors only makes sense with --resume")
    return args


def configure(args):
    global DATA_DIR, RESULTS_DIR, OUT_DIR, PORT, BASE_URL
    DATA_DIR = Path(args.data_dir)
    RESULTS_DIR = Path(args.results_dir)
    PORT = args.port
    BASE_URL = f"http://localhost:{PORT}"
    OUT_DIR = RESULTS_DIR / "smoke" if args.limit else RESULTS_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def print_menu():
    print("\n" + "=" * 68)
    print("  CMPE 249 — Quantization Sensitivity Experiment Runner")
    print("  Model : Cosmos-Reason2-2B  |  Hardware : Jetson Orin Nano 8GB")
    print("=" * 68 + "\n")
    for c in CONFIGS:
        mm, md = model_size_str(c["mmproj"]), model_size_str(c["model"])
        mm_ok = "✗ MISSING" if mm == "MISSING" else "✓"
        md_ok = "✗ MISSING" if md == "MISSING" else "✓"
        tag = "  ← unofficial" if not c["official"] else ""
        print(f"  [{c['num']}] {c['label']}{tag}")
        print(f"       Vision encoder  : {c['mmproj']:<48} {mm if mm != 'MISSING' else '':>6}  {mm_ok}")
        print(f"       Language decoder: {c['model']:<48} {md if md != 'MISSING' else '':>6}  {md_ok}")
        print()
    print("  [0] Exit\n")


def choose_config(args):
    if args.config_num is not None:
        c = get_config(args.config_num)
        if c is None:
            print(f"  ERROR: no config number {args.config_num}. "
                  f"Valid: {', '.join(str(x['num']) for x in CONFIGS)}")
            sys.exit(2)
        return c
    print_menu()
    while True:
        try:
            choice = input("  Select config number (0 to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            sys.exit(0)
        if choice == "0":
            print("  Exiting.")
            sys.exit(0)
        if choice.isdigit() and get_config(int(choice)):
            return get_config(int(choice))
        print(f"  Enter one of {', '.join(str(x['num']) for x in CONFIGS)}, or 0 to exit.")


def main(argv=None):
    args = parse_args(argv)
    configure(args)
    validate_registry()

    # ── config + datasets ──
    config = choose_config(args)
    try:
        if args.datasets:
            names = parse_dataset_list(args.datasets)
        else:
            names = [n for n in DATASET_SPECS if (DATA_DIR / n / "data.jsonl").exists()]
            skipped = [n for n in DATASET_SPECS if n not in names]
            if skipped:
                print(f"  Note: no data found for {', '.join(skipped)} — not included.")
    except ValueError as e:
        print(f"  ERROR: {e}")
        sys.exit(2)
    if not names:
        print(f"  ERROR: no datasets to run (looked in {DATA_DIR}).")
        sys.exit(2)

    # ── model files ──
    missing = [str(MODEL_DIR / f) for f in (config["mmproj"], config["model"])
               if not (MODEL_DIR / f).exists()]
    if missing and not args.dry_run:
        print("\n  ERROR: Missing model files:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)

    model_hashes = {"mmproj_sha256": None, "model_sha256": None}
    if not missing:
        print("\n  Checking model checksums (cached after the first run) ...")
        model_hashes = compute_model_hashes(config, RESULTS_DIR / ".model_hash_cache.json")
        for field, fname in (("mmproj", config["mmproj"]), ("model", config["model"])):
            fm = model_hashes[f"{field}_frozen_match"]
            tag = ("matches frozen manifest" if fm else
                   "!! DIFFERS from the frozen Phase 0/1 manifest !!" if fm is False else "not in manifest")
            print(f"    {fname}: {model_hashes[field + '_sha256'][:16]}…  {tag}")

    print(f"\n  Selected: [{config['num']}] {config['label']}")
    if config.get("note"):
        print(f"  NOTE: {config['note']}")
    if args.limit:
        print(f"  Smoke-test mode: {args.limit} records per dataset → {OUT_DIR}")

    # ── pre-flight (read-only) ──
    opts = {"limit": args.limit, "smoke": bool(args.limit), "resume": args.resume,
            "overwrite": args.overwrite, "retry_errors": args.retry_errors,
            "model_hashes": model_hashes}
    plans, problems = [], []
    for n in names:
        plan, probs = plan_dataset(config, n, DATA_DIR, OUT_DIR, opts)
        problems += probs
        if plan:
            plans.append(plan)
    print("\n  Pre-flight plan:")
    for pl in plans:
        print(f"    {pl['dataset']:<14} {describe_plan(pl)}")
        for w in pl["warnings"]:
            print(f"        warning: {w}")
    if problems:
        print("\n  PRE-FLIGHT FAILED — nothing was started or modified:")
        for pr in problems:
            print(f"    ✗ {pr}")
        sys.exit(1)
    if args.dry_run:
        print("\n  Dry run OK.")
        return 0

    if not args.yes:
        try:
            confirm = input("\n  Start experiment? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
            sys.exit(0)
        if confirm != "y":
            print("  Cancelled.")
            sys.exit(0)

    # ── run ──
    state = {"proc": start_server(config), "dead": False}
    if not wait_for_server():
        state["proc"].kill()
        sys.exit(1)

    exp_start = time.time()
    try:
        summary = run_inference(config, state, plans, opts)
    finally:
        stop_server(state["proc"])
    total_elapsed = time.time() - exp_start

    print(f"\n{'=' * 68}")
    verdict = ("COMPLETE" if summary["complete"] else "INTERRUPTED" if summary["interrupted"]
               else "INCOMPLETE (server failure)" if summary["server_failed"] else "INCOMPLETE")
    print(f"  EXPERIMENT {verdict} — Config [{config['num']}] {config['short']}")
    print(f"  Total time: {total_elapsed/60:.1f} min")
    print(f"{'=' * 68}")
    for name, d in summary["datasets"].items():
        print(f"  {name:<16} {d['metric']}={_pct(d['score_raw'])} (answered: {_pct(d['score_answered'])})  "
              f"avg_tok/s={d['avg_tok_s']}  errors={d['errors']}  done={d['total']}/{d['target']}"
              f"{'' if d['complete'] else '  <-- ' + d['status'].upper()}")
    for name in summary["not_run"]:
        print(f"  {name:<16} NOT RUN")
    print()

    summary["total_elapsed_s"] = round(total_elapsed, 1)
    save_summary(summary, OUT_DIR)
    cleanup_ram()      # so the Jetson is usable after the run
    if summary["complete"]:
        return 0
    return 130 if summary["interrupted"] else 3


if __name__ == "__main__":
    sys.exit(main())
