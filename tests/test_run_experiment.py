"""
Model-free unit tests for scripts/run_experiment.py (no GPU, no llama-server).

Run on the Jetson or the Mac:
    cd /Developer/ias-vlm-quantization-edge
    python3 -m unittest tests/test_run_experiment.py -v

TextVQA expected values below were produced by MMF's TextVQAAccuracyEvaluator
(see tests/check_textvqa_against_reference.py for the 50,000-case differential test).
"""
import json
import sys
import tempfile
from unittest import mock
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_experiment as rx

OTHERS = ["dog", "cat", "bird", "fish", "cow", "pig", "hen", "owl", "fox"]
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


class TestMcqParsers(unittest.TestCase):
    REC4 = {"choices": ["w", "x", "y", "z"]}

    def test_v2_accepts_common_formats(self):
        for text in ["B", "(B)", "B.", "B)", "B:", "Answer: B", "answer is B", "The answer is (B)",
                     "**B**", "B\n\nBecause ...", "B. y is wrong", " b "]:
            self.assertEqual(rx.parse_letter_v2(text, self.REC4), "B", msg=text)

    def test_v2_rejects_words_starting_with_a_letter(self):
        for text in ["Apple", "A cat sat", "Because", "Certainly", "The image shows", ""]:
            self.assertEqual(rx.parse_letter_v2(text, self.REC4), "", msg=text)

    def test_v2_rejects_letters_outside_choices(self):
        self.assertEqual(rx.parse_letter_v2("E", self.REC4), "")
        self.assertEqual(rx.parse_letter_v2("Answer: F", self.REC4), "")

    def test_v2_strips_closed_think_block(self):
        self.assertEqual(rx.parse_letter_v2("<think>maybe A</think>C", self.REC4), "C")

    def test_v1_is_legacy_first_character(self):
        self.assertEqual(rx.parse_letter_v1("Apple"), "A")      # known weakness, kept on purpose
        self.assertEqual(rx.parse_letter_v1("  b"), "B")
        self.assertEqual(rx.parse_letter_v1(""), "")
        self.assertEqual(rx.parse_letter_v1(None), "")

    def test_gt_letter_mapping(self):
        self.assertEqual(rx.normalize_gt_letter(0), "A")
        self.assertEqual(rx.normalize_gt_letter("2"), "C")
        self.assertEqual(rx.normalize_gt_letter("(B)"), "B")
        self.assertEqual(rx.normalize_gt_letter("b"), "B")
        self.assertEqual(rx.normalize_gt_letter(" (D) "), "D")


class TestPrompts(unittest.TestCase):
    def test_legacy_prompt_is_byte_identical_to_configs_1_to_8(self):
        rec = {"question": "Which is closer?", "choices": ["the car", "the tree"]}
        expected = ("Which is closer?\n\nA. the car\nB. the tree\n\n"
                    "Answer with only the letter of the correct choice. Do not explain.")
        self.assertEqual(rx.mcq_prompt_legacy(rec), expected)

    def test_context_prompt_includes_hint_only_when_present(self):
        base = {"question": "Q?", "choices": ["a", "b"]}
        with_ctx = rx.mcq_prompt_context({**base, "context": "Some hint."})
        self.assertTrue(with_ctx.startswith("Context: Some hint.\n\nQ?"))
        self.assertEqual(rx.mcq_prompt_context({**base, "context": "  "}),
                         rx.mcq_prompt_context(base))
        self.assertNotIn("Context:", rx.mcq_prompt_context(base))

    def test_context_prompt_never_leaks_lecture_or_solution(self):
        rec = {"question": "Q?", "choices": ["a", "b"], "context": "hint",
               "lecture": "LECTURE TEXT", "solution": "SOLUTION TEXT"}
        p = rx.mcq_prompt_context(rec)
        self.assertNotIn("LECTURE", p)
        self.assertNotIn("SOLUTION", p)


class TestTextVqa(unittest.TestCase):
    def test_soft_credit_ladder(self):
        for k, want in [(1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (5, 1.0)]:
            answers = ["red"] * k + (OTHERS * 2)[: 10 - k]
            self.assertAlmostEqual(rx.textvqa_score("red", answers), want, places=9, msg=f"k={k}")

    def test_not_simple_min_matches_over_three(self):
        # 3 matches -> 0.9 (not 1.0 = min(3/3,1)); that is the leave-one-out effect
        self.assertAlmostEqual(rx.textvqa_score("red", ["red"] * 3 + OTHERS[:7]), 0.9)

    def test_always_normalizes_even_when_humans_are_unanimous(self):
        # AUDIT REGRESSION: TextVQA's evaluator normalizes in every case. The general VQA
        # evaluator does not, and would score all of these 0.0.
        self.assertEqual(rx.textvqa_score("Stop", ["stop"] * 10), 1.0)
        self.assertEqual(rx.textvqa_score("stop.", ["stop"] * 10), 1.0)
        self.assertEqual(rx.textvqa_score("what?", ["what"] * 10), 1.0)

    def test_tokenization_step_commas_and_possessive(self):
        # AUDIT REGRESSION: needs word_tokenize (drop ',' and '?', split "'s"); values from MMF
        self.assertEqual(rx.textvqa_score("foo,bar", ["foobar"] * 10), 1.0)
        self.assertEqual(rx.textvqa_score("dog's", ["dog 's"] * 10), 1.0)

    def test_normalization_cases(self):
        self.assertEqual(rx.textvqa_score("none", ["0"] * 10), 1.0)               # number word
        self.assertEqual(rx.textvqa_score("apple", ["an apple"] * 10), 1.0)       # article
        self.assertEqual(rx.textvqa_score("two", ["2"] * 6 + ["3"] * 4), 1.0)
        self.assertEqual(rx.textvqa_score("dont", ["don't"] * 6 + ["do not"] * 4), 1.0)  # contraction
        self.assertEqual(rx.textvqa_score("3.5", ["3.5"] * 10), 1.0)              # decimal point kept
        self.assertEqual(rx.textvqa_score("1,000", ["1000"] * 10), 1.0)           # thousands comma
        self.assertEqual(rx.textvqa_score("zebra", ["dog"] * 5 + ["cat"] * 5), 0.0)

    def test_requires_exactly_ten_answers(self):
        with self.assertRaises(ValueError):
            rx.textvqa_score("x", ["x"] * 9)
        with self.assertRaises(ValueError):
            rx.textvqa_score("x", [])

    def test_score_textvqa_has_single_main_score(self):
        out = rx.score_textvqa("Stop", {"answers": ["stop"] * 10})
        self.assertEqual(out["score"], 1.0)
        self.assertNotIn("score_always_norm", out)

    def test_parse_short_keeps_the_whole_answer(self):
        self.assertEqual(rx.parse_short("coca cola"), "coca cola")
        self.assertEqual(rx.parse_short("  New   York City \nextra line"), "New York City")
        self.assertEqual(rx.parse_short("Answer: 10 o'clock"), "10 o'clock")
        self.assertEqual(rx.parse_short("<think>hmm</think>stop sign"), "stop sign")
        self.assertEqual(rx.parse_short(""), "")
        self.assertEqual(rx.parse_short(None), "")


class TestImages(unittest.TestCase):
    def test_mime_from_magic_bytes(self):
        self.assertEqual(rx.detect_mime(JPG, "x.png"), "image/jpeg")   # content beats suffix
        self.assertEqual(rx.detect_mime(PNG, "x.jpg"), "image/png")

    def test_mime_falls_back_to_suffix_then_jpeg(self):
        self.assertEqual(rx.detect_mime(b"????", "pic.PNG"), "image/png")
        self.assertEqual(rx.detect_mime(b"????", "pic.unknown"), "image/jpeg")

    def test_data_uri_uses_detected_type(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"        # wrong suffix on purpose
            p.write_bytes(PNG)
            self.assertTrue(rx.image_data_uri(p).startswith("data:image/png;base64,"))

    def test_relative_paths_resolve_against_dataset_dir_absolute_kept(self):
        rec = {"image_paths": ["images/a.png", "/abs/b.jpg"]}
        out = rx.resolve_image_paths(rec, "/data/scienceqa")
        self.assertEqual(out[0], Path("/data/scienceqa/images/a.png"))
        self.assertEqual(out[1], Path("/abs/b.jpg"))


class TestRegistry(unittest.TestCase):
    def test_shipped_registry_is_valid(self):
        rx.validate_registry()

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            rx.validate_registry({"d": {"format": "typo", "prompt": "x", "parser": "y", "max_tokens": 8}})

    def test_unknown_prompt_or_parser_raises(self):
        bad = {"d": {"format": "mcq", "prompt": "nope", "parser": "v1", "max_tokens": 8}}
        with self.assertRaises(ValueError):
            rx.validate_registry(bad)
        bad = {"d": {"format": "mcq", "prompt": "legacy", "parser": "nope", "max_tokens": 8}}
        with self.assertRaises(ValueError):
            rx.validate_registry(bad)

    def test_raw_only_is_opt_in_and_unscored(self):
        spec = {"format": "raw_only", "prompt": "plain", "parser": "raw", "max_tokens": 64}
        rx.validate_registry({"explore": spec})
        self.assertIsNone(rx.FORMATS["raw_only"]["scorer"])
        self.assertNotIn("raw_only", [s["format"] for s in rx.DATASET_SPECS.values()])

    def test_dataset_list_parsing(self):
        self.assertEqual(rx.parse_dataset_list("textvqa, scienceqa"), ["textvqa", "scienceqa"])
        with self.assertRaises(ValueError):
            rx.parse_dataset_list("textvqa,typo")
        with self.assertRaises(ValueError):
            rx.parse_dataset_list("textvqa,textvqa")

    def test_legacy_datasets_keep_legacy_protocol(self):
        for name in ("blink_depth", "blink_spatial", "cv_bench"):
            s = rx.DATASET_SPECS[name]
            self.assertEqual((s["format"], s["prompt"], s["parser"], s["max_tokens"]),
                             ("mcq", "legacy", "v1", 16), msg=name)

    def test_record_validation(self):
        mcq = rx.DATASET_SPECS["scienceqa"]
        good = {"idx": 1, "question": "q", "choices": ["a"], "answer": 0, "image_paths": ["i.png"]}
        self.assertIsNone(rx.validate_record(mcq, good))
        self.assertIn("answer", rx.validate_record(mcq, {k: v for k, v in good.items() if k != "answer"}))
        self.assertIn("format", rx.validate_record(mcq, {**good, "format": "open_vqa"}))
        self.assertIn("image_paths", rx.validate_record(mcq, {**good, "image_paths": []}))

    def test_textvqa_requires_exactly_ten_string_answers(self):
        # AUDIT REGRESSION: a record with a single human answer used to be accepted
        tv = rx.DATASET_SPECS["textvqa"]
        base = {"idx": 1, "question": "q", "image_paths": ["i.jpg"]}
        self.assertIsNone(rx.validate_record(tv, {**base, "answers": ["a"] * 10}))
        for bad in (["a"], ["a"] * 9, ["a"] * 11, [1] * 10, "stop", []):
            self.assertIn("exactly 10", rx.validate_record(tv, {**base, "answers": bad}), msg=repr(bad))

    def test_strict_mcq_rejects_out_of_range_ground_truth(self):
        sqa = rx.DATASET_SPECS["scienceqa"]
        rec = {"idx": 1, "question": "q", "choices": ["a", "b"], "image_paths": ["i.png"]}
        for ok in (0, 1, "A", "(B)"):
            self.assertIsNone(rx.validate_record(sqa, {**rec, "answer": ok}, strict=True), msg=repr(ok))
        for bad in (2, "C", 9, "AB", "-1"):
            self.assertIn("outside", rx.validate_record(sqa, {**rec, "answer": bad}, strict=True), msg=repr(bad))
        # legacy (non-strict) datasets keep working even with a bad ground truth
        self.assertIsNone(rx.validate_record(rx.DATASET_SPECS["blink_depth"], {**rec, "answer": "C"}, strict=False))

    def test_new_datasets_are_strict_legacy_are_not(self):
        self.assertTrue(rx.DATASET_SPECS["scienceqa"]["strict"] and rx.DATASET_SPECS["textvqa"]["strict"])
        for n in ("blink_depth", "blink_spatial", "cv_bench"):
            self.assertFalse(rx.DATASET_SPECS[n].get("strict"), msg=n)


class TestConfigs(unittest.TestCase):
    def test_numbers_unique_and_stable(self):
        nums = [c["num"] for c in rx.CONFIGS]
        self.assertEqual(nums, list(range(1, 12)))

    def test_new_vision_sweep_configs(self):
        for num, short, mmproj in [(9, "Q6v_Q8l", "mmproj-Cosmos-Reason2-2B-Q6_K.gguf"),
                                   (10, "Q5v_Q8l", "mmproj-Cosmos-Reason2-2B-Q5_K_M.gguf"),
                                   (11, "Q3v_Q8l", "mmproj-Cosmos-Reason2-2B-Q3_K_M.gguf")]:
            c = rx.get_config(num)
            self.assertEqual(c["short"], short)
            self.assertEqual(c["mmproj"], mmproj)
            self.assertEqual(c["model"], "Cosmos-Reason2-2B-Q8_0.gguf")
            self.assertFalse(c["official"])
            self.assertTrue(c["note"])

    def test_existing_config_definitions_unchanged(self):
        self.assertEqual(rx.get_config(2)["short"], "F16v_Q8l")
        self.assertEqual(rx.get_config(5)["short"], "Q8v_Q8l")
        self.assertEqual(rx.get_config(8)["short"], "Q4v_Q8l")
        self.assertEqual(rx.get_config(8)["mmproj"], "mmproj-Cosmos-Reason2-2B-Q4_0.gguf")

    def test_fmt_size(self):
        self.assertEqual(rx.fmt_size(None), "MISSING")
        self.assertEqual(rx.fmt_size(329 * 1024 ** 2), "329 MB")
        self.assertEqual(rx.fmt_size(int(3.8 * 1024 ** 3)), "3.8 GB")


class TestBuildLineAndMetrics(unittest.TestCase):
    def _resp(self, text, tok=20.0):
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "timings": {"predicted_per_second": tok}}

    def test_mcq_line_keeps_raw_text_and_both_parsers(self):
        cfg, spec = rx.get_config(9), rx.DATASET_SPECS["blink_depth"]
        rec = {"idx": "q1", "question": "Q", "choices": ["a", "b"], "answer": "B", "image_paths": ["x.jpg"]}
        line = rx.build_line(cfg, "blink_depth", spec, rec, 0, self._resp("Because B"), False, 1000)
        self.assertEqual(line["prediction_raw"], "Because B")
        self.assertEqual(line["predicted"], "B")            # legacy first character
        self.assertEqual(line["pred_parser_v1"], "B")
        self.assertEqual(line["pred_parser_v2"], "")        # v2 refuses to guess
        self.assertEqual(line["gt_answer"], "B")
        self.assertEqual(line["score"], 1.0)
        self.assertIsNone(line["error"])

    def test_error_line_has_no_score(self):
        cfg, spec = rx.get_config(9), rx.DATASET_SPECS["scienceqa"]
        rec = {"idx": 1, "question": "Q", "choices": ["a", "b"], "answer": 0, "image_paths": ["x.png"]}
        line = rx.build_line(cfg, "scienceqa", spec, rec, 3, {}, True, 900,
                             error_msg="timeout", error_type="TimeoutError")
        self.assertIsNone(line["score"])
        self.assertEqual(line["error_type"], "TimeoutError")
        self.assertEqual(line["rec_no"], 3)

    def test_vqa_line(self):
        cfg, spec = rx.get_config(9), rx.DATASET_SPECS["textvqa"]
        rec = {"idx": 7, "question": "Q", "answers": ["stop"] * 10, "image_paths": ["x.jpg"]}
        line = rx.build_line(cfg, "textvqa", spec, rec, 0, self._resp("Stop"), False, 1000)
        self.assertEqual(line["predicted"], "Stop")
        self.assertEqual(line["prediction_normalized"], "stop")
        self.assertEqual(line["score"], 1.0)           # TextVQA normalizes even unanimous answers
        self.assertNotIn("score_always_norm", line)

    def test_metrics_raw_vs_answered_and_no_pooling(self):
        lines = [{"error": None, "score": 1.0, "correct": True, "predicted": "A", "tok_s": 10.0,
                  "pred_parser_v1": "A", "pred_parser_v2": "A"},
                 {"error": None, "score": 0.0, "correct": False, "predicted": "B", "tok_s": 30.0,
                  "pred_parser_v1": "B", "pred_parser_v2": ""},
                 {"error": "boom", "error_type": "URLError", "score": None, "retried": True, "tok_s": 0.0}]
        m = rx.compute_metrics(lines, "mcq", target=3)
        self.assertEqual((m["total"], m["answered"], m["errors"], m["retries"]), (3, 2, 1, 1))
        self.assertAlmostEqual(m["score_raw"], 1 / 3)
        self.assertAlmostEqual(m["score_answered"], 1 / 2)
        self.assertEqual(m["avg_tok_s"], 20.0)
        self.assertEqual(m["parser_disagreements"], 1)
        self.assertTrue(m["complete"])          # request errors still count as ATTEMPTED

    def test_all_placeholders_is_not_complete(self):
        # AUDIT REGRESSION: five never-attempted records used to report complete=True
        lines = [{"rec_no": i, "error": "server unavailable", "error_type": "ServerUnavailable"} for i in range(5)]
        m = rx.compute_metrics(lines, "mcq", target=5)
        self.assertFalse(m["complete"])
        self.assertEqual((m["total"], m["pending"], m["errors"], m["answered"]), (0, 5, 0, 0))
        self.assertEqual(m["status"], "not attempted")
        self.assertIsNone(m["score_raw"])

    def test_partial_run_with_placeholders(self):
        lines = ([{"rec_no": i, "error": None, "score": 1.0, "predicted": "A", "tok_s": 5.0} for i in range(3)]
                 + [{"rec_no": i, "error": "x", "error_type": "ServerUnavailable"} for i in range(3, 10)])
        m = rx.compute_metrics(lines, "mcq", target=10)
        self.assertEqual((m["total"], m["pending"], m["status"]), (3, 7, "partial"))
        self.assertFalse(m["complete"])
        self.assertEqual(m["score_raw"], 1.0)      # placeholders do not dilute the score

    def test_missing_records_count_as_pending(self):
        m = rx.compute_metrics([{"rec_no": 0, "error": None, "score": 0.0, "predicted": "A"}], "mcq", target=4)
        self.assertEqual((m["pending"], m["complete"]), (3, False))

    def test_unscored_format_has_no_score(self):
        m = rx.compute_metrics([{"error": None, "predicted": "x", "tok_s": 1.0}], "raw_only")
        self.assertIsNone(m["score_raw"])
        self.assertEqual(m["metric"], "unscored")


class TestResume(unittest.TestCase):
    def _write(self, path, objs, trailing=""):
        with open(path, "w", encoding="utf-8") as f:
            for o in objs:
                f.write(json.dumps(o) + "\n")
            f.write(trailing)

    def test_resume_keeps_good_and_request_errors_drops_unattempted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            self._write(p, [{"rec_no": 0, "idx": "a", "error": None},
                            {"rec_no": 1, "idx": "b", "error": "timeout", "error_type": "TimeoutError"},
                            {"rec_no": 2, "idx": "c", "error": "x", "error_type": "ServerUnavailable"}])
            kept, dropped, rewrite = rx.load_resume_state(p)
            self.assertEqual([l["rec_no"] for l in kept], [0, 1])   # request error stays an error
            self.assertEqual(dropped, 1)
            self.assertTrue(rewrite)

    def test_clean_file_needs_no_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            self._write(p, [{"rec_no": 0, "idx": "a", "error": None}])
            self.assertEqual(rx.load_resume_state(p)[1:], (0, False))

    def test_retry_errors_drops_request_errors_too(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            self._write(p, [{"rec_no": 0, "idx": "a", "error": None},
                            {"rec_no": 1, "idx": "b", "error": "t", "error_type": "TimeoutError"}])
            kept, dropped, _ = rx.load_resume_state(p, retry_errors=True)
            self.assertEqual([l["rec_no"] for l in kept], [0])
            self.assertEqual(dropped, 1)

    def test_truncated_last_line_dropped_corrupt_middle_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            self._write(p, [{"rec_no": 0, "idx": "a", "error": None}], trailing='{"rec_no": 1, "id')
            kept, _, rewrite = rx.load_resume_state(p)
            self.assertEqual(len(kept), 1)
            self.assertTrue(rewrite)
            p.write_text('{"rec_no":0}\nNOT JSON\n{"rec_no":2}\n')
            with self.assertRaises(ValueError):
                rx.load_resume_state(p)

    def test_missing_final_newline_is_repaired_before_append(self):
        # AUDIT REGRESSION: a valid last line with no trailing newline used to be accepted as-is,
        # so the next append produced '}{' on one line and the summary silently saw 0 records.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text(json.dumps({"rec_no": 0, "idx": "a", "error": None, "score": 1.0}))   # no newline
            kept, dropped, rewrite = rx.load_resume_state(p)
            self.assertTrue(rewrite)                                   # must be rewritten first
            rx.rewrite_lines(p, kept)                                  # what run_dataset does
            with open(p, "a", encoding="utf-8") as f:                  # then it appends
                f.write(json.dumps({"rec_no": 1, "idx": "b", "error": None, "score": 1.0}) + "\n")
            lines = rx.load_result_lines(p)
            self.assertEqual([l["rec_no"] for l in lines], [0, 1])
            self.assertEqual(len(p.read_text().strip().splitlines()), 2)

    def test_duplicate_rec_no_is_rejected(self):
        # AUDIT REGRESSION: duplicates were accepted and inflated metrics
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            self._write(p, [{"rec_no": 0, "idx": "a", "error": None}, {"rec_no": 0, "idx": "a", "error": None}])
            with self.assertRaisesRegex(ValueError, "duplicate rec_no"):
                rx.load_resume_state(p)
            with self.assertRaisesRegex(ValueError, "duplicate rec_no"):
                rx.load_result_lines(p)

    def test_line_without_integer_rec_no_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            self._write(p, [{"idx": "a", "error": None}])
            with self.assertRaisesRegex(ValueError, "rec_no"):
                rx.load_resume_state(p)

    def test_summary_loader_never_silently_skips_bad_lines(self):
        # AUDIT REGRESSION: load_result_lines used to skip invalid lines without a word
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text('{"rec_no": 0}\n{"rec_no": 1}{"rec_no": 2}\n')
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                rx.load_result_lines(p)
        self.assertEqual(rx.load_result_lines(Path("/nonexistent/x.jsonl")), [])

    def test_rewrite_has_no_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            kept = [{"rec_no": 0, "idx": "a"}, {"rec_no": 1, "idx": "b"}]
            rx.rewrite_lines(p, kept)
            rx.rewrite_lines(p, kept)
            self.assertEqual(len(p.read_text().strip().splitlines()), 2)

    def test_fingerprint_diff_detects_protocol_change(self):
        cfg, spec = rx.get_config(9), rx.DATASET_SPECS["scienceqa"]
        a = rx.build_fingerprint(cfg, "scienceqa", spec, "sha1", None)
        b = rx.build_fingerprint(cfg, "scienceqa", {**spec, "parser": "v1"}, "sha1", None)
        c = rx.build_fingerprint(cfg, "scienceqa", spec, "sha2", None)
        self.assertEqual(rx.fingerprint_diff(a, a), [])
        self.assertTrue(any("parser" in x for x in rx.fingerprint_diff(a, b)))
        self.assertTrue(any("data_sha256" in x for x in rx.fingerprint_diff(a, c)))


class TestProvenance(unittest.TestCase):
    def test_protocol_bumped_after_audit(self):
        self.assertEqual(rx.PROTOCOL_VERSION, "p2.1")
        self.assertEqual(rx.FORMATS["open_vqa"]["scorer_id"], "textvqa_mmf_v1")

    def test_fingerprint_contains_model_checksums_and_blocks_resume_on_change(self):
        cfg, spec = rx.get_config(9), rx.DATASET_SPECS["scienceqa"]
        h1 = {"mmproj_sha256": "a" * 64, "model_sha256": "b" * 64}
        h2 = {"mmproj_sha256": "a" * 64, "model_sha256": "c" * 64}
        f1 = rx.build_fingerprint(cfg, "scienceqa", spec, "d", None, h1)
        f2 = rx.build_fingerprint(cfg, "scienceqa", spec, "d", None, h2)
        self.assertEqual(f1["mmproj_sha256"], "a" * 64)
        self.assertTrue(any("model_sha256" in x for x in rx.fingerprint_diff(f1, f2)))

    def test_runner_revision_is_recorded_but_not_in_fingerprint(self):
        rev = rx.runner_revision()
        self.assertEqual(rev["protocol_version"], "p2.1")
        self.assertTrue(rev["script_sha256"])
        fp = rx.build_fingerprint(rx.get_config(9), "scienceqa", rx.DATASET_SPECS["scienceqa"], "d", None)
        self.assertNotIn("script_sha256", fp)      # editing the script must not block resume

    def test_frozen_table_covers_all_used_model_files(self):
        for c in rx.CONFIGS:
            for fname in (c["mmproj"], c["model"]):
                h = rx.FROZEN_SHA256.get(fname)
                self.assertTrue(h and len(h) == 64 and set(h) <= set("0123456789abcdef"), msg=fname)

    def test_file_hash_is_cached_by_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            f, cache = Path(d) / "m.gguf", Path(d) / "cache.json"
            f.write_bytes(b"hello")
            real = rx.sha256_file
            with mock.patch.object(rx, "sha256_file", side_effect=real) as spy:
                h1 = rx.file_sha256_cached(f, cache)
                h2 = rx.file_sha256_cached(f, cache)
                self.assertEqual(h1, h2)
                self.assertEqual(spy.call_count, 1)                  # second call came from cache
                f.write_bytes(b"hello, changed")                     # size changes -> rehash
                h3 = rx.file_sha256_cached(f, cache)
                self.assertNotEqual(h1, h3)
                self.assertEqual(spy.call_count, 2)


class TestPlanning(unittest.TestCase):
    def _make_dataset(self, root, name="scienceqa", n=3):
        d = Path(root) / "bench" / name
        (d / "images").mkdir(parents=True)
        recs = []
        for i in range(n):
            (d / "images" / f"{i}.png").write_bytes(PNG)
            recs.append({"idx": f"{name}-{i}", "format": "mcq", "question": "q", "choices": ["a", "b"],
                         "answer": 0, "image_paths": [f"images/{i}.png"]})
        with open(d / "data.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        return Path(root) / "bench"

    OPTS = {"limit": None, "smoke": False, "resume": False, "overwrite": False, "retry_errors": False}

    def test_fresh_then_refuse_existing_without_flags(self):
        with tempfile.TemporaryDirectory() as root:
            bench, out = self._make_dataset(root), Path(root) / "res"
            out.mkdir()
            cfg = rx.get_config(5)
            plan, probs = rx.plan_dataset(cfg, "scienceqa", bench, out, self.OPTS)
            self.assertEqual((plan["action"], probs), ("fresh", []))
            plan["out_path"].write_text("{}\n")                       # pretend a result exists
            plan, probs = rx.plan_dataset(cfg, "scienceqa", bench, out, self.OPTS)
            self.assertTrue(any("already exists" in p for p in probs))

    def test_resume_of_legacy_file_without_meta_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            bench, out = self._make_dataset(root), Path(root) / "res"
            out.mkdir()
            cfg = rx.get_config(5)
            (out / "config5_Q8v_Q8l__scienceqa.jsonl").write_text("{}\n")
            plan, probs = rx.plan_dataset(cfg, "scienceqa", bench, out, {**self.OPTS, "resume": True})
            self.assertTrue(any("legacy" in p for p in probs))

    def test_missing_image_and_missing_dataset_reported(self):
        with tempfile.TemporaryDirectory() as root:
            bench, out = self._make_dataset(root), Path(root) / "res"
            out.mkdir()
            (bench / "scienceqa" / "images" / "1.png").unlink()
            plan, probs = rx.plan_dataset(rx.get_config(5), "scienceqa", bench, out, self.OPTS)
            self.assertTrue(any("missing" in p for p in probs))
            plan, probs = rx.plan_dataset(rx.get_config(5), "textvqa", bench, out, self.OPTS)
            self.assertIsNone(plan)
            self.assertTrue(any("not found" in p for p in probs))

    def test_new_dataset_with_duplicate_ids_is_rejected_legacy_only_warns(self):
        with tempfile.TemporaryDirectory() as root:
            bench, out = self._make_dataset(root), Path(root) / "res"
            out.mkdir()
            path = bench / "scienceqa" / "data.jsonl"
            recs = [json.loads(l) for l in path.read_text().splitlines()]
            recs[1]["idx"] = recs[0]["idx"]
            path.write_text("".join(json.dumps(r) + "\n" for r in recs))
            plan, probs = rx.plan_dataset(rx.get_config(5), "scienceqa", bench, out, self.OPTS)
            self.assertTrue(any("duplicate idx" in p for p in probs))
            # same file under a legacy (non-strict) name: warning only
            legacy = self._make_dataset(root + "/x", name="blink_depth")
            path = legacy / "blink_depth" / "data.jsonl"
            recs = [json.loads(l) for l in path.read_text().splitlines()]
            recs[1]["idx"] = recs[0]["idx"]
            path.write_text("".join(json.dumps(r) + "\n" for r in recs))
            plan, probs = rx.plan_dataset(rx.get_config(5), "blink_depth", legacy, out, self.OPTS)
            self.assertEqual(probs, [])
            self.assertTrue(any("duplicate idx" in w for w in plan["warnings"]))

    def test_new_dataset_with_out_of_range_answer_fails_preflight(self):
        with tempfile.TemporaryDirectory() as root:
            bench, out = self._make_dataset(root), Path(root) / "res"
            out.mkdir()
            path = bench / "scienceqa" / "data.jsonl"
            recs = [json.loads(l) for l in path.read_text().splitlines()]
            recs[0]["answer"] = 5
            path.write_text("".join(json.dumps(r) + "\n" for r in recs))
            plan, probs = rx.plan_dataset(rx.get_config(5), "scienceqa", bench, out, self.OPTS)
            self.assertTrue(any("outside" in p for p in probs))

    def test_resume_blocked_when_model_checksum_changed(self):
        with tempfile.TemporaryDirectory() as root:
            bench, out = self._make_dataset(root), Path(root) / "res"
            out.mkdir()
            cfg = rx.get_config(5)
            h1 = {"mmproj_sha256": "a" * 64, "model_sha256": "b" * 64}
            plan, _ = rx.plan_dataset(cfg, "scienceqa", bench, out, {**self.OPTS, "model_hashes": h1})
            plan["out_path"].write_text(json.dumps({"rec_no": 0, "idx": "scienceqa-0", "error": None}) + "\n")
            rx.write_json(plan["meta_path"], {"fingerprint": plan["fingerprint"], "sessions": []})
            ok_opts = {**self.OPTS, "resume": True, "model_hashes": h1}
            plan, probs = rx.plan_dataset(cfg, "scienceqa", bench, out, ok_opts)
            self.assertEqual((probs, plan["action"]), ([], "resume"))
            h2 = {**h1, "model_sha256": "c" * 64}
            plan, probs = rx.plan_dataset(cfg, "scienceqa", bench, out, {**ok_opts, "model_hashes": h2})
            self.assertTrue(any("model_sha256" in p for p in probs))

    def test_smoke_mode_always_plans_overwrite_of_disposable_file(self):
        with tempfile.TemporaryDirectory() as root:
            bench, out = self._make_dataset(root), Path(root) / "res"
            out.mkdir()
            (out / "config5_Q8v_Q8l__scienceqa.jsonl").write_text("{}\n")
            plan, probs = rx.plan_dataset(rx.get_config(5), "scienceqa", bench, out,
                                          {**self.OPTS, "smoke": True, "limit": 2})
            self.assertEqual((plan["action"], probs, len(plan["records"])), ("smoke", [], 2))


if __name__ == "__main__":
    unittest.main()
