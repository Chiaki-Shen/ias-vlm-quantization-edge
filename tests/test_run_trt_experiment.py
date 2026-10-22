"""
Model-free tests for tensorrt/run_trt_experiment.py (no GPU, no TensorRT).

    python3 -m unittest tests/test_run_trt_experiment.py -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tensorrt"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiment as rx  # noqa: E402
import run_trt_experiment as trt  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class FakeBackend:
    """Returns scripted replies in order, recording what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, image_paths, prompt, max_tokens):
        self.calls.append({"image_paths": image_paths, "prompt": prompt, "max_tokens": max_tokens})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return trt.as_chat_response(reply, "stop"), 123.0, 1


def make_dataset(root: Path, name: str, records: list[dict]) -> None:
    ds = root / name
    (ds / "images").mkdir(parents=True)
    for r in records:
        for p in r["image_paths"]:
            (ds / p).write_bytes(PNG)
    with open(ds / "data.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


MCQ = [
    {"idx": 0, "image_paths": ["images/0.png"], "question": "Which is closer?", "choices": ["left", "right"], "answer": "A"},
    {"idx": 1, "image_paths": ["images/1.png"], "question": "Which is closer?", "choices": ["left", "right"], "answer": "B"},
    {"idx": 2, "image_paths": ["images/2.png"], "question": "Which is closer?", "choices": ["left", "right"], "answer": "A"},
]


class TestSharedProtocol(unittest.TestCase):
    def test_sampling_is_greedy(self):
        s = trt.sampling_settings(16)
        self.assertEqual((s["temperature"], s["top_k"], s["max_tokens"]), (0.0, 1, 16))
        self.assertFalse(s["enable_thinking"])

    def test_messages_put_images_before_text_like_llama_cpp(self):
        msgs = trt.build_messages(["/a.png", "/b.png"], "Q?")
        content = msgs[0]["content"]
        self.assertEqual([c["type"] for c in content], ["image", "image", "text"])
        self.assertEqual(content[-1]["text"], "Q?")

    def test_response_shape_feeds_build_line(self):
        spec = rx.DATASET_SPECS["blink_depth"]
        line = rx.build_line({"short": "X"}, "blink_depth", spec, MCQ[0], 0,
                             trt.as_chat_response("A", "stop"), False, None)
        self.assertEqual(line["prediction_raw"], "A")
        self.assertIs(line["correct"], True)
        self.assertEqual(line["finish_reason"], "stop")


class TestRunDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_dataset(self.root / "benchmarks", "blink_depth", MCQ)
        self.config = trt.CONFIGS["T8"]

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, backend, resume=False, limit=None):
        records, ds_dir, problems = trt.load_records("blink_depth", self.root / "benchmarks", limit)
        self.assertEqual(problems, [])
        out_path, _ = trt.output_paths(self.config, "blink_depth", self.root)
        return trt.run_dataset(backend, self.config, "blink_depth", records, ds_dir, out_path, resume), out_path

    def test_uses_llama_cpp_prompt_and_scoring(self):
        backend = FakeBackend(["A", "A", "A"])
        lines, out_path = self._run(backend)

        self.assertEqual(backend.calls[0]["prompt"], rx.mcq_prompt_legacy(MCQ[0]))
        self.assertEqual(backend.calls[0]["max_tokens"], 16)
        self.assertEqual([ln["correct"] for ln in lines], [True, False, True])
        self.assertTrue(all(ln["runtime"] == "tensorrt-edgellm" and ln["latency_ms"] == 123.0 for ln in lines))
        self.assertEqual(out_path.name, "configT8_TRT_F16v_INT8l__blink_depth.jsonl")
        m = rx.compute_metrics(lines, "mcq", target=3)
        self.assertEqual((m["correct"], m["errors"], m["complete"]), (2, 0, True))

    def test_backend_error_is_logged_not_fatal(self):
        lines, _ = self._run(FakeBackend(["A", RuntimeError("exceeds maxHW"), "A"]))
        self.assertEqual(lines[1]["error_type"], "RuntimeError")
        self.assertIsNone(lines[1]["latency_ms"])
        self.assertEqual(rx.compute_metrics(lines, "mcq", target=3)["errors"], 1)

    def test_resume_skips_done_records(self):
        self._run(FakeBackend(["A"]), limit=1)
        backend = FakeBackend(["B", "A"])
        lines, _ = self._run(backend, resume=True)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual([ln["idx"] for ln in lines], [0, 1, 2])


class TestDryRun(unittest.TestCase):
    def test_dry_run_validates_without_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root / "benchmarks", "blink_depth", MCQ)
            rc = trt.main(["--config", "T4", "--datasets", "blink_depth", "--data-dir", str(root / "benchmarks"),
                           "--out-dir", str(root / "out"), "--workspace", str(root / "ws"), "--dry-run"])
            self.assertEqual(rc, 0)

    def test_refuses_to_overwrite_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root / "benchmarks", "blink_depth", MCQ)
            out_path, _ = trt.output_paths(trt.CONFIGS["T4"], "blink_depth", root / "out")
            out_path.parent.mkdir(parents=True)
            out_path.write_text("")
            rc = trt.main(["--config", "T4", "--datasets", "blink_depth", "--data-dir", str(root / "benchmarks"),
                           "--out-dir", str(root / "out"), "--workspace", str(root / "ws"), "--dry-run"])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
