# Quick functional check of one engine pair with the runner's message format and greedy settings.
import sys, time, os
sys.path.insert(0, os.path.expanduser("~/TensorRT-Edge-LLM"))
from experimental.server import LLM, SamplingParams
cfg = sys.argv[1]; root = os.path.expanduser(f"~/tensorrt-edgellm-workspace/{cfg}/engines")
t = time.perf_counter(); llm = LLM(engine_dir=f"{root}/llm", visual_engine_dir=f"{root}/visual")
print(f"[{cfg}] load {time.perf_counter()-t:.1f}s", flush=True)
img = "/Developer/models/bus.jpg"
qs = [("How many people are standing in this image? Answer with a number.", 16),
      ("Which is closer to the camera, the bus or the people? Options: (A) the bus (B) the people\nAnswer with the option's letter from the given choices directly.", 16),
      ("What is written on the front of the bus?", 32)]
for q, n in qs:
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    p = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, max_tokens=n, enable_thinking=False)
    t = time.perf_counter(); out = llm.generate([msgs], p)[0]; ms = (time.perf_counter()-t)*1000
    print(f"[{cfg}] {ms:7.0f} ms  {len(out.token_ids):2d} tok  finish={out.finish_reason}  -> {out.text!r}", flush=True)
