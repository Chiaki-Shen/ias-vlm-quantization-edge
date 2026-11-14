import sys, os
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.expanduser("~/TensorRT-Edge-LLM"))
from experimental.server import LLM, SamplingParams
d = "/tmp/vary"; os.makedirs(d, exist_ok=True)
Image.new("RGB", (448, 448), (220, 20, 20)).save(f"{d}/red.png")
Image.new("RGB", (448, 448), (20, 20, 220)).save(f"{d}/blue.png")
im = Image.new("RGB", (448, 448), "white"); ImageDraw.Draw(im).rectangle([100, 100, 348, 348], fill="black"); im.save(f"{d}/square.png")
cfg = sys.argv[1]; root = os.path.expanduser(f"~/tensorrt-edgellm-workspace/{cfg}/engines")
llm = LLM(engine_dir=f"{root}/llm", visual_engine_dir=f"{root}/visual")
p = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, max_tokens=40, enable_thinking=False)
for img in [f"{d}/red.png", f"{d}/blue.png", f"{d}/square.png", "/Developer/models/bus.jpg"]:
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "What is the main color in this image? Describe it briefly."}]}]
    print(f"[{cfg}] {os.path.basename(img):10s} -> {llm.generate([m], p)[0].text!r}", flush=True)
m = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
print(f"[{cfg}] text-only  -> {llm.generate([m], p)[0].text!r}", flush=True)
