import sys, json, time, csv, glob, statistics
from pathlib import Path
sys.path.insert(0, 'scripts')
import run_experiment as rx
rx.BASE_URL = "http://localhost:8000"

ds_dir = Path('benchmarks/textvqa')
recs = [json.loads(l) for l in open(ds_dir / 'data.jsonl') if l.strip()]
timed = json.load(open('results_latency/records_frozen.json'))['datasets']['textvqa']['timed']

# llama.cpp prompt tokens by dataset position (overnight run first)
base = {}
files = sorted(glob.glob('results_latency/*/requests.jsonl'),
               key=lambda f: 0 if '/overnight/' in f else 1)
for f in files:
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('dataset') == 'textvqa' and r.get('prompt_n') is not None and r['pos'] not in base:
            base[r['pos']] = r['prompt_n'] + (r.get('cache_n') or 0)
print(f"llama.cpp baseline tokens found for {sum(p in base for p in timed)}/{len(timed)} timed records")

rows, errors = [], 0
for i, pos in enumerate(timed):
    rec = recs[pos]
    t0 = time.time()
    resp = rx.call_server(rx.resolve_image_paths(rec, ds_dir), rx.vqa_prompt_short(rec), 32)
    dt = time.time() - t0
    if 'error' in resp:
        errors += 1
        print(f"[{i}] pos={pos} ERROR: {str(resp['error'])[:200]}")
        continue
    text = rx.extract_text(resp)
    fin = rx.extract_finish_reason(resp)
    vt = resp['usage']['prompt_tokens']
    lt = base.get(pos)
    rows.append([pos, rec.get('idx'), vt, lt, round(dt, 2), fin, '<think>' in text, text])
    if i < 10:
        print(f"[{i}] pos={pos} idx={rec.get('idx')} vllm_tok={vt} llama_tok={lt} {dt:.1f}s finish={fin}")
        print("     Q:", rec['question'][:80])
        print("     A:", repr(text))

with open('results_vllm/tokcheck_textvqa.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pos', 'idx', 'vllm_prompt_tokens', 'llama_prompt_tokens', 'sec', 'finish', 'has_think', 'answer'])
    w.writerows(rows)

print(f"\nok={len(rows)} errors={errors}")
pairs = [(r[2], r[3]) for r in rows if r[3]]
if pairs:
    ratios = [v / l for v, l in pairs]
    print(f"vllm/llama token ratio: median={statistics.median(ratios):.3f} min={min(ratios):.3f} max={max(ratios):.3f}")
    print(f"within 10%: {sum(abs(x - 1) <= 0.10 for x in ratios)}/{len(ratios)}")
    print(f"vllm tokens: median={statistics.median(p[0] for p in pairs)} max={max(p[0] for p in pairs)}; llama tokens: median={statistics.median(p[1] for p in pairs)} max={max(p[1] for p in pairs)}")
if rows:
    print(f"median sec/request: {statistics.median(r[4] for r in rows):.2f}")
    print(f"finish=length: {sum(r[5] == 'length' for r in rows)}, has <think>: {sum(r[6] for r in rows)}")
