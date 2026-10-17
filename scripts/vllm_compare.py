import json, random, glob, sys
from pathlib import Path

N = 200
random.seed(249)

def load(path):
    d = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            d[r['rec_no']] = r
    return d

def one(pattern):
    f = glob.glob(pattern)
    if len(f) != 1:
        sys.exit(f"expected exactly one file for {pattern}, got {f}")
    return f[0]

def boot_ci(diffs, reps=10000):
    n = len(diffs)
    means = sorted(sum(random.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]

out = []
def say(s=""):
    print(s)
    out.append(s)

for ds in ['scienceqa', 'textvqa']:
    say(f"===== {ds} (rec_no 0 to {N - 1}) =====")
    sysmap = {
        'vLLM pass1': load(f'results_vllm/vLLM_FP8w16kv__{ds}.jsonl'),
        'vLLM pass2': load(f'results_vllm/vLLM_FP8w16kv_pass2__{ds}.jsonl'),
        'Config 2':   load(one(f'archive/*/config2_*__{ds}.jsonl')),
        'Config 5':   load(one(f'archive/*/config5_*__{ds}.jsonl')),
    }
    keys = [k for k in range(N) if all(k in m and not m[k].get('error') for m in sysmap.values())]
    say(f"records usable in all four files: {len(keys)}/{N}")
    bad = sum(1 for k in keys if len({(m[k]['idx'], m[k]['question']) for m in sysmap.values()}) != 1)
    say(f"idx/question mismatches across files: {bad}")

    p1, p2 = sysmap['vLLM pass1'], sysmap['vLLM pass2']
    same_raw = sum(p1[k]['prediction_raw'] == p2[k]['prediction_raw'] for k in keys)
    same_norm = sum(p1[k]['prediction_normalized'] == p2[k]['prediction_normalized'] for k in keys)
    say(f"pass1 vs pass2 identical raw text: {same_raw}/{len(keys)}, identical normalized answer: {same_norm}/{len(keys)}")

    say("accuracy on these records:")
    for name in ['Config 2', 'Config 5', 'vLLM pass1', 'vLLM pass2']:
        s = [sysmap[name][k]['score'] for k in keys]
        say(f"  {name:11s} {100 * sum(s) / len(s):6.2f}%")

    def paired(a, b):
        da = [sysmap[a][k]['score'] - sysmap[b][k]['score'] for k in keys]
        mean = sum(da) / len(da)
        lo, hi = boot_ci(da)
        wins = sum(x > 0 for x in da)
        loss = sum(x < 0 for x in da)
        chg = sum(sysmap[a][k]['prediction_normalized'] != sysmap[b][k]['prediction_normalized'] for k in keys)
        say(f"  {a} minus {b}: {100 * mean:+.2f} pts  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  "
            f"first better on {wins}, second better on {loss}, answers differ on {chg}")

    say("paired differences (same records):")
    paired('vLLM pass1', 'Config 2')
    paired('vLLM pass1', 'Config 5')
    paired('Config 5', 'Config 2')
    say()

Path('results_vllm/compare_summary.txt').write_text("\n".join(out) + "\n")
print("saved results_vllm/compare_summary.txt")
