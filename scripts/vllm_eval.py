import sys, json, time, statistics, argparse
from pathlib import Path
sys.path.insert(0, 'scripts')
import run_experiment as rx
rx.BASE_URL = "http://localhost:8000"

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', default='scienceqa,textvqa')
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--label', default='vLLM_FP8w16kv')
ap.add_argument('--outdir', default='results_vllm')
a = ap.parse_args()

config = {"short": a.label}
out_dir = Path(a.outdir)
out_dir.mkdir(exist_ok=True)

for ds in a.datasets.split(','):
    spec = rx.DATASET_SPECS[ds]
    fmt = rx.FORMATS[spec['format']]
    prompt_fn = fmt['prompts'][spec['prompt']]
    ds_dir = Path('benchmarks') / ds
    recs = [json.loads(l) for l in open(ds_dir / 'data.jsonl') if l.strip()][:a.n]
    out_path = out_dir / f"{a.label}__{ds}.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                done.add(json.loads(l)['rec_no'])
            except Exception:
                pass
    print(f"\n[{ds}] {len(recs)} records, {len(recs) - len(done)} to run -> {out_path}", flush=True)
    consec_err = 0
    t_ds = time.time()
    with open(out_path, 'a', encoding='utf-8') as out_f:
        for i, rec in enumerate(recs):
            if i in done:
                continue
            paths = rx.resolve_image_paths(rec, ds_dir)
            prompt = prompt_fn(rec)
            t0 = time.time()
            resp = rx.call_server(paths, prompt, spec['max_tokens'])
            retried = False
            if 'error' in resp:
                retried = True
                time.sleep(3)
                t0 = time.time()
                resp = rx.call_server(paths, prompt, spec['max_tokens'])
            wall = time.time() - t0
            emsg, etype = "", None
            if 'error' in resp:
                emsg, etype = resp.get('error', 'unknown'), resp.get('error_type', 'Error')
                consec_err += 1
            else:
                consec_err = 0
            line = rx.build_line(config, ds, spec, rec, i, resp, retried,
                                 rx.get_mem_mb()[0], emsg, etype)
            line['wall_s'] = round(wall, 3)
            line['prompt_tokens'] = (resp.get('usage') or {}).get('prompt_tokens')
            out_f.write(json.dumps(line) + "\n")
            out_f.flush()
            if emsg:
                print(f"  [!] rec {i} error: {str(emsg)[:150]}", flush=True)
            if consec_err >= 5:
                print("  5 consecutive errors, aborting (server probably died). Re-run to resume.", flush=True)
                sys.exit(1)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(recs)} done, {time.time() - t_ds:.0f}s elapsed", flush=True)
    lines = [json.loads(l) for l in open(out_path)]
    ok = [l for l in lines if not l.get('error')]
    sc = [l['score'] for l in ok if l.get('score') is not None]
    print(f"[{ds}] records={len(lines)} errors={len(lines) - len(ok)} "
          f"mean_score={100 * sum(sc) / max(len(lines), 1):.2f}% "
          f"median_wall={statistics.median(l['wall_s'] for l in ok):.2f}s "
          f"finish_length={sum(l.get('finish_reason') == 'length' for l in ok)} "
          f"think={sum('<think>' in l.get('prediction_raw', '') for l in ok)}", flush=True)
