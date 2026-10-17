import json, sys, collections
from datetime import datetime
from zoneinfo import ZoneInfo
d = sys.argv[1] if len(sys.argv) > 1 else "results_latency/p1"
rec = json.load(open(d + "/../records_frozen.json"))
tv = rec["datasets"]["textvqa"]
kb, mp, large = tv.get("image_kb", {}), tv.get("image_megapixels", {}), set(tv["large_included"])
rs = [json.loads(l) for l in open(d + "/requests.jsonl")]
by = collections.defaultdict(list)
for r in rs:
    by[r["server_id"]].append(r)
z = ZoneInfo("America/Los_Angeles")
print("server  cfg  block         valid_before  1st_bad(pos, KB, MP, large?)   error                         mem_avail_before  Pacific time")
for sid in sorted(by):
    q = sorted(by[sid], key=lambda x: x["t"])
    bad = next((i for i, r in enumerate(q) if not r["valid"]), None)
    c = q[0]["config"]
    if bad is None:
        print(f"{sid:>6}  {c:>3}  all {len(q)} requests valid")
        continue
    r = q[bad]
    ok_before = sum(1 for x in q[:bad] if x["valid"] and x["phase"] == "timed")
    ts = datetime.fromtimestamp(r["t"], z)
    refused = sum(1 for x in q[bad:] if not x["valid"])
    print(f"{sid:>6}  {c:>3}  {r['block']:<12} {ok_before:>12}  ({r['pos']}, {kb.get(str(r['pos']))}, {mp.get(str(r['pos']))}, {r['pos'] in large})  "
          f"{(r['error'] or '')[:28]:<28}  {r['mem_avail_mb']!s:>16}  {ts:%m-%d-%Y %H:%M:%S}   (+{refused-1} more bad after it)")
