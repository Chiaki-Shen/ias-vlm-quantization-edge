# Oct 4 reproducibility reruns (Configs 2, 5, 8; BLINK + CV-Bench)

Re-run with the current runner (protocol p2.1, script sha256 4bfae0d4) to check
the legacy Sep 23-24 results in `results/`, which were produced by an earlier runner.

Result: all 2,905 predictions per config match the legacy files exactly
(0 question/image mismatches, 0 prediction differences, identical 118-record
CV-Bench error set). The records that needed a server restart and retry were
different between the two runs, yet every answer matched, so the retries do not
affect accuracy.

A llama-server OOM kill at 20:51 PDT on Oct 4 fell in the last minutes of the
Config 8 CV-Bench pass; the runner restarted and retried, and no record was lost.
