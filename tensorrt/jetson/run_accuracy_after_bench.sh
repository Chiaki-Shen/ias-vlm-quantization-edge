#!/bin/bash
# Waits for the latency benchmark to finish, then runs the full T8 and T4 accuracy runs (resumable).
cd ~/ias-vlm-quantization-edge
L=~/tensorrt-edgellm-workspace/logs
while [ ! -s results_latency_trt/full/summary.txt ]; do sleep 30; done
sleep 60
for C in T8 T4; do
  echo "=== $C start $(date)" >> $L/accuracy_progress.log
  bash tensorrt/jetson/run_in_container.sh python3 tensorrt/run_trt_experiment.py --config $C --resume > $L/accuracy_$C.log 2>&1
  echo "=== $C exit $? $(date)" >> $L/accuracy_progress.log
done
echo ALL_DONE >> $L/accuracy_progress.log
