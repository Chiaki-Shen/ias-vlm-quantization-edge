#!/bin/bash
# T16 accuracy: ScienceQA first (answers the ScienceQA-gap question), then the rest. Swap file still on (accuracy only).
cd ~/ias-vlm-quantization-edge
L=~/tensorrt-edgellm-workspace/logs
( while true; do echo "$(date +%T) $(free -m | awk "/Mem/{print \$3, \$7}") $(free -m | awk "/Swap/{print \$3}")"; sleep 2; done ) > $L/mem_t16_run.log &
M=$!
for D in scienceqa blink_depth,blink_spatial,cv_bench,textvqa; do
  echo "=== T16 $D start $(date)" >> $L/t16_progress.log
  bash tensorrt/jetson/run_in_container.sh python3 -u tensorrt/run_trt_experiment.py --config T16 --datasets $D --resume > $L/accuracy_T16_${D%%,*}.log 2>&1
  RC=$?
  echo "=== T16 $D exit $RC $(date)" >> $L/t16_progress.log
  [ $RC -ne 0 ] && break
done
kill $M
echo ALL_DONE >> $L/t16_progress.log
