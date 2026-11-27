#!/bin/bash
# TensorRT audit board runs (tensorrt/AUDIT.md), resumable step by step:
#   1. build the recalibrated LLM engines (cr2b-int4awq-mm, cr2b-int8sq-mm); INT8 needs the NVMe swap file
#   2. wait until the desktop is stopped (same condition as the T8 / T4 accuracy runs)
#   3. ScienceQA for T8m and T4m (the dataset with the gap), then the per-record token audit on both runtimes,
#      then TextVQA for T8m and T4m
cd ~/ias-vlm-quantization-edge
L=~/tensorrt-edgellm-workspace/logs; P=$L/audit_progress.log; W=~/tensorrt-edgellm-workspace
mkdir -p $L
log() { echo "$* $(date)" >> $P; }
LLAMA() { docker run --rm --runtime=nvidia --network host --user "$(id -u):$(id -g)" -e HOME=$HOME -v $HOME:$HOME \
  -v /Developer:/Developer -v /Developer/llama-cpp-new:/opt/llama-cpp-new -e LD_LIBRARY_PATH=/opt/llama-cpp-new/build/bin \
  -w $HOME/ias-vlm-quantization-edge cmpelkk/jetson-llm:latest "$@"; }

for C in cr2b-int4awq-mm cr2b-int8sq-mm; do
  if [ -s $W/$C/engines/llm/llm.engine ] || ls $W/$C/engines/llm/*.engine >/dev/null 2>&1; then log "build $C already done"; continue; fi
  log "build $C start"
  bash tensorrt/jetson/run_in_container.sh bash tensorrt/jetson/build_llm_only.sh $C > $L/build_$C.log 2>&1
  RC=$?; log "build $C exit $RC"
  [ $RC -ne 0 ] && { log "STOPPED: build failed"; exit 1; }
done

log "waiting for desktop to be stopped (gdm=$(systemctl is-active gdm))"
while systemctl is-active --quiet gdm; do sleep 30; done
sleep 60
log "desktop stopped; runs start ($(free -m | awk '/Mem/{print "avail="$7"MB"}'))"

run_trt() {  # config datasets
  log "accuracy $1 $2 start"
  bash tensorrt/jetson/run_in_container.sh python3 -u tensorrt/run_trt_experiment.py --config $1 --datasets $2 --resume \
    > $L/accuracy_$1_$2.log 2>&1
  log "accuracy $1 $2 exit $?"
}

run_trt T8m scienceqa
run_trt T4m scienceqa

if [ ! -s results/audit/tokens_trt_textvqa.jsonl ]; then
  log "tokens trt start"
  bash tensorrt/jetson/run_in_container.sh python3 -u tensorrt/audit_tokens.py --runtime trt --engine cr2b-int8sq \
    --datasets scienceqa,textvqa --textvqa-limit 500 > $L/tokens_trt.log 2>&1
  log "tokens trt exit $?"
fi
if [ ! -s results/audit/tokens_llamacpp_textvqa.jsonl ]; then
  log "tokens llamacpp start"
  LLAMA python3 -u tensorrt/audit_tokens.py --runtime llamacpp --datasets scienceqa,textvqa --textvqa-limit 500 \
    > $L/tokens_llamacpp.log 2>&1
  log "tokens llamacpp exit $?"
fi
python3 tensorrt/audit_tokens.py --compare --datasets scienceqa,textvqa > $L/tokens_compare.log 2>&1
log "tokens compare exit $?"

run_trt T8m textvqa
run_trt T4m textvqa
log AUDIT_DONE
