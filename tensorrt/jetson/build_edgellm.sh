#!/usr/bin/env bash
# NOTES.md step 2b: build Edge-LLM v0.9.1 + Python bindings. Runs inside jetson-llm:v1 (TensorRT 10.7, CUDA 12.6).
# Deviations from NOTES.md (JetPack 6.2's TensorRT 10.3 lacks APIs v0.9.1 uses):
#  - build in the course container (TensorRT 10.7.0.23) instead of the host
#  - CuTe DSL groups without f16_moe (MoE only; needs cudaLibrary_t from CUDA >= 12.8)
#  - patches/edgellm-v0.9.1-jp62-fp4-plugin-guard.patch (FP4 plugin needs TensorRT >= 10.8)
#  - link the CUDA 12.0-12.6 cudart shim and --wrap into executables (upstream only adds them to shared libs)
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd ~
[ -d TensorRT-Edge-LLM ] || git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout v0.9.1
git submodule update --init --recursive
echo "edge-llm at $(git rev-parse --short HEAD)"
nvcc --version | grep release
SHIM=$HOME/TensorRT-Edge-LLM/build/cpp/libtrt_edgellm_cutedsl_cudart_shim.a
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=12.6 \
  "-DENABLE_CUTE_DSL=ffpa;gdn;gemm;int4_fp16_gemm;ssd" \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--wrap=_cudaLaunchKernelEx" \
  "-DCMAKE_CXX_STANDARD_LIBRARIES=$SHIM /usr/local/cuda/targets/aarch64-linux/lib/stubs/libcuda.so"
make -j4
cd ..
pip3 install --user pybind11==2.13.6
TRT_PACKAGE_DIR=/usr python3 experimental/server/setup_pybind.py build_ext --inplace
ls -l build/examples/llm/llm_build build/examples/multimodal/visual_build build/examples/llm/llm_inference
find . -name '_edgellm_runtime*.so' | head
