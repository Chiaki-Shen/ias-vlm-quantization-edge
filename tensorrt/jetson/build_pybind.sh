#!/usr/bin/env bash
# Python bindings from the main build tree, so they get the same Orin flags (setup_pybind.py hard-codes jetson-thor).
set -euo pipefail
cd ~/TensorRT-Edge-LLM/build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -Dpybind11_DIR=$(python3 -c "import pybind11;print(pybind11.get_cmake_dir())")
make -j4 _edgellm_runtime
find . -name "_edgellm_runtime*.so"
