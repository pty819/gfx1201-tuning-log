#!/bin/bash
source "$HOME/rocm10/bin/rocm-env.sh"
export ZENDNN_ROOT="$HOME/ZenDNN/build/install"
for d in zendnnl/lib zendnnl/lib64 deps/aocldlp/lib deps/aocldlp/lib64 deps/onednn/lib deps/onednn/lib64 deps/libxsmm/lib deps/aoclutils/lib deps/aoclutils/lib64; do
  [[ -d "$ZENDNN_ROOT/$d" ]] && export LD_LIBRARY_PATH="$ZENDNN_ROOT/$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export ZENDNNL_MATMUL_ALGO=1
export HIP_FORCE_DEV_KERNARG=1
export HIP_VISIBLE_DEVICES=0
[[ -f /usr/share/vulkan/icd.d/radeon_icd.json ]] && export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
exec "$@"
