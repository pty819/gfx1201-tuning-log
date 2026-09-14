#!/bin/bash
# Compile inside the vllm-radiance container (ROCm 7.14, gfx1201):
#   podman cp w4_hip_demo.cu <container>:/tmp/w4demo/  etc, then:
hipcc -O3 -std=c++17 --offload-arch=gfx1201 -fPIC -shared w4_hip_demo.cu -o libw4demo.so
