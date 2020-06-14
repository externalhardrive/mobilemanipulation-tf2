#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --policy gaussian \
    --universe gym \
    --domain Locobot \
    --task ContinuousMultistepGrasping-v0 \
    --exp-name locobot-continuous-multistep-grasping-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 12222 \
