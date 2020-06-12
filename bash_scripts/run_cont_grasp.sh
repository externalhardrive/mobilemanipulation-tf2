#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task ContinuousMultistepGraspingEnv-v0 \
    --exp-name locobot-image-continuous-dense-grasping \
    --checkpoint-frequency 20 \
    --trial-cpus 6 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 11111 \