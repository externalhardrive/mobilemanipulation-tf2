#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task ImageNavigation-v0 \
    --exp-name locobot-image-navigation-test \
    --checkpoint-frequency 20 \
    --trial-cpus 6 \
    --trial-gpus 2 \
    --server-port 11111 \
    --run-eagerly False \