#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task ImageNavigation-v0 \
    --exp-name locobot-image-navigation-test \
    --checkpoint-frequency 20 \
    --cpus 4 \
    --gpus 1 \
    --trial-cpus 4 \
    --trial-gpus 1 \
    --server-port 11111 \
    --run-eagerly False
