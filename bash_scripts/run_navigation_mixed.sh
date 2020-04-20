#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task MixedNavigation-v0 \
    --exp-name locobot-mixed-navigation-test \
    --checkpoint-frequency 20 \
    --cpus 1 \
    --gpus 0 \
    --trial-cpus 1 \
    --trial-gpus 0 \
    --server-port 11111 \
    --run-eagerly False
