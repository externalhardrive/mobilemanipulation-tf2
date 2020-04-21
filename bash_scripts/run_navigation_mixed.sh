#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task MixedNavigation-v0 \
    --exp-name locobot-mixed-navigation-test \
    --checkpoint-frequency 20 \
    --cpus 4 \
    --gpus 1 \
    --trial-cpus 4 \
    --trial-gpus 1 \
    --server-port 11112 \
    --run-eagerly False
