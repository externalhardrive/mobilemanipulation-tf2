#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task MixedNavigation-v0 \
    --exp-name locobot-mixed-navigation-test \
    --checkpoint-frequency 10 \
    --trial-cpus 6 \
    --trial-gpus 1 \
    --run-eagerly False \
    # --server-port 11112 \
