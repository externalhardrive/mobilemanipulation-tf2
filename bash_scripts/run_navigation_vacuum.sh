#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task NavigationVacuum-v0 \
    --exp-name locobot-navigation-vacuum-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 0 \
    --run-eagerly False \
    --server-port 11112 \
