#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationDQNGrasping-v0 \
    --exp-name locobot-navigation-dqn-grasping-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 12222 \
