#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SACDiscrete \
    --policy discrete \
    --universe gym \
    --domain Locobot \
    --task DiscreteGraspingEnv-v0 \
    --exp-name tests-grasp-discrete-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 11112 \
