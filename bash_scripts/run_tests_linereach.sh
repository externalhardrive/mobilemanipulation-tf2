#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SACMixed \
    --universe gym \
    --domain Tests \
    --task LineReach-v0 \
    --exp-name tests-line-reach-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 11112 \
