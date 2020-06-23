#!/bin/bash
softlearning run_example_local examples.perturbation \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationVacuumRNDPerturbation-v0 \
    --exp-name locobot-navigation-vacuum-rnd-perturbation-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 11112 \
