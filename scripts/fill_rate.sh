#!/bin/bash

for f in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95
do
    echo ${f},$(./gpu_lookup ${f})
done
