#!/bin/bash

echo "size,fill,ns"

for size in 32768 131072 524288 2097152 8388608
do
    for f in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95
    do
        echo ${size},${f},$(./gpu_lookup ${size} ${f})
    done
done
