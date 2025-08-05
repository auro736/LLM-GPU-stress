#!/bin/bash
PERFORMANCE=stress
main_directory="exe/bash/postprocessing"
# cd $main_directory

kernels=(1)

for kernel in "${kernels[@]}"; do
    for file in "$main_directory"/*; do
        # if [ -f "$file" ]; then
        # if [[ "$file" == *"resnet"* || "$file" == *"lenet"* || "$file" == *"mnasnet"* || "$file" == *"gpuburn5min"* ]]; then
            echo "Processing: $file"
            bash $file $kernel $PERFORMANCE
        # fi
    done
done
