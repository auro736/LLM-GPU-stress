#!/bin/bash
# InstructionStats, ComputeWorkloadAnalysis, LaunchStats, Occupancy, Memory
PERFORMANCE=$1

main_directory="exe/bash/profiling_$PERFORMANCE"
inter='_'
# cd $main_directory

export PATH="/home/bepi/anaconda3/bin:$PATH"
source /home/bepi/anaconda3/bin/activate
conda deactivate 

conda activate gpustress

kernels=(1)

for kernel in "${kernels[@]}"; do
    for file in "$main_directory"/*; do
        # if ![[ "$file" == *"resnet"* || "$file" == *"lenet"* || "$file" == *"mnasnet"* || "$file" == *"gpuburn5min"* ]]; then
            # if [[ "$file" == *"gpuburn5min"* ]]; then

            echo "Executing telemetry controller in parallel"
            echo $file

            IFS='/' read -ra parts <<< "$file"

            IFS='.' read -ra parts1 <<< "${parts[3]}"
            date +"%c"
            
            python3 exe/gpu_telemetry_querying.py --file_name ${parts1[0]}$inter$kernel --performance $PERFORMANCE &
            PID_CONTROLLER=$!

            echo "Executing: $file"
            bash $file $kernel 

            kill "$PID_CONTROLLER"

            wait "$PID_CONTROLLER" 2>/dev/null

            echo "End run"
            sleep 1800
        # fi
    done
done
