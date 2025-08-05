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

python3 exe/gpu_telemetry_querying.py --file_name gpuburn5min_1_prova --performance $PERFORMANCE &
PID_CONTROLLER=$!

./test-apps/gpu-burn/gpu_burn -i 0 -c ./test-apps/gpu-burn/compare.ptx -m 50% > data/raw/stress/gpuburn5min_1_prova.txt 300

kill "$PID_CONTROLLER"

wait "$PID_CONTROLLER" 2>/dev/null

echo "End run"