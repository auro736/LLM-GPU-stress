export INJECTION_KERNEL_COUNT=$1

export INJECTION_METRICS="idc__request_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_adu.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_ipa.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_lsu.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_tex.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__mio2rf_writeback_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__mio_pq_read_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__mio_pq_write_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__pipe_tensor_cycles_active.avg "

export PATH="/home/bepi/anaconda3/bin:$PATH"
source /home/bepi/anaconda3/bin/activate
conda deactivate 

conda activate gpustress

echo "mnasnet0_5"
# this inferences should occupy 90% of the memory with an epsilon of 189.75 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name mnasnet0_5 \
    --dataset_name CIFAR10 \
    --batch_size 10000 \
    --num_iterations 100 \
    --duration 350 > data/raw/SMMemory/NN50Percmnasnet05_$INJECTION_KERNEL_COUNT.txt
    