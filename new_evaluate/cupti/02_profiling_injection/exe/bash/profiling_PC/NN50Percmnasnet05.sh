export INJECTION_KERNEL_COUNT=$1

export INJECTION_METRICS="sm__cycles_active.sum "
export INJECTION_METRICS=$INJECTION_METRICS"sys__cycles_active.sum "
export INJECTION_METRICS=$INJECTION_METRICS"sm__warps_active.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_op_generic_atom_dot_alu.sum "
export INJECTION_METRICS=$INJECTION_METRICS"sm__cycles_elapsed.sum "
export INJECTION_METRICS=$INJECTION_METRICS"sys__cycles_elapsed.sum "
export INJECTION_METRICS=$INJECTION_METRICS"dram__sectors_read.sum "
export INJECTION_METRICS=$INJECTION_METRICS"dram__sectors_write.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_op_generic_atom_dot_cas.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_op_global_red.sum "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_pipe_fma.sum "
# export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_pipe_fp16.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_pipe_fp64.sum "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_issued.sum "
export INJECTION_METRICS=$INJECTION_METRICS"lts__t_sectors_op_read_lookup_miss.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warps_launched.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__thread_inst_executed.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_op_local_st.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_op_local_ld.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_op_global_st.sum "
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_op_global_ld.sum"

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
    --duration 450 > data/raw/PC/NN50Percmnasnet05_$INJECTION_KERNEL_COUNT.txt
    