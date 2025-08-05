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

# this inferences should occupy 90% of the memory with an epsilon of 701.00 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name mnasnet0_5 \
    --dataset_name CIFAR10 \
    --batch_size 2048 \
    --num_iterations 2835 > data/raw/PC/NN20Percmnasnet05_$INJECTION_KERNEL_COUNT.txt

# this inferences should occupy 90% of the memory with an epsilon of 215.00 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name mobilenet_v2 \
    --dataset_name CIFAR10 \
    --batch_size 1024 \
    --num_iterations 2309 > data/raw/PC/NN20Percmobilenetv2_$INJECTION_KERNEL_COUNT.txt

# this inferences should occupy 90% of the memory with an epsilon of 257.00 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name resnet18 \
    --dataset_name CIFAR10 \
    --batch_size 2048 \
    --num_iterations 2708 > data/raw/PC/NN20Percresnet18_$INJECTION_KERNEL_COUNT.txt

# this inferences should occupy 90% of the memory with an epsilon of 1041.00 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name LeNet5 \
    --dataset_name MNIST \
    --batch_size 10000 \
    --num_iterations 3606 > data/raw/PC/NN20PercLeNet5_$INJECTION_KERNEL_COUNT.txt