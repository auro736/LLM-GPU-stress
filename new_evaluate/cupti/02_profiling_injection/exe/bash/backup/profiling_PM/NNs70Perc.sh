export INJECTION_KERNEL_COUNT=$1

export INJECTION_METRICS="gpc__cycles_elapsed.avg.per_second "
export INJECTION_METRICS=$INJECTION_METRICS"gr__cycles_active.sum.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"tpc__warps_active_realtime.avg.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"sm__cycles_active.avg.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed.avg.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"dramc__sectors_op_read.avg.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"dramc__sectors_op_write.avg.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"pcie__read_bytes.avg.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"pcie__write_bytes.avg.pct_of_peak_sustained_elapsed "
export INJECTION_METRICS=$INJECTION_METRICS"pcie__read_bytes.sum  "
export INJECTION_METRICS=$INJECTION_METRICS"pcie__write_bytes.sum "

export PATH="/home/bepi/anaconda3/bin:$PATH"
source /home/bepi/anaconda3/bin/activate
conda deactivate 

conda activate gpustress

# this inferences should occupy 90% of the memory with an epsilon of 1756.5 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name mnasnet0_5 \
    --dataset_name CIFAR10 \
    --batch_size 10000 \
    --num_iterations 1383 > data/raw/PM/NN70Percmnasnet05_$INJECTION_KERNEL_COUNT.txt

# this inferences should occupy 90% of the memory with an epsilon of 532.5 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name mobilenet_v2 \
    --dataset_name CIFAR10 \
    --batch_size 4096 \
    --num_iterations 1659 > data/raw/PM/NN70Percmobilenetv2_$INJECTION_KERNEL_COUNT.txt

# this inferences should occupy 90% of the memory with an epsilon of 420.5 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name resnet18 \
    --dataset_name CIFAR10 \
    --batch_size 8912 \
    --num_iterations 1377 > data/raw/PM/NN70Percresnet18_$INJECTION_KERNEL_COUNT.txt

# this inferences should occupy 90% of the memory with an epsilon of 4958.5 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name LeNet5 \
    --dataset_name MNIST \
    --batch_size 10000 \
    --num_iterations 3606 > data/raw/PM/NN70PercLeNet5_$INJECTION_KERNEL_COUNT.txt