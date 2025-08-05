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

env CUDA_INJECTION64_PATH=./libinjection.so ./test-apps/gpu-burn/gpu_burn -i 0 -c ./test-apps/gpu-burn/compare.ptx -m 50% > data/raw/SMMemory/gpuburn5min_$INJECTION_KERNEL_COUNT.txt 300