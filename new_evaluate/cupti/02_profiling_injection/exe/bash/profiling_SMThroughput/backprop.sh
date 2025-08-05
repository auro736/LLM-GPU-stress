
export INJECTION_KERNEL_COUNT=$1

export INJECTION_METRICS="idc__request_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_adu.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_cbu_pred_on_any.avg "
# export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_fp16.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_ipa.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_lsu.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_tex.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_uniform.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed_pipe_xu.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__issue_active.avg  "
export INJECTION_METRICS=$INJECTION_METRICS"sm__mio2rf_writeback_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__mio_inst_issued.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__mio_pq_read_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__mio_pq_write_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__pipe_alu_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__pipe_fp64_cycles_active.avg "
# export INJECTION_METRICS=$INJECTION_METRICS"sm__pipe_fma_cycles_active.avg "
# export INJECTION_METRICS=$INJECTION_METRICS"sm__pipe_shared_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__pipe_tensor_cycles_active.avg "
export INJECTION_METRICS=$INJECTION_METRICS"sm__throughput.avg "

env CUDA_INJECTION64_PATH=./libinjection.so ./test-apps/gpu-rodinia/bin/linux/cuda/backprop 65536 > data/raw/SMThroughput/backprop_$INJECTION_KERNEL_COUNT.txt
