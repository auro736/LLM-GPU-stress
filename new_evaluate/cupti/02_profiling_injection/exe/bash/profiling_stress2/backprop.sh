export INJECTION_KERNEL_COUNT=$1

## INT
export INJECTION_METRICS="sm__pipe_imma_cycles_active.avg.pct_of_peak_sustained_active " 

## Floating point
export INJECTION_METRICS="sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active " 
export INJECTION_METRICS="sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active " 
export INJECTION_METRICS="sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active " 
export INJECTION_METRICS="sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active "

## SFU
export INJECTION_METRICS="sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active " 

## ALU
export INJECTION_METRICS="sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active " 

## Global memory
export INJECTION_METRICS="smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio" 

## Local memory
export INJECTION_METRICS="smsp__sass_average_data_bytes_per_sector_mem_local_op_ld.ratio " 

## Texture cache
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum " # global_hit_rate

## Constant cache
# export INJECTION_METRICS="sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active " #################

## Shared memory
export INJECTION_METRICS="sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active "

## Register Files
# export INJECTION_METRICS="sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active " #################


start_time=$(date +%s)
end_time=$((start_time + 300))

while [ "$(date +%s)" -lt "$end_time" ]; do
    now=$(date +%s)
    remaining=$((end_time - now))
    [ $remaining -le 0 ] && break
    env CUDA_INJECTION64_PATH=./libinjection.so ./test-apps/gpu-rodinia/bin/linux/cuda/backprop 65536 >> data/raw/stress/backprop_$INJECTION_KERNEL_COUNT.txt &
    app_pid=$!

    wait_timeout=$remaining
    (sleep "$wait_timeout" && kill -TERM $app_pid 2>/dev/null) & watchdog_pid=$!

    wait $app_pid 2>/dev/null
    kill -KILL $watchdog_pid 2>/dev/null
done