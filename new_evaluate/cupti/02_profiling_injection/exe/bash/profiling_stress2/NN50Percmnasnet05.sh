export INJECTION_KERNEL_COUNT=$1

## Computate workload
export INJECTION_METRICS="sm__inst_executed.avg.per_cycle_elapsed " # Executed Ipc Elapsed
export INJECTION_METRICS=$INJECTION_METRICS"sm__instruction_throughput.avg.pct_of_peak_sustained_active " # SM Busy
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed.avg.per_cycle_active " # Executed Ipc Active
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_issued.avg.pct_of_peak_sustained_active " # Issue Slots Busy
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_issued.avg.per_cycle_active " # Issued Ipc Active

## Memory workload
export INJECTION_METRICS=$INJECTION_METRICS"dram__bytes_read.sum.per_second " # dram_read_throughput
export INJECTION_METRICS=$INJECTION_METRICS"dram__bytes_write.sum.per_second " # dram_write_throughput

export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum " # global_hit_rate

export INJECTION_METRICS=$INJECTION_METRICS"lts__t_sector_op_read_hit_rate.pct " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"lts__t_sector_op_write_hit_rate.pct " # global_hit_rate

# Stall
## Computation
export INJECTION_METRICS=$INJECTION_METRICS"smsp__sass_thread_inst_executed_op_fp64_pred_on.sum "  # inst_fp_64
export INJECTION_METRICS=$INJECTION_METRICS"smsp__sass_thread_inst_executed_op_integer_pred_on.sum " # inst_integer
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct " # stall_exec_dependency

## Memory
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_imc_miss_per_warp_active.pct " # stall_constant_memory_dependency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct " # stall_memory_dependency

## Controller
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_wait_per_warp_active.pct " # stall_exec_dependency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_not_selected_per_warp_active.pct " # stall_not_selected
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_sleeping_per_warp_active.pct " # stall_sleeping
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_barrier_per_warp_active.pct " # stall_sync
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_membar_per_warp_active.pct " # stall_sync

# Throttle
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct " # stall_texture
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct " # stall_pipe_busy
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct " # stall_pipe_busy
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct " # stall_memory_throttle
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_drain_per_warp_active.pct " # stall_memory_throttle
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
    --duration 350 > data/raw/stress/NN50Percmnasnet05_$INJECTION_KERNEL_COUNT.txt
    