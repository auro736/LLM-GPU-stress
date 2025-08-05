export INJECTION_KERNEL_COUNT=$1

export INJECTION_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active " # achieved_occupancy
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum " # atomic_transactions
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum " # atomic_transactions
export INJECTION_METRICS=$INJECTION_METRICS"smsp__sass_average_branch_targets_threads_uniform.pct " # branch_efficiency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_pipe_cbu.sum " # branch_efficiency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_pipe_adu.sum " # branch_efficiency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active " # double_precision_fu_utilization
export INJECTION_METRICS=$INJECTION_METRICS"dram__bytes_read.sum " # dram_read_bytes
export INJECTION_METRICS=$INJECTION_METRICS"dram__bytes_read.sum.per_second " # dram_read_throughput
export INJECTION_METRICS=$INJECTION_METRICS"dram__throughput.avg.pct_of_peak_sustained_elapsed " # dram_utilization
export INJECTION_METRICS=$INJECTION_METRICS"dram__bytes_write.sum " # dram_write_bytes
export INJECTION_METRICS=$INJECTION_METRICS"dram__bytes_write.sum.per_second " # dram_write_throughput
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warps_eligible.sum.per_cycle_active " # eligible_warps_per_cycle
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum " # global_hit_rate
export INJECTION_METRICS=$INJECTION_METRICS"smsp__inst_executed.sum " # inst_executed
export INJECTION_METRICS=$INJECTION_METRICS"smsp__sass_thread_inst_executed_op_fp64_pred_on.sum "  # inst_fp_64
export INJECTION_METRICS=$INJECTION_METRICS"smsp__sass_thread_inst_executed_op_integer_pred_on.sum " # inst_integer
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_imc_miss_per_warp_active.pct " # stall_constant_memory_dependency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct " # stall_exec_dependency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_wait_per_warp_active.pct " # stall_exec_dependency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct " # stall_memory_dependency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_drain_per_warp_active.pct " # stall_memory_throttle
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct " # stall_memory_throttle
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_not_selected_per_warp_active.pct " # stall_not_selected
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct " # stall_other
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_misc_per_warp_active.pct " # stall_other
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct " # stall_pipe_busy
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct " # stall_pipe_busy
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_sleeping_per_warp_active.pct " # stall_sleeping
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_barrier_per_warp_active.pct " # stall_sync
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_membar_per_warp_active.pct " # stall_sync
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct " # stall_texture
export INJECTION_METRICS=$INJECTION_METRICS"smsp__thread_inst_executed_per_inst_executed.ratio " # warp_execution_efficiency

export PATH="/home/bepi/anaconda3/bin:$PATH"
source /home/bepi/anaconda3/bin/activate
conda deactivate 

conda activate gpustress



echo "mobilenet_v2"
# this inferences should occupy 90% of the memory with an epsilon of 1365.5 MB for 1 hour
env CUDA_INJECTION64_PATH=./libinjection.so python3 /home/bepi/Desktop/Ph.D_/projects/GPU_stress/code/ScalableGPUMonitoring/cupti/02_profiling_injection/test-apps/NNs/evaluate.py\
    --model_name mobilenet_v2 \
    --dataset_name CIFAR10 \
    --batch_size 2048 \
    --num_iterations 100 \
    --duration 350 > data/raw/LaunchStats/NN50Percmobilenetv2_$INJECTION_KERNEL_COUNT.txt



    