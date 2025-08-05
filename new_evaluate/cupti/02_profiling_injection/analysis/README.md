## Preliminary definitions

- **Active cycles**: The cycles in which the units was processing data
- **Issued intructions**: Assembly (SASS) instructions that generated at least one request to the L1
- **Executed instructions**: Assembly (SASS) instructions that generated at least one request as was actually executed 
- **Throttle**: In general throttling is the phenomenon where the device is taking too much load and starts overheating so that the Frequency is lowered. In the case of NVIDIA Performance Metrics. They can assume different definitions based on the reason (i.e., memory or computation overload).


## Metrics definitions

### Computational intensity

- **Issued Intructions per Active Cycle**: The instructions that were requested but not executed during the active cycles per Streaming Multiprocessor
- **Streaming Multiprocessor degree of  busyness**: SM instructions throughput as a percentage w.r.t. his peak performance 
- **Executed Instructions per Active Cycles**: The instructions that were requested and executed during the active cycles per Streaming Multiprocessor


### Memory efficiency usage

- **L2 hit percentage**: Hit rate at L2 cache for all read requests from texture cache
- **L1 hit percentage**: Hit rate for global loads in unified L1/TEX cache.
- **DRAM read bytes**: DRAM throughput in terms of read bytes
- **DRAM written bytes**: DRAM throughput in terms of written bytes


### Stall reasons distributions

- **Throttle stalls**: Percentage of stalls due to the GPU overloads either due to memory or compute capabilities (Including: stall_texture, stall_math_pipe_busy, stall_memory_throttle).
- **Controller stalls**: Percentage of stalls due to GPU scheduling policy (Including: stall_execution_dependency, stall_not_selected, stall_sleeping, stall_divergences/syncronization).
- **Memory stalls**: Percentage of stall due to memory wait (Including: stall_constant_memory_dependency, stall_memory_dependency).