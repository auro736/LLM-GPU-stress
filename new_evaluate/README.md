# ScalableGPUMonitoring
This repo serves as a tool for simultaneously monitoring the behavior of an application for the stress induced by the device self-heating. To do so, we combine internal events measurements with operative telemetry to effectively measure the stress induced by different parallel workloads on GPUs. The proposed solution combines workload profiling with the online monitoring of system telemetry parameters. In particular, the profiling analysis describes the use of GPU resources by evaluating internal hardware events through the integrated and accessible Performance Counters (PCs) in a GPU. Moreover, the telemetry parameters provide the thermal, power, and clock frequency variables, resulting from executing parallel workloads that serve for monitoring the hardware internal state.

## Prerequisites
At the current repo status, this tool supports only versions of cuda 12.x
For running the python script, the python version is quite flexible. Anyway I will leave my python version for the sake of experiment reproducibility.
The experiments were run on a laptop MSI Cyborg 15 A13VF with an Intel Core i7-13620 CPU with 20 cores, 16 GB of RAM, and equipped with one NVIDIA GPU RTX 4060 with CUDA version 12.6.

## Framework setup
Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform. For the system requirements and installation instructions of cuda toolkit, please refer to [Linux Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and the [Windows Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Clone the repo and enter the repo folder
```bash
git clone https://github.com/GiuseppeEsposito98/ScalableGPUMonitoring.git
cd ScalableGPUMonitoring
```

You will have the possibility to profile python scripts as well, so I would recommend to setup conda environments:
1. To download and install conda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
2. Create and aactivate a conda environment
```bash
conda create -n gpustress
conda activate gpustress
```
3. Install the required packages
```bash
pip install -r requirements.txt
```

4. Now you require to compile the libraries for the profiling so enter the cupti/00_extensions/src/profilerhost_util folder
```bash
cd cupti/00_extensions/src/profilerhost_util
```
5. Make sure that, in the MakeFile, the paths assigned to the cuda libraries are correct (they should be because as soon as you install CUDA toolkit the deafult installation location is the one indicated in the variable CUDA_INSTALL_PATH of the MakeFile).

6. Once everything is set, you should be able to run the Makefile:
```bash
make clean
make
```
7. Repeat the same procedure from step 4 to step 6 for the libraries and MakeFile available in cupti/02_profiling_injection
```bash
cd ../../../02_profiling_injection 
make clean
make libinjection.so
``` 

8. To try the tool, you require some applications to run so you can go in test-apps folder where the script for running the inference of some neural networks are available but the CUDA script for running GPU-burn or some veriations of GPU-burn are not available.
```bash
cd test-apps
```

9. Please refer to the intallation guide available in [GPU-burn](https://github.com/wilicc/gpu-burn) for proceeding with it installation

# Framework usage

## How to test all the applications monitoring different parameters

1. You will need to lunch the experiments from the 02_profiling_injection folder
```bash
cd ~/ScalableGPUMonitoring/cupti/02_profiling_injection
```
2. The bash file available at bash/profile.sh gives the possibility to analyze the performance of the applications from several perspectives through CUPTI (for performance counters) and NVML (for telemetry data). Of course, the main aim of the tool is to analyze the stress, then here I report the command that you need to query the target metrics for the stress assessment.
```bash
bash exe/profile.sh stress
```
This command will execute the available applications for 5 minutes each waiting for the device cooldown for 30 minutes simultaneously monitoring Performance Counters data through CUPTI and telemetry monitoring through NVML. 
3. If everything runs correctly, you will see some txt files in cupti/02_profiling_injection/data/raw/stress (Performance counters data) and a corresponding number of csv files in cupti/02_profiling_injection/data/postprocessed/stress (telemetry data)
```bash
ls data/raw/stress
ls data/postprocessed/stress
```
4. An additional postprocessing step is needed for the PC data, for parsing them in csv files
```bash
bash exe/postprocess.sh stress
```
5. If everything runs correctly, you will see additional csv files in cupti/02_profiling_injection/data/postprocessed/stress (PC data)
```bash
ls data/postprocessed/stress
```
6. The notebooks available in cupti/02_profiling_injection/analysis allow to explore all the analysis performed on the available data. 

## How to profile a target application from the stress perspective
1. Before lunching the experiment you need to setup the environment to support your target application. Specially if it is a custom one. Then you need to create a script in ~/ScalableGPUMonitoring/cupti/02_profiling_injection/exe/bash/profiling_stress2/<app_name>.sh which will contain the setup for the stress monitoring along with the setup to run the application. 
```bash
touch ~/ScalableGPUMonitoring/cupti/02_profiling_injection/exe/bash/profiling_stress2/\<app_name\>.sh
nano ~/ScalableGPUMonitoring/cupti/02_profiling_injection/exe/bash/profiling_stress2/\<app_name\>.sh
```
2. Now you need to export the following variables that will specify the target metrics to profile and the profiling granularity level.
```bash
export INJECTION_KERNEL_COUNT=$1

# Workload
## Compute
export INJECTION_METRICS="sm__inst_executed.avg.per_cycle_elapsed " # Executed Ipc Elapsed
export INJECTION_METRICS=$INJECTION_METRICS"sm__instruction_throughput.avg.pct_of_peak_sustained_active " # SM Busy
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_executed.avg.per_cycle_active " # Executed Ipc Active
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_issued.avg.pct_of_peak_sustained_active " # Issue Slots Busy
export INJECTION_METRICS=$INJECTION_METRICS"sm__inst_issued.avg.per_cycle_active " # Issued Ipc Active
export INJECTION_METRICS=$INJECTION_METRICS"smsp__sass_thread_inst_executed_op_fp64_pred_on.sum "  # inst_fp_64
export INJECTION_METRICS=$INJECTION_METRICS"smsp__sass_thread_inst_executed_op_integer_pred_on.sum " # inst_integer

## Memory 
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

export INJECTION_METRICS=$INJECTION_METRICS"lts__t_sector_op_read_hit_rate.pct " # L2 hit rate read
export INJECTION_METRICS=$INJECTION_METRICS"lts__t_sector_op_write_hit_rate.pct " # L2 hit rate write

# Stall
## Memory
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_imc_miss_per_warp_active.pct " # stall_constant_memory_dependency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct " # stall_memory_dependency

## Controller
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_wait_per_warp_active.pct " # stall_exec_dependency
export INJECTION_METRICS=$INJECTION_METRICS"smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct " # stall_exec_dependency
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
``` 
3. In order to run the application while it is profiled you need to execute the following command:
```bash
env CUDA_INJECTION64_PATH=./libinjection.so \<path_to_app_binary\> \<app_parameters\> > data/raw/stress/\<app_name\>_$INJECTION_KERNEL_COUNT.txt
```
4. You will need to lunch the experiments from the 02_profiling_injection folder
```bash
cd ~/ScalableGPUMonitoring/cupti/02_profiling_injection
```
5. The bash file available at bash/profile.sh gives the possibility to analyze the performance of a specific application from the stress perspective through CUPTI (for performance counters) and NVML (for telemetry data). This will generate a complete report comprising all the metrics of interest for each tested application
```bash
bash exe/complete_stress_profile.sh
```
This command will execute the target applications for 5 minutes each waiting for the device cooldown for 30 minutes simultaneously monitoring Performance Counters data through CUPTI and telemetry monitoring through NVML. 
6. If everything runs correctly, you will see some txt files in cupti/02_profiling_injection/data/raw/stress (Performance counters data) and a corresponding number of csv files in cupti/02_profiling_injection/data/postprocessed/stress (telemetry data)
```bash
ls data/raw/stress2
ls data/postprocessed/stress2
```
7. And eventually a json file reporting the values of the metrics of interest
```bash
cat cupti/02_profiling_injection/data/postprocessed/stress/evaluation.json
```