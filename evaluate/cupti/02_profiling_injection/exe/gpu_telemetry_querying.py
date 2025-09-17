import csv
import time
import os
from datetime import datetime
from pynvml import *
import argparse
from pynvml import (
    NVML_MEMORY_ERROR_TYPE_CORRECTED,
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
    NVML_VOLATILE_ECC,
    NVML_AGGREGATE_ECC,
    NVML_CLOCK_SM, 
    NVML_TEMPERATURE_THRESHOLD_GPU_MAX, 
    NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
)


def sample_telemetry(device):
    timestamp = time.perf_counter_ns() #
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    name = nvmlDeviceGetName(device)
    temp = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU)
    util = nvmlDeviceGetUtilizationRates(device)
    mem = nvmlDeviceGetMemoryInfo(device)
    clock_sm = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM)
    clock_max_sm = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM)
    clock_mem = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM)
    clock_gr = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS)
    max_temp = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX)
    slowdown_temp = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)

    try:
        total_energy = nvmlDeviceGetTotalEnergyConsumption(device)
    except NVMLError:
        total_energy = -1
    
    try:
        clock_freq = nvmlDeviceGetAdaptiveClockInfoStatus(device)
    except NVMLError:
        clock_freq = -1

    try:
        adaptive_clock_freq = nvmlDeviceGetCurrentClockFreqs(device)
    except NVMLError:
        adaptive_clock_freq = -1

    try:
        fan = nvmlDeviceGetFanSpeed(device)
    except NVMLError:
        fan = -1

    try:
        power = nvmlDeviceGetPowerUsage(device) / 1000.0
    except NVMLError:
        power = -1

    # ECC errors
    try:
        ecc_vol_corr = nvmlDeviceGetTotalEccErrors(
            device, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_VOLATILE_ECC
        )
    except NVMLError:
        ecc_vol_corr = -1

    try:
        ecc_vol_uncorr = nvmlDeviceGetTotalEccErrors(
            device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_VOLATILE_ECC
        )
    except NVMLError:
        ecc_vol_uncorr = -1

    try:
        ecc_agg_corr = nvmlDeviceGetTotalEccErrors(
            device, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_AGGREGATE_ECC
        )
    except NVMLError:
        ecc_agg_corr = -1

    try:
        ecc_agg_uncorr = nvmlDeviceGetTotalEccErrors(
            device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_AGGREGATE_ECC
        )
    except NVMLError:
        ecc_agg_uncorr = -1

    return [
        timestamp,
        0,
        name,
        temp,
        max_temp,
        slowdown_temp,
        util.gpu,
        util.memory,
        mem.total // (1024 * 1024),
        mem.used // (1024 * 1024),
        mem.free // (1024 * 1024),
        clock_sm,
        clock_mem,
        clock_gr,
        fan,
        power,
        ecc_vol_corr,
        ecc_vol_uncorr,
        ecc_agg_corr,
        ecc_agg_uncorr,
        total_energy,
        clock_freq,
        adaptive_clock_freq,
        clock_max_sm
    ]

def get_argparser():
    parser = argparse.ArgumentParser(description='Postprocessing for .txt data')
    parser.add_argument('--file_name', required=True, help='Input file name')
    parser.add_argument('--performance', required=True, help='Either Performance Metrics (PM) or Performance Counters (PC)')
    return parser

def main(args):

    # Nome file
    csv_file = f"data/postprocessed/{args.performance}/{args.file_name}_telemetry.csv"

    # Inizializza NVML
    nvmlInit()
    device = nvmlDeviceGetHandleByIndex(0)  # GPU 0

    

    # Header CSV
    header = [
        "timestamp_ns", "gpu_index", "name", "temperature_C", "max_temp", "slowdown_temp",
        "util_gpu_percent", "util_mem_percent",
        "mem_total_MB", "mem_used_MB", "mem_free_MB",
        "clock_sm_MHz", "clock_mem_MHz", "clock_graphics_MHz",
        "fan_speed_percent", "power_draw_W", "ecc_volatile_corrected", 
        "ecc_volatile_uncorrected", "ecc_aggregate_corrected", "ecc_aggregate_uncorrected",
        "total_energy_mJ", 'clock_freq', 'adaptive_clock_frequency', 'clock_max_sm'
    ]

    # Sampling period (in nanosecondi)
    sampling_ns = 1_000_000_000  # 1_000_000_000 ns = 1 s


    # Scrivi header
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Loop di logging
    try:
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            while True:
                row = sample_telemetry(device)
                writer.writerow(row)
                f.flush()  # salva subito
                time.sleep(sampling_ns / 1e9)  # converti ns → sec
    except KeyboardInterrupt:
        print("Interrotto dall'utente.")
    finally:
        nvmlShutdown()


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())