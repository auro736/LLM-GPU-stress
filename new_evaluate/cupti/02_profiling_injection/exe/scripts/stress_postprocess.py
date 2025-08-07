import os
import pandas as pd
from pynvml import *
import argparse
import ruptures as rpt
import json
import numpy as np

def get_argparser():
    parser = argparse.ArgumentParser(description='Postprocessing for .txt data')
    parser.add_argument('--performance', required=True, help='Either Performance Metrics (PM) or Performance Counters (PC)')
    return parser

def main(args):
   
    mapping_table = {
    'gpuburn5min': 'GPU-burn',
    'NN50Perclenet5': 'LeNet5',
    'NN50Percmnasnet05': 'MnasNet',
    'NN50Percmobilenetv2': 'MobileNetV2',
    'NN50Percresnet18': 'ResNet18',
    'backprop': 'Back Propagation',
    'gaussian': 'Gaussian Elimination',
    'hotspot': 'Hotspot',
    'needle': 'Needleman-Wunsch',
    'scgpu': 'Stream Cluster',
    'rora' : 'rora', 
    'vectoradd' : 'vectoradd'
        }

    location_mapping={
        'sm': 'Streaming Multiprocessor',
        'dram': 'Dynamic RAM',
        'l1tex': 'L1 Cache',
        'lts': 'L2 Cache',
        'smsp': 'Streaming Multiprocessor SubPartition'
        }
    
    metric_event_mapping= {
    # Workload
    ## Compute
    'request_cycles_active': 'Number of cycles where the IDC processed requests from SM',
    'instruction_throughput': 'Instruction throughput',
    'inst_executed': 'Executed instructions',
    'inst_issued': 'Issued instructions',
    'sass_thread_inst_executed_op_fp64_pred_on': 'Instructions FP64',
    'sass_thread_inst_executed_op_integer_pred_on': 'Instructions Integers',

    ## Memory
    ### DRAM
    'bytes_read': 'Read Bytes',
    'bytes_write': 'Written bytes',

    ### L1 Cache
    't_sectors_pipe_lsu_mem_global_op_ld_lookup_hit': 'Global Memory Load Sectors – Cache Hit (per Thread Set via LSU)',
    't_sectors_pipe_lsu_mem_global_op_st_lookup_hit': 'Global Memory Store Sectors – Cache Hit (per Thread Set via LSU)',
    't_sectors_pipe_lsu_mem_global_op_red_lookup_hit': 'Global Memory Reduction – Cache Hit (per Thread Set via LSU)',
    't_sectors_pipe_lsu_mem_global_op_atom_lookup_hit': 'Global Memory Atomic – Cache Hit (per Thread Set via LSU)',
    't_sectors_pipe_lsu_mem_global_op_ld': ' Global Memory Load Sectors Served by L1 Cache (via LSU)',
    't_sectors_pipe_lsu_mem_global_op_st': ' Global Memory Store Sectors Served by L1 Cache (via LSU)',
    't_sectors_pipe_lsu_mem_global_op_red': 'Global Memory Reduction Sectors Served by L1 Cache (via LSU)',
    't_sectors_pipe_lsu_mem_global_op_atom': 'Global Memory Atomic Sectors Served by L1 Cache (via LSU)',
    
    ### L2 Cache
    't_sector_op_read_hit_rate': 'L2 hit rate by read instruction',
    't_sector_op_write_hit_rate': 'L2 hit rate by write instruction',

    # Stall
    ## Memory
    'warp_issue_stalled_imc_miss_per_warp_active': 'Warp Issue Stalls Due to IMC (Immediate Constant Cache) Misses per Active Warp ',
    'warp_issue_stalled_long_scoreboard_per_warp_active': 'Warp Issue Stalls Due to Long Scoreboard (Long Wait for Resource) per Active Warp',

    ## Controller
    'warp_issue_stalled_short_scoreboard_per_warp_active': 'Warp Issue Stalls Due to Short Scoreboard (Resource Wait) per Active Warp',
    'warp_issue_stalled_wait_per_warp_active': 'Warp Issue Stalls Due to Wait (Resource/Data Not Ready) per Active Warp',
    'warp_issue_stalled_not_selected_per_warp_active': 'Warp Issue Stalls Due to Not Being Selected per Active Warp',
    'warp_issue_stalled_sleeping_per_warp_active': 'Warp Issue Stalls Due to Sleeping per Active Warp',
    'warp_issue_stalled_membar_per_warp_active': 'Warp Issue Stalls Due to Membar per Active Warp',
    'warp_issue_stalled_barrier_per_warp_active': 'Warp Issue Stalls Due to Barrier per Active Warp',
    'warp_issue_stalled_dispatch_stall_per_warp_active': 'Warp Issue Stalls Due to Dispatch Stall per Active Warp',

    ## Throttle
    'warp_issue_stalled_drain_per_warp_active': 'Warp Issue Stalls Due to Drain (Memory/Resource Write Completion) per Active Warp',
    'warp_issue_stalled_lg_throttle_per_warp_active': 'Warp Issue Stalls Due to Large Unit Throttling (Resource Limitation) per Active Warp',
    'warp_issue_stalled_math_pipe_throttle_per_warp_active': 'Warp Issue Stalls Due to Math Pipe Throttling per Active Warp',
    'warp_issue_stalled_mio_throttle_per_warp_active': 'Warp Issue Stalls Due to MIO Throttling per Active Warp',
    'warp_issue_stalled_tex_throttle_per_warp_active': 'Warp Issue Stalls Due to Texture Throttling per Active Warp',

    ## Others
    'warp_issue_stalled_misc_per_warp_active': 'Warp Issue Stalls Due to Miscellaneous Issues per Active Warp',

    }
    
    data_path = f'data/postprocessed/{args.performance}'
    app_names = pd.Series([file.split('_')[0] for file in os.listdir(data_path) if file.endswith('csv')]).unique()
    print(app_names)
    total_json = {}
    for app in app_names:
        if not app in list(mapping_table.keys()):
            mapping_table[app] = app
        total_json[f'{mapping_table[app]}'] = {}
        print(mapping_table)

        ## Performance counters data processing
        pc_file_name = f'{app}_1.csv'
        pc_file_path = os.path.join(data_path, pc_file_name)
        print(pc_file_path)
        pc_csv = pd.read_csv(pc_file_path)

        pc_csv['app'] = mapping_table[f'{app}']
        pc_csv['progress'] = (pc_csv['session_id'] - pc_csv['session_id'].min()) / (pc_csv['session_id'].max() - pc_csv['session_id'].min()) * 100
        if app.split('_')[0] == 'hotspot':
            pc_csv['Index'] = range(len(pc_csv))
            pc_csv['progress'] = (pc_csv['Index'] - pc_csv['Index'].min()) / (pc_csv['Index'].max() - pc_csv['Index'].min()) * 100
        pc_csv['Range'] = 1

        pc_csv['HR_location'] = pc_csv['location'].map(location_mapping)

        pc_csv['HR_metric_name'] = pc_csv['metric_name'].map(metric_event_mapping)

        df_l2 = pc_csv[pc_csv['HR_location']=='L2 Cache']
        df_sm = pc_csv[pc_csv['HR_location']=='Streaming Multiprocessor']
        df_smsp = pc_csv[pc_csv['HR_location']=='Streaming Multiprocessor SubPartition']
        df_l1 = pc_csv[pc_csv['HR_location']=='L1 Cache']
        df_dram = pc_csv[pc_csv['HR_location']=='Dynamic RAM']
        
        df_pivot_l2 = df_l2.pivot_table(
            index=["progress", "HR_location", "range_name", "Range", "app", 'rollup_operation', 'Post'],
            columns="HR_metric_name",
            values="metric_value"
        ).reset_index()

        df_pivot_sm = df_sm.pivot_table(
            index=["progress", "HR_location", "range_name", "Range", "app", 'rollup_operation', 'Post'],
            columns="HR_metric_name",
            values="metric_value"
        ).reset_index()

        df_pivot_smsp = df_smsp.pivot_table(
            index=["progress", "HR_location", "range_name", "Range", "app", 'rollup_operation', 'Post'],
            columns="HR_metric_name",
            values="metric_value"
        ).reset_index()

        # print(df_smsp['app'])

        df_pivot_smsp['Memory Stall']=(df_pivot_smsp['Warp Issue Stalls Due to IMC (Immediate Constant Cache) Misses per Active Warp '] +\
                    df_pivot_smsp['Warp Issue Stalls Due to Long Scoreboard (Long Wait for Resource) per Active Warp']) /2

        df_pivot_smsp['Controller Stall']=(df_pivot_smsp['Warp Issue Stalls Due to Not Being Selected per Active Warp'] +\
                        df_pivot_smsp['Warp Issue Stalls Due to Short Scoreboard (Resource Wait) per Active Warp'] +\
                        df_pivot_smsp['Warp Issue Stalls Due to Wait (Resource/Data Not Ready) per Active Warp'] +\
                        df_pivot_smsp['Warp Issue Stalls Due to Sleeping per Active Warp'] +\
                        df_pivot_smsp['Warp Issue Stalls Due to Membar per Active Warp'] +\
                        df_pivot_smsp['Warp Issue Stalls Due to Barrier per Active Warp'] ) /7

        df_pivot_smsp['Throttle Stall']=(df_pivot_smsp['Warp Issue Stalls Due to Drain (Memory/Resource Write Completion) per Active Warp'] +\
                                        df_pivot_smsp['Warp Issue Stalls Due to Large Unit Throttling (Resource Limitation) per Active Warp'] +\
                                        df_pivot_smsp['Warp Issue Stalls Due to Math Pipe Throttling per Active Warp'] +\
                                        df_pivot_smsp['Warp Issue Stalls Due to MIO Throttling per Active Warp'] +\
                                        df_pivot_smsp['Warp Issue Stalls Due to Texture Throttling per Active Warp']) /5

        df_pivot_l1 = df_l1.pivot_table(
            index=["progress", "HR_location", "range_name", "Range", "app", 'rollup_operation', 'Post'],
            columns="HR_metric_name",
            values="metric_value"
        ).reset_index()

        df_pivot_l1['Global hit rate'] = (df_pivot_l1['Global Memory Atomic – Cache Hit (per Thread Set via LSU)']+ \
                                        df_pivot_l1['Global Memory Load Sectors – Cache Hit (per Thread Set via LSU)']+\
                                        df_pivot_l1['Global Memory Reduction – Cache Hit (per Thread Set via LSU)']+\
                                        df_pivot_l1['Global Memory Store Sectors – Cache Hit (per Thread Set via LSU)']) / \
                                        (df_pivot_l1[' Global Memory Load Sectors Served by L1 Cache (via LSU)']+ \
                                        df_pivot_l1[' Global Memory Store Sectors Served by L1 Cache (via LSU)']+\
                                        df_pivot_l1['Global Memory Atomic Sectors Served by L1 Cache (via LSU)']+\
                                        df_pivot_l1['Global Memory Reduction Sectors Served by L1 Cache (via LSU)'])

        df_pivot_dram = df_dram.pivot_table(
            index=["progress", "HR_location", "range_name", "Range", "app", 'rollup_operation', 'Post'],
            columns="HR_metric_name",
            values="metric_value"
        ).reset_index()

        pivot_dfs = {
            'L2 Cache': df_pivot_l2, 
            'Streaming Multiprocessor': df_pivot_sm, 
            'Streaming Multiprocessor SubPartition': df_pivot_smsp, 
            'L1 Cache': df_pivot_l1, 
            'Dynamic RAM': df_pivot_dram
            }

        a = pivot_dfs['L1 Cache'].groupby(by=["app"])[['Global hit rate']]\
            .quantile(0.75)\
                .reset_index()
        b = pivot_dfs['L2 Cache'].groupby(by=["app"])[[ 'L2 hit rate by read instruction','L2 hit rate by write instruction']]\
            .quantile(0.75)\
                .reset_index()

        c = pivot_dfs['Streaming Multiprocessor'].groupby(by=["app"])[[ 'Executed instructions','Instruction throughput', 'Issued instructions']]\
            .quantile(0.75)\
                .reset_index()

        d = pivot_dfs['Dynamic RAM'].groupby(by=["app"])[[ 'Read Bytes', 'Written bytes']]\
            .quantile(0.75)\
                .reset_index()

        e = pivot_dfs['Streaming Multiprocessor SubPartition'].groupby(by=["app"])[['Memory Stall', 'Controller Stall', 'Throttle Stall']]\
            .quantile(0.75)\
                .reset_index()

        merge_1 = pd.merge(a, b[['app', 'L2 hit rate by read instruction','L2 hit rate by write instruction']], on='app')
        merge_2 = pd.merge(merge_1, c[['app', 'Executed instructions','Instruction throughput', 'Issued instructions']], on='app')
        merge_3 = pd.merge(merge_2, d[['app','Read Bytes', 'Written bytes']], on='app')
        final_merge = pd.merge(merge_3, e[['app', 'Memory Stall', 'Controller Stall', 'Throttle Stall']], on='app')

        for key in [col for col in final_merge.columns]:
            if key != 'app':
                total_json[f'{mapping_table[app]}'][f'{key}'] = float(final_merge[key])
        ## Telemetry data processing
        telemetry_file_name = f'{app}_1_telemetry.csv'
        telemetry_file_path = os.path.join(data_path, telemetry_file_name)
        telemetry_csv = pd.read_csv(telemetry_file_path)

        telemetry_csv['Index'] = range(len(telemetry_csv))
        telemetry_csv['progress'] = telemetry_csv['Index'].transform(
            lambda x: 100 * (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0)
        
        duration = telemetry_csv['Index'].max()

        signal = np.array(telemetry_csv['temperature_C'])
        algo = rpt.Pelt(model="rbf").fit(signal)
        result = algo.predict(pen=51)
        response_time = result[0]

        steady_temp = telemetry_csv[telemetry_csv['Index']>125]['temperature_C'].mean()

        telemetry_csv['total_energy_J'] = telemetry_csv['total_energy_mJ']/1000
        spent_energy = telemetry_csv[['total_energy_J']].max() - telemetry_csv[['total_energy_J']].min()
        spent_energy['mean_energy_J'] = spent_energy['total_energy_J']/(duration/60)
        cf = telemetry_csv['clock_sm_MHz'].mean()

        new_row = {'Steady Temp °C':steady_temp,
                   'Energy spent J/min': spent_energy['mean_energy_J'],
                   'Clock Frequency MHz': cf,
                   'response (s)': response_time
                   }
        

        total_json[f'{mapping_table[app]}']['Steady Temp °C'] = steady_temp,
        total_json[f'{mapping_table[app]}']['Energy spent J/min'] = spent_energy['mean_energy_J'],
        total_json[f'{mapping_table[app]}']['Clock Frequency MHz'] = cf,
        total_json[f'{mapping_table[app]}']['response (s)'] = response_time,
        
    with open(f"{data_path}/evaluation.json", "w", encoding="utf-8") as f:
        json.dump(total_json, f, indent=4, ensure_ascii=False)
        
        


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())