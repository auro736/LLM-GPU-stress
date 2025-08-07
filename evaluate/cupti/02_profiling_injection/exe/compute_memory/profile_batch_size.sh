#!/bin/bash
export PATH="/home/bepi/anaconda3/bin:$PATH"
source /home/bepi/anaconda3/bin/activate
conda deactivate 

conda activate gpustress

batch_sizes=(16 32 64 128 256 512 1024 2048 4096 8912 10000)
mem_occ=(10 20 30 40 45 50 60 70 80 90 95)

for batch_size in "${batch_sizes[@]}"; do
    echo "Computing memory occupancy for ${batch_size} and comparing it with"
    for mem in "${mem_occ[@]}"; do
        echo "Memory occupancy: ${mem}"

        echo "For Mnasnet on CIFAR10"
        python exe/compute_memory/monitor_memory.py \
        --model_name mnasnet0_5 \
        --dataset_name CIFAR10 \
        --batch_size ${batch_size} \
        --occ_memory ${mem} \
        --exec_time 3600 \
        -monitor

        echo "For Mobilenet V2 on CIFAR10"
        python exe/compute_memory/monitor_memory.py \
        --model_name mobilenet_v2 \
        --dataset_name CIFAR10 \
        --batch_size ${batch_size} \
        --occ_memory ${mem} \
        --exec_time 3600 \
        -monitor

        echo "For Resnet18 on CIFAR10"
        python exe/compute_memory/monitor_memory.py \
        --model_name resnet18 \
        --dataset_name CIFAR10 \
        --batch_size ${batch_size} \
        --occ_memory ${mem} \
        --exec_time 3600 \
        -monitor
        
        echo "For LeNet5 on MNIST"
        python exe/compute_memory/monitor_memory.py \
        --model_name LeNet5 \
        --dataset_name MNIST \
        --batch_size ${batch_size} \
        --occ_memory ${mem} \
        --exec_time 3600 \
        -monitor
    done
done

python exe/compute_memory/monitor_memory.py --model_name mobilenet_v2 --dataset_name CIFAR10 --batch_size ${batch_size} --occ_memory ${mem} --exec_time 3600 -monitor