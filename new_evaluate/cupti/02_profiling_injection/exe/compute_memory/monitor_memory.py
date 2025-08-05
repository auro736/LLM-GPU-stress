import argparse
import torch
import os
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pynvml_utils import nvidia_smi
import pandas as pd
from time import time
import math

def get_gpu_memory_used_nvml():
    nvsmi = nvidia_smi.getInstance()
    occupied = float(nvsmi.DeviceQuery('memory.used')['gpu'][0]['fb_memory_usage']['used'])
    total = float(nvsmi.DeviceQuery('memory.total')['gpu'][0]['fb_memory_usage']['total'])
    return occupied, total
    # return used_gb

def get_argparser():
    parser = argparse.ArgumentParser(description="Model Evaluation Script.")
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name of the model from torchvision.models (e.g., resnet18, mobilenet_v2). Please provide the correct name as the name of the function in torchvision that instantiate the Neural Netowrk"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="Name of the dataset from torchvision.datasets (e.g., CIFAR10, CIFAR100). Please provide the correct name as the name of the function in torchvision that instantiate the Dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for evaluation (default: 64)"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1,
        help="Number of times to run inference over the dataset (default: 1)"
    )
    parser.add_argument(
        "--occ_memory", type=int, default=50,
        help="Experimental setup for GPU burn memory occupancy (in percentage)"
    )
    parser.add_argument(
        "--exec_time", type=int, default=60,
        help="Experimental setup for GPU burn execution time (in seconds)"
    )
    parser.add_argument(
        "-monitor", action="store_true",
        help="Verbose mode to show the memory usage during NN inference (default: False)"
    )
    return parser

class LeNet5(torch.nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = torch.nn.Linear(400, 120)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(120, 84)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(84, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out



def get_transformations(dataset_name):
    """Return appropriate transformations based on dataset."""
    dataset_name = dataset_name.lower()

    if 'cifar' in dataset_name.lower():
        transform = transforms.Compose(
            [
            transforms.Resize((70, 70)),        
            transforms.CenterCrop((64, 64)),            
            transforms.ToTensor(),                
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    elif 'mnist' in dataset_name.lower():
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
            ]
        )

    else:
        raise ValueError(f"No transformations defined for dataset {dataset_name}")
    return transform

def get_model(model_name):
    """Retrieve model from torchvision.models."""
    print(f'Running the inference of {model_name}')
    if model_name == 'LeNet5': 
        model = LeNet5(10)
    else:
        if model_name not in models.__dict__:
            raise ValueError(f"Model {model_name} not found in torchvision.models")
        
        model = models.__dict__[model_name]()

        if model_name == 'resnet18':
            model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        elif model_name == 'mnasnet0_5' or model_name == 'mobilenet_v2':
            model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10, bias=True)


    return model

def get_dataset(dataset_name, transform):
    root_data_path = '~/dataset'
    if dataset_name not in datasets.__dict__:
        raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets")
    
    dataset_class = datasets.__dict__[dataset_name]
    print(dataset_class)
    if 'cifar' in dataset_name.lower() or 'mnist' in dataset_name.lower():
        testset = dataset_class(root=root_data_path, 
                                train=False, 
                                download=True, 
                                transform=transform)
    else: 
        raise ValueError(f"Add the proper initialization for {dataset_name} dataset")

    return testset

def evaluate(model, dataloader, device, monitor, iterations):
    """Evaluate the model on the test set."""
    model.eval()
    correct = 0
    total = 0
    accuracy = ''
    with torch.no_grad():
        for it_idx in range(iterations):
            start_time = time()
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            accuracy = f'{100 * correct / total}%'
            end_time = time()
            if it_idx % 20:
                print(f'Running iteration: {it_idx}')


    return accuracy, end_time-start_time



def main(args):
    target_memory_percentage = args.occ_memory

    occupied_start, total_start = get_gpu_memory_used_nvml()

    available_start = ((total_start - occupied_start)*target_memory_percentage)/100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_name)

    model = model.to(device)

    transform = get_transformations(args.dataset_name)
    testset = get_dataset(args.dataset_name, transform)

    try:
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        accuracy, inf_time = evaluate(model, testloader, device, args.monitor, args.num_iterations)
        occupied_end, total_end = get_gpu_memory_used_nvml()

        act_occupied = occupied_end - occupied_start

        # Presa la memoria utilizzata da gpu_burn e quella effettivamente utilizzata dalla rete neurale, 
        # epsilon lo definisco come una misura di distanza tra le due
        # di conseguenza, io voglio prendere la batch size che minimizza l'epsilon
        epsilon = available_start - act_occupied

        iterations = f'{math.floor(args.exec_time//inf_time)}'
        one_inf_time = f'{inf_time}s' 
        epsilon_form = f'{epsilon}MB'
        act_occupied_form = f'{act_occupied}MB'

    except:
        act_occupied=f'Not available'
        epsilon_form=f'Not available'
        iterations=f'Not available'
        act_occupied_form = f'Not available'
        ValueError(f"Batch size {args.batch_size} exceeds the available memory {total_start}")
    
    finally:
        row = pd.DataFrame({
            'Target_memory': f'{args.occ_memory}%',
            'Target_time': f'{args.exec_time}s',
            'Neural_Network': args.model_name,
            'Dataset': args.dataset_name,
            'Batch_size': args.batch_size,
            'Occupied_memory': act_occupied_form,
            'Memory_epsilon': epsilon_form,
            'One_iteration_inference_time': one_inf_time,
            'Iterations': iterations
        }, index=[0])
        print(row)

    # available_end = ((total_end - occupied_end)*target_memory_percentage)/100

    

    sheet_root_path = './exe/compute_memory'
    sheet_name = 'exploratory_analysis.csv'

    sheet_path = os.path.join(sheet_root_path, sheet_name)

    if os.path.exists(sheet_path):
        sheet_df = pd.read_csv(sheet_path)
        sheet_df = pd.concat([sheet_df, row], ignore_index=True)
    else:
        sheet_df = row

    sheet_df.to_csv(sheet_path, index=False)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
