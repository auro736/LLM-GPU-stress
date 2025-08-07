import argparse
import torch
import os
import math
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
        "--is_train", action="store_true", default=False,
        help="The current version of the script does not support the memory usage computation during the training process"
    )
    return parser


def get_transformations(dataset_name):
    """Return appropriate transformations based on dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name in ["cifar10", "cifar100"]:
        transform = transforms.Compose(
            [
            transforms.Resize((70, 70)),        
            transforms.CenterCrop((64, 64)),            
            transforms.ToTensor(),                
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    else:
        raise ValueError(f"No transformations defined for dataset {dataset_name}")
    return transform

def get_model(model_name):
    """Retrieve model from torchvision.models."""
    if model_name not in models.__dict__:
        raise ValueError(f"Model {model_name} not found in torchvision.models")
    
    model = models.__dict__[model_name]()

    if model_name == 'resnet18':
        model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
    elif model_name == 'mnasnet0_5' or model_name == 'mobilenet_v2':
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10, bias=True)

    return model

def get_dataset(dataset_name, transform):
    
    if dataset_name not in datasets.__dict__:
        raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets")
    
    dataset_class = datasets.__dict__[dataset_name]
    
    # Assume the usual train/test split datasets like CIFAR
    if 'cifar' in dataset_name.lower():
        testset = dataset_class(root="~/dataset", train=False, download=True, transform=transform)
    else: 
        raise ValueError(f"Add the proper initialization for {dataset_name} dataset")
    
    return testset

def evaluate(model, dataloader, device):
    counter = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            model(inputs)
            counter+=1
            if counter == 1:
                break
            


def get_model_size_mb(model):
    """Calcola la dimensione dei parametri del modello in MB."""
    total_params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params_bytes / (1024 ** 2)  # Converti in MB

def estimate_activation_size_mb(input_shape, model, batch_size=1, dtype=torch.float32):
    """Stima la memoria occupata dalle attivazioni in MB."""
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    
    # Calcolo approssimato: ogni layer produce un output della stessa grandezza dell'input
    # Non è esatto per modelli complessi, ma è una buona approssimazione rapida
    # (Potremmo migliorarlo layer per layer se vuoi)
    
    total_activation_elements = batch_size
    for dim in input_shape:
        total_activation_elements *= dim
    
    # Approssimazione: numero medio di feature maps è simile al numero di input channels
    # oppure potremmo considerare 2-3x rispetto all'input
    estimated_factor = 3  # Moltiplichiamo per 3 (grossolana approssimazione)
    
    total_activation_bytes = total_activation_elements * bytes_per_element * estimated_factor
    return total_activation_bytes / (1024 ** 2)  # Converti in MB

def estimate_total_inference_memory(model, input_shape, batch_size=1, dtype=torch.float32):
    """Stima la memoria totale necessaria per l'inferenza (parametri + attivazioni) in MB."""
    model_size = get_model_size_mb(model)
    activation_size = estimate_activation_size_mb(input_shape, model, batch_size, dtype)
    
    total_size = model_size + activation_size
    return total_size

       
def conv2d_output_shape(h_in, w_in, conv):
    # Usa la formula ufficiale di PyTorch
    kernel_h, kernel_w = conv.kernel_size
    stride_h, stride_w = conv.stride
    padding_h, padding_w = conv.padding
    dilation_h, dilation_w = conv.dilation

    h_out = math.floor((h_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1)
    w_out = math.floor((w_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1)
    return h_out, w_out

def linear_output_shape(in_features, linear):
    return linear.out_features

def estimate_model_memory(model, input_size, dtype=torch.float32):
    C, H, W = input_size[1], input_size[2], input_size[3]
    current_shape = (C, H, W)

    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_memory_bytes = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            C_out = module.out_channels
            H, W = conv2d_output_shape(H, W, module)
            current_shape = (C_out, H, W)
            elements = input_size[0] * C_out * H * W
            total_memory_bytes += elements * bytes_per_element

        elif isinstance(module, torch.nn.Linear):
            if isinstance(current_shape, tuple):
                features_in = torch.prod(torch.tensor(current_shape)).item()
            else:
                features_in = current_shape
            features_out = linear_output_shape(features_in, module)
            current_shape = features_out
            elements = input_size[0] * features_out
            total_memory_bytes += elements * bytes_per_element

        elif isinstance(module, (torch.nn.ReLU, torch.nn.BatchNorm2d, torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d)):
            if isinstance(module, torch.nn.MaxPool2d):
                H, W = conv2d_output_shape(H, W, module)
                current_shape = (C, H, W)

    total_memory_mb = total_memory_bytes / (1024 ** 2)
    return total_memory_mb



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_name)

    model = model.to(device)

    model_opt = torch.compile(model=model, mode='reduce-overhead')

    transform = get_transformations(args.dataset_name)
    testset = get_dataset(args.dataset_name, transform)

    memory_usage = estimate_activation_size_mb(input_shape=testset[0][0].shape, model=model)

    print(f'Memory usage estimation: {memory_usage} MB')
    # print(f'DISCLAIMER: since I am using torch.compile, the validation of the used formula to estimate memory consumption can be done with PYNVML only for during the first inference iteration because after that, the model weigths will be saved in the cache and reused for the received inputs')


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)