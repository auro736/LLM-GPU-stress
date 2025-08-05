import argparse
import torch
import os
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
from time import time

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
        "-verbose", action="store_true",
        help="Verbose mode to show the speedup gained from the use of torch.compile() (default: False)"
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Experiment duration in seconds"
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

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

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

def evaluate(model, dataloader, device, iterations, duration, start, verbose):
    """Evaluate the model on the test set."""
    model.eval()
    correct = 0
    total = 0
    accuracy = {}

    eager_times = []
    compile_times = []
    eager_times_per_it = []

    with torch.no_grad():
        for it_idx in range(iterations):
            compile_times_per_it = []
            print(f'Running iteration: {it_idx}')
            step = 0
            for inputs, labels in dataloader:
                step += 1
                print(f'Running step: {step}')
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, elapsed_time = timed(lambda: model(inputs))
                if it_idx==0 and verbose:
                    eager_times_per_it.append(elapsed_time)
                    # print(f'Eager eval time: {elapsed_time}')
                elif it_idx != 0 and verbose: 
                    compile_times_per_it.append(elapsed_time)
                    # print(f'Eval time from compiled NN: {elapsed_time}')
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                end = time()
                print(f'Duration: {end-start}')
                if end-start > duration:
                    break
            if end-start > duration:
                break
            if it_idx==0 and verbose:
                eager_times.append(np.median(eager_times_per_it))
            elif it_idx!=0 and verbose:
                compile_times.append(np.median(compile_times_per_it))
            accuracy[it_idx] = f'{100 * correct / total}%'

    if verbose:
        eager_med = np.median(eager_times)
        compile_med = np.median(compile_times)
        speedup = eager_med / compile_med
        print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
        print("~" * 10)

    return accuracy

def main(args):
    start = time()
    logging.info(f'Running the inference of {args.model_name} on {args.dataset_name}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_ckpt_path = './test-apps/NNs/checkpoints/'
    ckpt_path = os.path.join(root_ckpt_path, args.model_name)
    # ckpt_path = ckpt_path + '.pth'
    model = get_model(args.model_name)
    # model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model = model.to(device)

    # model_opt = torch.compile(model=model, mode='reduce-overhead')

    transform = get_transformations(args.dataset_name)
    testset = get_dataset(args.dataset_name, transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    accuracy = evaluate(model, testloader, device, args.num_iterations, args.duration, start, args.verbose)
    # print(f"Test Accuracies per iteration: {accuracy}")

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
