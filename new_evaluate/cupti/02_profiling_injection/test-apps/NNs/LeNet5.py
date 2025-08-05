import torch
import torch.nn as nn
import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="Model Evaluation Script.")
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
    return parser

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def evaluate(model, dataloader, device, iterations, verbose):
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
            for inputs, labels in dataloader:
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

# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)

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


def main(args):

    path = os.getcwd()
    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    test_dataset = torchvision.datasets.MNIST(
        root='~/dataset',
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
            ]
        ),
        download=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = LeNet5(10)
    
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(path, "test-apps/NNs/checkpoints/", 'LeNet'), map_location=torch.device("cpu"), weights_only=False
    #     )
    # )
    model = model.to(device)
    model.eval()

    model_opt = torch.compile(model)


    accuracy = evaluate(model_opt, test_loader, device, args.num_iterations, args.verbose)
            

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)