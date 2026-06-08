import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),        # input layer: 784 px (28x28) -> 256 neurons
            nn.ReLU(),                  # ReLU activation
            nn.Dropout(0.3),            # randomly zero 30% of neurons during training
            nn.Linear(256, 128),        # hidden layer: 256 -> 128 neurons
            nn.ReLU(),
            nn.Linear(128, 10)          # output layer: 128 -> 10 classes
        )
    
    def forward(self, x):
        return self.network(x)

    def print_parameters(self):
        total = 0
        
        for name, param in self.named_parameters():
            count = param.numel()
            total += count

            print(f"{name:30} shape:{str(param.shape):25} params: {count:,}")

        print(f"\n{"Total":30} {"":25} {total:,}")