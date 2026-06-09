#%% Imports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils.models.simplenet import SimpleNet
from torch.utils.data import DataLoader, TensorDataset


print(torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name())
print("Imported")

#%% Test tensors + tensor conversion
t = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])
print("\nTensor:\n", t)
print("Shape:", t.shape)
print("Type:", t.dtype) # torch.float32

arr = t.numpy()
back = torch.from_numpy(arr)
print("\nNumpy:", arr)
print("Tensor:", back)
print("\nMean:", t.mean())
print("Sum:", t.sum())
print("Transposed:", t.T)

#%% Build neural network
model_test = SimpleNet()
print(model_test)
print(model_test.print_parameters())

print("Running forward dummy pass to validate shape...")

# 2D tensor 8x784 representing batch of 8 images, each "flattened" to 784 pixels (each has normal distribution)
dummy_input = torch.randn(8, 784) 

output = model_test(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}") # should be [8, 10]


#%% Training loop
# initialize synthetic dataset (stand-in for real image data)
torch.manual_seed(42)
X = torch.randn(1000, 784) # 1000 rand-normal dist samples, 784 features each
y = torch.randint(0, 10, (1000,)) # 1000 random class labels, spaced between 0-9
dataset = TensorDataset(X, y) # simulates a table of y columns and X rows

# set data loader and shuffling batches
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model, loss, optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# loop with 5 epochs
num_epochs = 5

for epoch in range(num_epochs):
    model.train() # sets model to training mode and enables dropout
    running_loss = 0.0 # loss accumulator

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad() # clear gradients from previous step
        outputs = model(batch_X) # forward pass
        loss = criterion(outputs, batch_y) # compute loss
        loss.backward() # backward pass - compute gradients
        optimizer.step() # update weights
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

print("\nTraining complete")
# %%
