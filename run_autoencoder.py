#%% Imports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils.models.simplenet import SimpleNet


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
model = SimpleNet()
print(model)
print(model.print_parameters())

# run dummy forward pass to validate shape
dummy_input = torch.randn(8, 784) # batch of 8 images, each flattened to 784
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}") # should be [8, 10]


#%% 
