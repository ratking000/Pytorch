import torch
from torch import nn

input_image = torch.rand(3,28,28)
print("input_image size = ", input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print("flat_image size = ", flat_image.size())
print("flat_image = ", flat_image)

# layer1 = nn.Linear(in_features=28*28, out_features=20)

seq_modules = nn.Sequential(
    flatten,
    nn.Linear(28*28, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

print(logits)

# print(f"Before ReLU: {hidden1}\n\n")
# hidden1 = nn.ReLU()(hidden1)
# print(f"After ReLU: {hidden1}")