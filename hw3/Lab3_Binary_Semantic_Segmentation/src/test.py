import torch
import torch.nn as nn

# Dummy input and target tensors
batch_size, num_classes, height, width = 4, 10, 32, 32
output = torch.randn(batch_size, num_classes, height, width)  # Model output
target = torch.randint(0, num_classes, (batch_size, height, width))  # Ground truth labels

# Check shapes
print(f"Output shape: {output.shape}")
print(f"Target shape: {target.shape}")

# Ensure the target is the correct shape
if target.dim() == 4:
    target = target.squeeze(1)  # Adjust target dimensions if necessary

# Loss function
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)

print(f"Loss: {loss.item()}")
