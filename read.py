import torch

# Loading a tensor or model from the .et file (if it's a PyTorch format)
tensor_or_model = torch.load('test2.0.et', weights_only=False)

# Now you can use the loaded tensor or model
print(tensor_or_model)

