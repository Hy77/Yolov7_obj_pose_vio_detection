import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())

# this will choose GPU instead of CPU automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")