import torch

# in my case this will almost always be cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if __name__ == "__main__":
    print(f"You are using {device} device")