import importlib
import torch

try:
  torch_directml = importlib.import_module("torch_directml")
except ImportError:
  torch_directml = None


def get_device():
  if torch_directml is not None and torch_directml.is_available():
    print("Using DirectML:", torch_directml.device_name(0).rstrip("\x00"))
    return torch_directml.device()

  if torch.cuda.is_available():
    print("Using CUDA")
    return torch.device("cuda")

  if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("Using XPU")
    return torch.device("xpu")

  print("Using CPU")
  return torch.device("cpu")
