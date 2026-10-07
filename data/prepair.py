from torchvision import datasets

train_data = datasets.MNIST(
  root="./",
  train=True,
  download=True,
)

test_data = datasets.MNIST(
  root="./",
  train=False,
  download=True,
)
