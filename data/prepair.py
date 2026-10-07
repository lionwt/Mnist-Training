from torchvision import datasets, transforms

train_data = datasets.MNIST(
  root="./data",
  train=True,
  download=True,
  transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
  root="./data",
  train=False,
  download=True,
  transform=transforms.ToTensor()
)
