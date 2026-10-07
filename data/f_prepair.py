from torchvision import datasets, transforms

f_train_data = datasets.FashionMNIST(
  root="./data",
  train=True,
  download=True,
  transform=transforms.ToTensor()
)

f_test_data = datasets.FashionMNIST(
  root="./data",
  train=False,
  download=True,
  transform=transforms.ToTensor()
)