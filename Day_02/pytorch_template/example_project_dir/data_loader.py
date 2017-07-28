from torchvision import datasets
from torchvision import transforms
from torch.utils import data


def get_loader(batch_size=100, is_train=True, data_dir=None):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # (0, 1) => (-0.5, 0.5) => (-1, 1)
    ])

    dataset = datasets.MNIST(root=data_dir, train=is_train, transform=transform, download=False)

    loader = data.DataLoader(
        dataset=dataset, batch_size=100, shuffle=True)

    return loader
