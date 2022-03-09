import torchvision
from torchvision import transforms
from augment.cutmix import CutMix
from augment.mixup import Mixup
from augment.maxup import Maxup
from augment.cutmixup import CutMixup
from torch.utils.data import DataLoader


def get_data(data_dir, dataset):
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(), ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), ]
    )
    if dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
        test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=data_dir, train=True, transform=train_transform, download=True)
        test_data = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=2),
             transforms.ToTensor(), ]
        )
        num_classes = 10
        train_data = torchvision.datasets.SVHN(root=data_dir, split='train', transform=train_transform, download=True)
        test_data = torchvision.datasets.SVHN(root=data_dir, split='test.log', transform=test_transform, download=True)
    return train_data, test_data, num_classes


def loader_maxup(data_dir, dataset, augment, batch_size, workers, m):
    train_data, test_data, num_classes = get_data(data_dir, dataset)
    if augment == 'cutmix':
        train_data = CutMix(train_data, m, num_classes=num_classes)
    elif augment == 'mixup':
        train_data = Mixup(train_data, m, num_classes=num_classes)
    elif augment == 'cutmixup':
        train_data = CutMixup(train_data, m, num_classes=num_classes)
    else:
        train_data = Maxup(train_data, m, num_classes=num_classes)
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=workers, shuffle=True
    )
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=workers, shuffle=False
    )
    return train_data_loader, test_data_loader, num_classes

