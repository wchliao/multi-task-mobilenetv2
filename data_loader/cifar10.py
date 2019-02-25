import torch
import torchvision
from .base import BaseDataLoader
from .base import MultiTaskDataLoader

"""
CIFAR-10 is only used to test model. 
The multi-task loader is nearly the same as the single task loader.
"""

class CIFAR10Loader(BaseDataLoader):
    def __init__(self, batch_size=128, type='train', shuffle=True, drop_last=False):
        super(CIFAR10Loader, self).__init__(batch_size, type, shuffle, drop_last)

        self.batch_size = batch_size
        self.type = type
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._create_dataloader()


    def _create_dataloader(self):
        if self.type == 'train':
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.RandomCrop(32, padding=4),
                 torchvision.transforms.RandomHorizontalFlip(),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )

            dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform)

            num_data = len(dataset)
            index = list(range(num_data))
            sampler = torch.utils.data.sampler.SubsetRandomSampler(index[:45000])

            self.dataloader = torch.utils.data.DataLoader(dataset,
                                                          batch_size=self.batch_size,
                                                          sampler=sampler,
                                                          drop_last=self.drop_last)

        elif self.type == 'valid':
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )

            dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform)

            num_data = len(dataset)
            index = list(range(num_data))
            sampler = torch.utils.data.sampler.SubsetRandomSampler(index[45000:])

            self.dataloader = torch.utils.data.DataLoader(dataset,
                                                          batch_size=self.batch_size,
                                                          sampler=sampler,
                                                          drop_last=self.drop_last)

        elif self.type == 'test':
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )

            dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)

            self.dataloader = torch.utils.data.DataLoader(dataset, 
                                                          batch_size=self.batch_size,
                                                          drop_last=self.drop_last)
        else:
            raise ValueError('Unknown data type: {}'.format(type))


    def get_loader(self, task=None):
        if task is None:
            return MultiTaskDataLoader([self.dataloader])
        else:
            return self.dataloader


    @property
    def image_size(self):
        return 32


    @property
    def num_channels(self):
        return 3


    @property
    def num_classes(self):
        return [10]
