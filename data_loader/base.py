import numpy as np
import torch
import torchvision


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class BaseDataLoader:
    def __init__(self, batch_size=1, type='train', shuffle=True, drop_last=False):
        pass

    def get_loader(self, task):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError


def images_transform(images, transform):
    images = [torchvision.transforms.ToPILImage()(image) for image in images]
    images = torch.stack([transform(image) for image in images])

    return images


class SingleTaskDataLoader:
    def __init__(self, dataloader, transform=None):
        self.dataloader = dataloader
        self.transform = transform

    def __iter__(self):
        self.iter = iter(self.dataloader)
        return self

    def __next__(self):
        data, labels = self.iter.__next__()
        if self.transform:
            data = images_transform(data, self.transform)

        return data, labels


class MultiTaskDataLoader:
    def __init__(self, dataloaders, transform=None):
        self.dataloaders = [SingleTaskDataLoader(loader, transform) for loader in dataloaders]

        self.num_tasks = len(dataloaders)
        self.task_order = list(range(self.num_tasks))
        self.size = max([len(d) for d in dataloaders]) * self.num_tasks


    def __iter__(self):
        self.iters = [iter(loader) for loader in self.dataloaders]
        self.task_step = 0
        self.data_step = 0

        return self


    def __next__(self):
        if self.data_step >= self.size:
            self.data_step = 0
            raise StopIteration

        if self.task_step >= self.num_tasks:
            np.random.shuffle(self.task_order)
            self.task_step = 0

        task = self.task_order[self.task_step]

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.task_step += 1
        self.data_step += 1

        return data, labels, task