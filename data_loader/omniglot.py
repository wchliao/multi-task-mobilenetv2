import os
import torch
import torchvision
import numpy as np
from PIL import Image
from .base import CustomDataset
from .base import BaseDataLoader
from .base import SingleTaskDataLoader, MultiTaskDataLoader


class OmniglotLoader(BaseDataLoader):
    def __init__(self, batch_size=128, type='train', shuffle=True, drop_last=False):
        super(OmniglotLoader, self).__init__(batch_size, type, shuffle, drop_last)

        self.batch_size = batch_size
        self.type = type
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Read data

        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

        root = './data'
        omniglot_path = os.path.join(root, 'omniglot-py')

        images = []
        labels = []

        for dirname, background in zip(['images_background', 'images_evaluation'], [True, False]):
            _ = torchvision.datasets.Omniglot(root=root, background=background, download=True)
            background_path = os.path.join(omniglot_path, dirname)

            for language in sorted(os.listdir(background_path)):
                language_path = os.path.join(background_path, language)
                task_images = []
                task_labels = []

                for label, characters in enumerate(sorted(os.listdir(language_path))):
                    characters_path = os.path.join(language_path, characters)

                    for id, filename in enumerate(sorted(os.listdir(characters_path))):
                        if id < 18 and type == 'train':
                            pass
                        elif (id == 18 or id == 19) and type == 'test':
                            pass
                        else:
                            continue

                        image_path = os.path.join(characters_path, filename)
                        image = Image.open(image_path, mode='r').convert('L')
                        image = transform(image)
                        task_images.append(image)
                        task_labels.append(label)

                    images.append(task_images)
                    labels.append(task_labels)

        self._num_classes = [len(np.unique(task_labels)) for task_labels in labels]

        # Prepare data loaders

        self.dataloader = []
        for task_images, task_labels in zip(images, labels):
            dataset = CustomDataset(data=task_images, labels=task_labels)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     drop_last=self.drop_last)
            self.dataloader.append(dataloader)


    def get_loader(self, task=None):
        if task is None:
            return MultiTaskDataLoader(self.dataloader)
        else:
            assert task in list(range(50)), 'Unknown loader: {}'.format(task)
            return SingleTaskDataLoader(self.dataloader[task])


    @property
    def image_size(self):
        return 105


    @property
    def num_channels(self):
        return 1


    @property
    def num_classes(self):
        return self._num_classes
