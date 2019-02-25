import yaml
from collections import namedtuple


# Named tuples for configurations

with open('configs/train.yaml', 'r') as f:
    _configs = yaml.load(f)

ModelConfigs = namedtuple('ModelConfigs', _configs.keys())

with open('configs/architecture.yaml', 'r') as f:
    _configs = yaml.load(f)

LayerArguments = namedtuple('LayerArguments', _configs[0].keys())


# Others

TaskInfo = namedtuple('TaskInfo', ['image_size', 'num_classes', 'num_channels', 'num_tasks'])
