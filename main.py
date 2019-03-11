import argparse
import yaml
from namedtuple import TaskInfo, ModelConfigs, LayerArguments
from data_loader import CIFAR10Loader, CIFAR100Loader, OmniglotLoader
from agent import SingleTaskModel, MultiTaskModel


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--data', type=int, default=1, help='0: CIFAR-10\n'
                                                            '1: CIFAR-100\n'
                                                            '2: Omniglot')
    parser.add_argument('--task', type=int, default=None)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--path', type=str, default='saved_models/default/')

    parser.add_argument('--save_history', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    configs = _load_configs()
    architecture = _load_architecture()

    if args.data == 0:
        train_data = CIFAR10Loader(batch_size=configs.batch_size, type='train', drop_last=True)
        test_data = CIFAR10Loader(batch_size=configs.batch_size, type='test', drop_last=False)
    elif args.data == 1:
        train_data = CIFAR100Loader(batch_size=configs.batch_size, type='train', drop_last=True)
        test_data = CIFAR100Loader(batch_size=configs.batch_size, type='test', drop_last=False)
    elif args.data == 2:
        train_data = OmniglotLoader(batch_size=configs.batch_size, type='train', drop_last=True)
        test_data = OmniglotLoader(batch_size=configs.batch_size, type='test', drop_last=False)
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_tasks = len(train_data.num_classes)

    if args.task is None:
        task_info = TaskInfo(image_size=train_data.image_size,
                             num_classes=train_data.num_classes,
                             num_channels=train_data.num_channels,
                             num_tasks=num_tasks
                             )

        agent = MultiTaskModel(architecture=architecture, task_info=task_info)

    else:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)

        task_info = TaskInfo(image_size=train_data.image_size,
                             num_classes=train_data.num_classes[args.task],
                             num_channels=train_data.num_channels,
                             num_tasks=1
                             )

        train_data = train_data.get_loader(args.task)
        test_data = test_data.get_loader(args.task)

        agent = SingleTaskModel(architecture=architecture, task_info=task_info)

    if args.load:
        agent.load(args.path)

    agent.train(train_data=train_data,
                test_data=test_data,
                configs=configs,
                save_history=args.save_history,
                path=args.path,
                verbose=args.verbose
                )

    if args.save:
        agent.save(args.path)


def evaluate(args):
    configs = _load_configs()
    architecture = _load_architecture()

    if args.data == 0:
        data = CIFAR10Loader(batch_size=configs.batch_size, type='test', drop_last=False)
    elif args.data == 1:
        data = CIFAR100Loader(batch_size=configs.batch_size, type='test', drop_last=False)
    elif args.data == 2:
        data = OmniglotLoader(batch_size=configs.batch_size, type='test', drop_last=False)
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_tasks = len(data.num_classes)

    if args.task is None:
        task_info = TaskInfo(image_size=data.image_size,
                             num_classes=data.num_classes,
                             num_channels=data.num_channels,
                             num_tasks=num_tasks
                             )

        agent = MultiTaskModel(architecture=architecture, task_info=task_info)

    else:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)

        task_info = TaskInfo(image_size=data.image_size,
                             num_classes=data.num_classes[args.task],
                             num_channels=data.num_channels,
                             num_tasks=1
                             )
        data = data.get_loader(args.task)

        agent = SingleTaskModel(architecture=architecture, task_info=task_info)

    agent.load(args.path)
    accuracy = agent.eval(data)

    print('Accuracy: {}'.format(accuracy))


def _load_configs():
    with open('configs/train.yaml', 'r') as f:
        configs = yaml.load(f)

    return ModelConfigs(**configs)


def _load_architecture():
    with open('configs/architecture.yaml', 'r') as f:
        configs = yaml.load(f)

    return [LayerArguments(**config) for config in configs]


def main():
    args = parse_args()
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        print('No flag is assigned. Please assign either \'--train\' or \'--eval\'.')


if __name__ == '__main__':
    main()