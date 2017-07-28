import argparse
import os
from torch import optim
from torch import nn

base_dir = os.path.dirname(os.path.abspath(__file__))

optimizer_dict = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}


class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                setattr(self, key, value)

        self.loss_fn = nn.CrossEntropyLoss


def parse_args():
    parser = argparse.ArgumentParser()

    #================ Train ==============#
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=20)

    #================ Model ==============#
    parser.add_argument('--x_size', type=int, default=784)
    parser.add_argument('--h1_size', type=int, default=500)
    parser.add_argument('--h2_size', type=int, default=200)
    parser.add_argument('--label_size', type=int, default=10)

    #================ Path  ==============#
    parser.add_argument('--save_dir', type=str, default=os.path.join(base_dir, 'log'))
    parser.add_argument('--data_dir', type=str, default=os.path.join(base_dir, 'datasets'))

    #================ Misc. ==============#
    parser.add_argument('--log_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    #=============== Parse Arguments===============#
    kwargs = parser.parse_args()

    # Namespace => Dictionary
    kwargs = vars(kwargs)

    return kwargs


def get_config():
    kwargs = parse_args()
    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    print(config)
