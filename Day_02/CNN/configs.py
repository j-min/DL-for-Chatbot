import argparse
import pprint
from pathlib import Path
from torch import optim
from torch import nn

project_dir = Path(__file__).resolve().parent

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

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    #================ Train ==============#
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=20)

    #================ Model ==============#
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--n_channel_per_window', type=int, default=2)
    parser.add_argument('--label_size', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--dropout', type=float, default=0.5)

    #================ Path  ==============#
    parser.add_argument('--save_dir', type=str, default=project_dir.joinpath('log'))
    parser.add_argument('--data_dir', type=str, default=project_dir.joinpath('datasets'))

    #================ Misc. ==============#
    parser.add_argument('--log_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    #=============== Parse Arguments===============#
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)

    kwargs.update(optional_kwargs)

    return Config(**kwargs)




if __name__ == '__main__':
    config = get_config()
    print(config)
