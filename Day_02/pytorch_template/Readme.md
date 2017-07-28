# Minimal PyTorch project template for beginners

## solver.py

```
from models import Model

class Solver(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

    def build(self, is_train):
        self.model = Model(self.config, is_train)

        self.loss_fn = self.config.loss_fn(..)

        if is_train:
            self.optimizer = self.config.optimizer(..)

    def train(self):
        for batch_input in data_loader:
            batch_output = self.model(batch_input)
            loss = self.loss_fn(batch_input, batch_output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self):
        for batch_input in data_loader:
            batch_output = self.model(batch_input)
            loss = self.loss_fn(batch_input, batch_output)
```

## data_loader.py

```
import torch.utils.data as data

class Dataset(data.Dataset):
    ...

def collate_fn(...):
    ...

def get_loader(is_train=True, batch_size, ...):

    dataset = Dataset(...)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn)

    return data_loader
```

## model.py

```
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, config):
        ...
    def forward(self, input):
        ...
```


## configs.py

```
import argparse

class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                setattr(self, key, value)    

def parse_args():
    parser = argparse.ArgumentParser()

    #================ Train ==============#
    parser.add_argument('--batch_size', type=int, default=20)
    ...

    #================ Model ==============#
    parser.add_argument('--hidden_size', type=int, default=100)
    ...

    #================ Path  ==============#
    parser.add_argument('--save_dir', type=str, default=os.path.join(base_dir, 'logdir'))

    #================ Misc. ==============#
    parser.add_argument('--print_every_epoch', type=int, default=1)
    ...

    #=============== Parse Arguments===============#
    kwargs = parser.parse_args()

    # Namespace => Dictionary
    kwargs = vars(kwargs)

    return kwargs

def get_config():
    kwargs = parse_args()
    return Config(**kwargs)
```

## train.py

```
from solver import Solver
from data_loader import get_loader
from configs import get_config

if __name__ == '__main__':
    config = get_config()

    data_loader = get_loader(
        is_train=True
        batch_size=config.batch_size)

    solver = Solver(config, data_loader)
    solver.build(is_train=True)
    solver.train()
```

## eval.py

```
from solver import Solver
from data_loader import get_loader
from configs import get_config

if __name__ == '__main__':
    config = get_config()

    data_loader = get_loader(
        is_train=False
        batch_size=config.batch_size)

    solver = Solver(config, data_loader)
    solver.build(is_train=False)
    solver.eval()
```
