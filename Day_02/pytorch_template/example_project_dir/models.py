from torch import nn


class NNClassifier(nn.Module):
    def __init__(self, config):
        super(NNClassifier, self).__init__()

        self.config = config

        self.layer_1 = nn.Linear(config.x_size, config.h1_size)
        self.layer_2 = nn.Linear(config.h1_size, config.h2_size)
        self.layer_3 = nn.Linear(config.h2_size, config.label_size)

        self.lrelu = nn.LeakyReLU()

        self.softmax = nn.Softmax()

        self.net = nn.Sequential(
            self.layer_1,  # 784 => 200
            self.lrelu,
            self.layer_2,  # 200 => 50
            self.lrelu,
            self.layer_3,  # 50 => 10
            self.softmax,
        )

    def forward(self, x):
        # [batch_size, 784] => [batch_size, 1]
        return self.net(x)
