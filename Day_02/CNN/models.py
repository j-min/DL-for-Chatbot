import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.conv = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=config.n_kernels,
                kernel_size=(3, config.hidden_size)),

            nn.Conv2d(
                in_channels=1,
                out_channels=config.n_kernels,
                kernel_size=(4, config.hidden_size)),

            nn.Conv2d(
                in_channels=1,
                out_channels=config.n_kernels,
                kernel_size=(5, config.hidden_size))
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(3 * config.n_kernels, config.label_size)

    def forward(self, x):
        # x: [batch_size, max_seq_len]

        # [batch_size, max_seq_len, hidden_size]
        x = self.embedding(x)

        # [batch_size, 1, max_seq_len, hidden_size]
        x = x.unsqueeze(1)

        list_after_kernel = []
        for conv in self.conv:
            # [batch_size, n_kernels, _, 1]
            x_ = F.relu(conv(x))

            # [batch_size, n_kernels, _]
            x_ = x_.squeeze(3)

            list_after_kernel.append(x_)

        list_kernel_max = []
        for x in list_after_kernel:

            # [batch_size, n_kernels, 1]
            x_ = F.max_pool1d(x, x.size(2))

            # [batch_size, n_kernels]
            x_ = x_.squeeze(2)
            list_kernel_max.append(x_)

        # [batch_size, 3 x n_kernels]
        out = torch.cat(list_kernel_max, 1)

        out = self.dropout(out)

        # [batch_size, label_size]
        logit = self.fc(out)

        return logit
