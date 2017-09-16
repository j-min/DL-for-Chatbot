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
                out_channels=config.n_channel_per_window,
                kernel_size=(3, config.hidden_size)),

            nn.Conv2d(
                in_channels=1,
                out_channels=config.n_channel_per_window,
                kernel_size=(4, config.hidden_size)),

            nn.Conv2d(
                in_channels=1,
                out_channels=config.n_channel_per_window,
                kernel_size=(5, config.hidden_size))
        ])

        n_total_channels = len(self.conv) * config.n_channel_per_window

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(n_total_channels, config.label_size)

    def forward(self, x):
        """
        Args:
            x: [batch_size, max_seq_len]
        Return:
            logit: [batch_size, label_size]
        """

        # [batch_size, max_seq_len, hidden_size]
        x = self.embedding(x)

        # [batch_size, 1, max_seq_len, hidden_size]
        x = x.unsqueeze(1)

        # Apply Convolution filter followed by Max-pool
        out_list = []
        for conv in self.conv:

            ########## Convolution #########

            # [batch_size, n_kernels, _, 1]
            x_ = F.relu(conv(x))

            # [batch_size, n_kernels, _]
            x_ = x_.squeeze(3)

            ########## Max-pool #########

            # [batch_size, n_kernels, 1]
            x_ = F.max_pool1d(x_, x_.size(2))

            # [batch_size, n_kernels]
            x_ = x_.squeeze(2)

            out_list.append(x_)

        # [batch_size, 3 x n_kernels]
        out = torch.cat(out_list, 1)

        ######## Dropout ########
        out = self.dropout(out)

        # [batch_size, label_size]
        logit = self.fc(out)

        return logit
