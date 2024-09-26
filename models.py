import torch
from torch import nn
from torch.nn import init


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, change_init_weight=False):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.sequential = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )
        if change_init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.sequential:
            if isinstance(m, nn.Linear):
                if m.out_features == self.output_size:
                    init.normal_(m.weight[1:], mean=0, std=1e-2)
                    init.zeros_(m.bias)
                    with torch.no_grad():
                        m.weight[0].fill_(5 / self.hidden_size)
                else:
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
                    init.zeros_(m.bias)

    def forward(self, x):
        return self.sequential(x)
