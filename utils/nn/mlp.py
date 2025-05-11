from torch import nn

class MLP(nn.Module):
    """Multi-layer perceptron (MLP) with batch normalization and ReLU activation."""

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super(MLP, self).__init__()
        self.capsule = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x