from torch import nn
import torch
from ..rff.layers import GaussianEncoding

class LocationEncoderCapsule(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=1024, output_dim=512, sigma=2**0):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=input_dim, encoded_size=int(output_dim/2))
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(output_dim, hidden_dim),
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
    
class RFFMLP(nn.Module):
    """Multi-layer perceptron (MLP) with batch normalization and ReLU activation."""
    def __init__(self, input_dim=2, hidden_dim=1024, output_dim=512, sigma=[2**0, 2**4, 2**8]):
        super(RFFMLP, self).__init__()
        self.num_hierarchies = len(sigma)

        for i, s in enumerate(sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, sigma=s))

    def forward(self, input):
        location_features = torch.zeros(input.shape[0], 512).to('cuda')

        for i in range(self.num_hierarchies):
            location_features += self._modules['LocEnc' + str(i)](input)
        return location_features