from torch import nn
import torch
import numpy as np

from utils.pe.projection_rff import ProjectionRFF
from utils.pe.spherical_harmonics import SphericalHarmonics
from utils.pe.projection import Projection
from utils.nn.mlp import MLP
from utils.nn.rff_mlp import RFFMLP
from utils.nn.siren import SirenNet

def get_positional_encoding(positional_encoding_type, **kwargs):
    """
    Returns a positional encoding module based on the specified encoding type.
    
    Args:
        encoding_type (str): The type of positional encoding to use. Options are 'rff', 'siren', 'sh', 'capsule'.
        input_dim (int): The input dimension for the positional encoding.
        output_dim (int): The output dimension for the positional encoding.
        **kwargs: Additional arguments for specific encoding types.
        
    Returns:
        nn.Module: The positional encoding module.
    """
    if positional_encoding_type == "projectionrff":
        return ProjectionRFF(**kwargs)
    elif positional_encoding_type == "projection":
        return Projection(**kwargs)
    elif positional_encoding_type == "sh":
        return SphericalHarmonics(**kwargs)
    else:
        raise ValueError(f"Unsupported encoding type: {positional_encoding_type}")
    
def get_neural_network(neural_network_type, input_dim, **kwargs):
    """
    Returns a neural network module based on the specified network type.
    
    Args:
        neural_network_type (str): The type of neural network to use. Options are 'siren'.
        input_dim (int): The input dimension for the neural network.
        output_dim (int): The output dimension for the neural network.
        **kwargs: Additional arguments for specific network types.
        
    Returns:
        nn.Module: The neural network module.
    """
    if neural_network_type == "siren":
        return SirenNet(
            input_dim=input_dim,
            **kwargs
        )
    elif neural_network_type == "mlp":
        return MLP(
            input_dim=input_dim, 
            **kwargs
        )
    elif neural_network_type == "rffmlp":
        return RFFMLP(
            input_dim=input_dim, 
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported network type: {neural_network_type}")
    

class LocationEncoder(nn.Module):
    def __init__(self, position_encoding_type="projectionrff", neural_network_type="siren", **kwargs):
        super().__init__()

        self.position_encoder = get_positional_encoding(
            position_encoding_type, **kwargs
        )

        self.neural_network = nn.ModuleList([
            get_neural_network(
                neural_network_type,
                input_dim=dim
            ) for dim in self.position_encoder.embedding_dim
        ])

    def forward(self, x):
        embedding = self.position_encoder(x)
        
        if embedding.ndim == 2:
            # If the embedding is (batch, n), we need to add a dimension
            embedding = embedding.unsqueeze(0)

        location_features = torch.zeros(embedding.shape[1], 512).to('cuda')

        for nn, e in zip(self.neural_network, embedding):
            location_features += nn(e)

        return location_features