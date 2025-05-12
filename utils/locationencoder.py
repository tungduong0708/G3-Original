from torch import nn
import torch
import numpy as np

from utils.pe.projection_rff import ProjectionRFF
from utils.pe.spherical_harmonics import SphericalHarmonics
from utils.pe.projection import Projection
from utils.nn.mlp import MLP
from utils.nn.rff_mlp import RFFMLP
from utils.nn.siren import SirenNet

def get_positional_encoding(positional_encoding_type, hparams):
    """
    Returns a positional encoding module based on the specified encoding type.
    
    Args:
        encoding_type (str): The type of positional encoding to use. Options are 'rff', 'siren', 'sh', 'capsule'.
        input_dim (int): The input dimension for the positional encoding.
        output_dim (int): The output dimension for the positional encoding.
        hparams: Additional arguments for specific encoding types.
        
    Returns:
        nn.Module: The positional encoding module.
    """
    if positional_encoding_type == "projectionrff":
        return ProjectionRFF(
            projection=hparams["projection"],
            sigma=hparams["sigma"],
            hparams=hparams
        )
    elif positional_encoding_type == "projection":
        return Projection(
            projection=hparams["projection"],
            hparams=hparams
        )
    elif positional_encoding_type == "sh":
        return SphericalHarmonics(
            legendre_polys=hparams["legendre_polys"],
            harmonics_calculation=hparams["harmonics_calculation"],
            hparams=hparams
        )
    else:
        raise ValueError(f"Unsupported encoding type: {positional_encoding_type}")
    
def get_neural_network(neural_network_type, input_dim, hparams=None):
    """
    Returns a neural network module based on the specified network type.
    
    Args:
        neural_network_type (str): The type of neural network to use. Options are 'siren'.
        input_dim (int): The input dimension for the neural network.
        output_dim (int): The output dimension for the neural network.
        hparams: Additional arguments for specific network types.
        
    Returns:
        nn.Module: The neural network module.
    """
    if neural_network_type == "siren":
        return SirenNet(
            input_dim=input_dim,
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            hparams=hparams
        )
    elif neural_network_type == "mlp":
        return MLP(
            input_dim=input_dim, 
            hidden_dim=hparams["hidden_dim"],
            hparams=hparams
        )
    elif neural_network_type == "rffmlp":
        return RFFMLP(
            input_dim=input_dim, 
            hidden_dim=hparams["hidden_dim"],
            sigma=hparams["sigma"],
            hparams=hparams
        )
    else:
        raise ValueError(f"Unsupported network type: {neural_network_type}")
    

class LocationEncoder(nn.Module):
    def __init__(self, positional_encoding_type="projectionrff", neural_network_type="siren", hparams=None):
        super().__init__()

        self.position_encoder = get_positional_encoding(
            positional_encoding_type, hparams
        )

        self.neural_network = nn.ModuleList([
            get_neural_network(
                neural_network_type,
                input_dim=dim,
                hparams=hparams
            ) for dim in self.position_encoder.embedding_dim
        ])

    def forward(self, x):
        embedding = self.position_encoder(x)
        print("SH output requires_grad:", embedding.requires_grad)
        
        if embedding.ndim == 2:
            # If the embedding is (batch, n), we need to add a dimension
            embedding = embedding.unsqueeze(0)

        location_features = torch.zeros(embedding.shape[1], 512).to('cuda')

        for nn, e in zip(self.neural_network, embedding):
            location_features += nn(e)

        return location_features