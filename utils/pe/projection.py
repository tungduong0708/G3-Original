import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pyproj import Proj, Transformer

SF = 66.50336

class Projection(nn.Module):
    def __init__(self, projection="ecef"):
        super(Projection, self).__init__()

        self.projection = projection.lower()

        proj_wgs84 = Proj('epsg:4326')

        if self.projection == "mercator":
            proj_target = Proj('epsg:3857')
            self.normalizer = 20037508.3427892
            self.embedding_dim =  [2]
        elif self.projection == "eep":
            proj_target = Proj('epsg:8857')
            self.normalizer = 180/SF 
            self.embedding_dim =  [2]
        elif self.projection == "ecef":
            proj_target = Proj('epsg:4978')
            self.normalizer = 6378137.0  # radius of Earth, not exact for ECEF but usable
            self.embedding_dim =  [3]
        else:
            raise ValueError(f"Unsupported projection: {self.projection}")

        self.transformer = Transformer.from_proj(proj_wgs84, proj_target, always_xy=True)

    def forward(self, input):
        lat = input[:, 0].float().detach().cpu().numpy()
        lon = input[:, 1].float().detach().cpu().numpy()
        # lon (batch), lat (batch)

        # Shape: (batch, 2) or (batch, 3) depending on projection
        if self.projection == "ecef":
            alt = np.zeros_like(lat)
            projected = self.transformer.transform(lon, lat, alt)
            location = list(zip(*projected))  # X, Y, Z
            location = torch.Tensor(location).to('cuda')
        else:
            projected = self.transformer.transform(lon, lat)
            location = [[y, x] for x, y in zip(*projected)]
            location = torch.Tensor(location).to('cuda')

        location = location / self.normalizer
        return location