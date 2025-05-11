import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, input_dim, output_dim, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None, dropout = False):
        super().__init__()
        self.input_dim = input_dim
        self.is_first = is_first
        self.output_dim = output_dim
        self.dropout = dropout

        weight = torch.zeros(output_dim, input_dim)
        bias = torch.zeros(output_dim) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.input_dim

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        out = self.activation(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim = 1024, output_dim = 512, num_layers = 3, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, degreeinput = False, dropout = False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_input_dim = input_dim if is_first else hidden_dim

            self.layers.append(Siren(
                input_dim = layer_input_dim,
                output_dim = hidden_dim,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(input_dim = hidden_dim, output_dim = output_dim, w0 = w0, use_bias = use_bias, activation = final_activation, dropout = False)

    def forward(self, x, mods = None):

        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)