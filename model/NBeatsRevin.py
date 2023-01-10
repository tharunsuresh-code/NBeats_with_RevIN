import torch
from model.RevIN import RevIN
from nbeats_pytorch.model import NBeatsNet

class NBeatsRevin(torch.nn.Module):
    def __init__(self, stack_types, forecast_length, backcast_length, hidden_layer_units, device='cpu'):
        super(NBeatsRevin, self).__init__()
        self.nbeats = NBeatsNet(
            device=device,
            stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
            forecast_length=forecast_length,
            backcast_length=backcast_length,
            hidden_layer_units=128,
        )
        self.revin_layer = RevIN(1)

    def forward(self, x_in):
        x = self.revin_layer(x_in, 'norm')
        b, x_out = self.nbeats(x)
        out = self.revin_layer(x_out, 'denorm')
        return b, out