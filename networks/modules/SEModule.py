import argparse
import torch
from torch import nn
from torch.nn import functional as F

class SEModule(nn.Module):
    """Dropout + Channel Attention
    """
    def __init__(self, in_channels, out_channels, gating_reduction, drop_rate):
        super(SEModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.adaptAvgPool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.gate = nn.Linear(out_channels, out_channels//gating_reduction)
        self.fc2 = nn.Linear(out_channels//gating_reduction, out_channels)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        attn_layer = self.adaptAvgPool(x).squeeze(-1)
        attn_layer = F.relu(self.fc1(attn_layer))
        attn_layer = F.relu(self.gate(attn_layer))
        attn_layer = F.relu(self.fc2(attn_layer))
        # if(self.gating_last_bn):
        #     attn_layer = self.last_bn(attn_layer)
        out = x*attn_layer.unsqueeze(-1)
        return out

if __name__ == "__main__":
    model = SEModule(4, 8, gating_reduction= 2, drop_rate=0.2)
    model.to('cuda')
    input = torch.randn((2, 4, 10))
    input=input.to('cuda')
    output = model(input)
    print(output.shape)