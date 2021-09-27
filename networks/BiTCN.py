import os
os.sys.path.append('.')
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from networks.modules.SEModule import SEModule

class Chomp1d(nn.Module):
    def __init__(self, chomp_size) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:,:,:-self.chomp_size].contiguous()

class CasualConvBlock(nn.Module):
    def __init__(self, n_input, n_output, kernel_size, stride, padding, dilation, dropout_rate, skip_connect_type) -> None:
        super(CasualConvBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_input, n_output, kernel_size, stride = stride, \
            padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_output)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.block = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1
        )
        self.skip_connect_type = skip_connect_type
        if skip_connect_type == "Res-connect":
            self.skip_connect = nn.Conv1d(n_input, n_output, kernel_size=1) if n_input != n_output else None
        elif skip_connect_type == "SE-Module":
            self.skip_connect = SEModule(n_input, n_output, gating_reduction=2, drop_rate=dropout_rate)
        
    def forward(self, x):
        out = self.block(x)
        if self.skip_connect_type == "Res-connect":
            res = self.skip_connect(x) if self.skip_connect is not None else x
            out = out + res
        elif self.skip_connect_type == "SE-Module": 
            res = self.skip_connect(x)
            out = out * res     
        return out


class TCN(nn.Module):
    #def __init__(self, in_channels, num_filters, kernel_sizes, dilations, skip_types, dropout):
    def __init__(self, params, reverse):
        self.reverse = reverse
        in_channels = params.in_channels
        num_filters = params.num_filters
        kernel_sizes = params.kernel_sizes
        dilations = params.dilations
        skip_types = params.skip_types
        dropout = params.dropout_rate
        super(TCN, self).__init__()
        layers = []
        layers.append(nn.BatchNorm1d(in_channels))
        num_layers = len(num_filters)
        for i in range(num_layers):
            kernel_size = kernel_sizes[i]
            dilation_fac = dilations[i]
            in_channels = in_channels if i==0 else num_filters[i-1]
            out_channels = num_filters[i]
            skip_type = skip_types[i]
            layers.append(
                CasualConvBlock(in_channels, out_channels, kernel_size = kernel_size, stride = 1,padding=(kernel_size-1)*dilation_fac,\
                     dilation=dilation_fac, dropout_rate=dropout, skip_connect_type=skip_type)
            )
        self.network = nn.Sequential(*layers)
        self.adaptAvgPool = nn.AdaptiveAvgPool1d(output_size=1)
    
    def forward(self, x):
        x = x.squeeze(1)  #[Bs,1,C,W] -> [Bs,C,W]
        if self.reverse:
            x = torch.flip(x, dims=[2])
        out = self.network(x)
        #out = torch.sum(out, dim=1)
        #out = out.view([out.shape[0], -1])
        out = self.adaptAvgPool(out).squeeze(-1)
        return out

class UiTCN(nn.Module):
    def __init__(self, params, reverse) -> None:
        super(UiTCN, self).__init__()
        self.feat_net = TCN(params, reverse)
        self.output_layer = nn.Linear(params.num_filters[-1], params.num_classes)
    def forward(self, x):
        out = self.output_layer(self.feat_net(x))
        return out

class BiTCN(nn.Module):
    def __init__(self, params) -> None:
        super(BiTCN, self).__init__()
        self.tcn_positive = TCN(params, reverse = False)
        self.tcn_reverse = TCN(params, reverse = True)
        self.output_layer = nn.Linear(params.num_filters[-1] * 2, params.num_classes)
    
    def forward(self, x):
        out = torch.cat([self.tcn_positive(x), self.tcn_reverse(torch.flip(x, dims=[3]))], dim=1)
        out = self.output_layer(out)
        return out


if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    print('test MSCNNN on DB1')
    config = CN.load_cfg(open('./cfgs/BiTCN_db1.yaml'))
    model = BiTCN(config)
    input = torch.randn((8,1,10,20))
    output = model(input)
    print(output.shape)
    print('test MSCNNN on DB2')
    config = CN.load_cfg(open('./cfgs/BiTCN_db2.yaml'))
    model = BiTCN(config)
    input = torch.randn((8,1,12,200))
    output = model(input)
    print(output.shape)




