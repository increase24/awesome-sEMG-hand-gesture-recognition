# https://www.sciencedirect.com/science/article/abs/pii/S0167865517304439
# multi-stream convolutional neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCNN(nn.Module):
    def __init__(self, cfg):
        super(MSCNN, self).__init__()
        self.num_stream = cfg.num_stream
        self.layers = nn.ModuleList()
        for idx in range(cfg.num_stream):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(1, cfg.filter1, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(cfg.filter1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(cfg.filter1, cfg.filter2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(cfg.filter2),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(cfg.filter2, cfg.filter3, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm1d(cfg.filter3),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(cfg.filter3, cfg.filter4, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm1d(cfg.filter4),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=cfg.dropout_rate)
                )
            )

        self.fc1 = nn.Sequential(
            nn.Linear(cfg.featDim_concat, cfg.fc1),
            nn.BatchNorm1d(cfg.fc1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(cfg.fc1, cfg.fc2),
            nn.BatchNorm1d(cfg.fc2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout_rate)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(cfg.fc2, cfg.fc3),
            nn.BatchNorm1d(cfg.fc3),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(cfg.fc3, cfg.num_classes)


    def forward(self, x):
        out_streams = []
        for idx in range(self.num_stream):
            out_streams.append(self.layers[idx](x[:,:,idx,:]).view(x.shape[0], -1))
        output = torch.cat(out_streams, dim=1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.output_layer(output)
        return output

if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    print('test MSCNNN on DB1')
    config = CN.load_cfg(open('./cfgs/MSCNN_db1.yaml'))
    model = MSCNN(config)
    model.eval()
    input = torch.randn((8,1,10,20))
    output = model(input)
    print(output.shape)
    print('test MSCNNN on DB2')
    config = CN.load_cfg(open('./cfgs/MSCNN_db2.yaml'))
    model = MSCNN(config)
    model.eval()
    input = torch.randn((8,1,12,200))
    output = model(input)
    print(output.shape)

        
