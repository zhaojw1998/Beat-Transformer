import torch 
from torch import nn 
from music_transformer import TransformerEncoderLayer
import torch.nn.functional as F 
import math
import sys

from tcn import residual_block


class DemixedTCN(nn.Module):
    def __init__(self, attn_len=5, instr=5, ntoken=2, dmodel=128, nhead=2, d_hid=512, nlayers=9, norm_first=True, dropout=.1):
        super(DemixedTCN, self).__init__()
        self.nhead = nhead
        self.nlayers = nlayers
        self.attn_len = attn_len
        self.head_dim = dmodel // nhead
        self.dmodel = dmodel
        assert self.head_dim * nhead == dmodel, "embed_dim must be divisible by num_heads"

        #self.Er = nn.Parameter(torch.randn(nhead, self.head_dim, attn_len))
        #instr_pe = self.generate_instr_pe(ninstr, dmodel)
        #self.register_buffer('instr_pe', instr_pe)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 3), stride=1, padding=(2, 0))#126
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#42
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))#31
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#10
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(3, 6), stride=1, padding=(1, 0))#5
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#1
        self.dropout3 = nn.Dropout(p=dropout)

        self.head_er = nn.Parameter(torch.randn(nhead, self.head_dim, 1))
        
        self.Transformer_layers = nn.ModuleDict({})
        for idx in range(nlayers):
            #self.Transformer_layers[f'time_attention_{idx}'] = DilatedTransformerLayer(dmodel, nhead, d_hid, dropout, Er_provided=False, attn_len=attn_len, norm_first=norm_first)
            self.Transformer_layers[f'time_attention_{idx}'] = residual_block(2**idx, dmodel, dmodel, attn_len, dropout)
            
            
            if (idx >= 3) and (idx <= 5):
                self.Transformer_layers[f'instr_attention_{idx}'] = TransformerEncoderLayer(dmodel, nhead, d_hid, dropout, Er_provided=False, max_len=instr, norm_first=norm_first)
            
        self.out_linear = nn.Linear(dmodel, ntoken)

        self.dropout_t = nn.Dropout(p=.5)
        self.out_linear_t = nn.Linear(dmodel, 300)
        

    def forward(self, x):
        #x: (batch, instr, time, dmodel), FloatTensor
        #batch, time, dmodel = x.shape
        batch, instr, time, melbin = x.shape
        x = x.reshape(-1, 1, time, melbin)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)    #(batch*instr, channel, time, 1)

        x = x.reshape(-1, self.dmodel, time).transpose(1, 2)    #(batch*instr, time, channel=dmodel)
        t = []

        for layer in range(self.nlayers):
            x = x.transpose(-1, -2)
            x, skip = self.Transformer_layers[f'time_attention_{layer}'](x)

            x = x.transpose(-1, -2)
            skip = skip.transpose(-1, -2).reshape(batch, instr, time, self.dmodel)
            #skip = skip.reshape(batch, instr, time, self.dmodel)
            t.append(skip.mean(1))
  
            if (layer >= 3) and (layer <= 5):
                x = x.reshape(batch, instr, time, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, instr, self.dmodel)

                #x = self.Transformer_layers[f'instr_attention_{layer}'](x, layer=layer)
                x = self.Transformer_layers[f'instr_attention_{layer}'](x)

                x = x.reshape(batch, time, instr, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, time, self.dmodel)
            
        x = torch.relu(x)
        x = x.reshape(batch, instr, time, self.dmodel)
        x = x.mean(1)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)

        return x, t

    def inference(self, x):
        #x: (batch, instr, time, dmodel), FloatTensor
        #batch, time, dmodel = x.shape
        batch, instr, time, melbin = x.shape
        x = x.reshape(-1, 1, time, melbin)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)    #(batch*instr, channel, time, 1)

        x = x.reshape(-1, self.dmodel, time).transpose(1, 2)    #(batch*instr, time, channel=dmodel)
        t = []

        attn = [torch.eye(time, device=x.device).repeat(batch, self.nhead, 1, 1)]

        for layer in range(self.nlayers):
            x, skip, layer_attn = self.Transformer_layers[f'time_attention_{layer}'].inference(x, layer=layer, head_er=self.head_er)
            skip = skip.reshape(batch, instr, time, self.dmodel)
            t.append(skip.mean(1))

            attn.append(torch.matmul(attn[-1], layer_attn.transpose(-2, -1)))
  
            if (layer >= 3) and (layer <= 5):
                x = x.reshape(batch, instr, time, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, instr, self.dmodel)

                #x = self.Transformer_layers[f'instr_attention_{layer}'](x, layer=layer)
                x = self.Transformer_layers[f'instr_attention_{layer}'](x)

                x = x.reshape(batch, time, instr, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, time, self.dmodel)
            
        x = torch.relu(x)
        x = x.reshape(batch, instr, time, self.dmodel)
        x = x.mean(1)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)

        return x, t, attn




if __name__ == '__main__':
    from spectrogram_dataset import audioDataset
    from torch.utils.data import DataLoader

    DEVICE = 'cpu'
    model = DemixedTCN(attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8, d_hid=1024, nlayers=9, norm_first=True, dropout=.1)
    model.to(DEVICE)
    model.eval()

    for name, param in model.state_dict().items():
        print(name, param.shape)
    # name: str
    # param: Tensor

    total = sum([param.nelement() for param in model.parameters()])
    print(total)
    #print("Number of parameter: %.2fM" % (total/1e6))
