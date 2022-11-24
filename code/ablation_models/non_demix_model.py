import torch 
from torch import nn 
from DilatedTransformerLayer import DilatedTransformerLayer


class DilatedTransformerModel(nn.Module):
    def __init__(self, attn_len=5, ntoken=2, dmodel=128, nhead=2, d_hid=512, nlayers=9, norm_first=True, dropout=.1):
        super(DilatedTransformerModel, self).__init__()
        self.nhead = nhead
        self.nlayers = nlayers
        self.attn_len = attn_len
        self.head_dim = dmodel // nhead
        assert self.head_dim * nhead == dmodel, "embed_dim must be divisible by num_heads"

        #self.Er = nn.Parameter(torch.randn(nlayers, nhead, self.head_dim, attn_len))

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 3), stride=1, padding=(2, 0))#126
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=(1, 0))#79
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#26
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))#31
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))#15
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#5
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(3, 6), stride=1, padding=(1, 0))#5
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(3, 3), stride=1, padding=(1, 0))#3
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#1
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.Transformer_layers = nn.ModuleDict({})
        for idx in range(nlayers):
            self.Transformer_layers[f'Transformer_layer_{idx}'] = DilatedTransformerLayer(dmodel, nhead, d_hid, dropout, Er_provided=False, attn_len=attn_len, norm_first=norm_first)

        self.out_linear = nn.Linear(dmodel, ntoken)

        self.dropout_t = nn.Dropout(p=.5)
        self.out_linear_t = nn.Linear(dmodel, 300)
        

    def forward(self, x):
        #x: (batch, time, dmodel), FloatTensor
        x = x.unsqueeze(1)  #(batch, channel, time, dmodel)
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
        x = self.dropout3(x)    #(batch, channel, time, 1)
        x = x.transpose(1, 3).squeeze(1).contiguous()    #(batch, time, channel=dmodel)
        
        batch, time, dmodel = x.shape
        t = []
        for layer in range(self.nlayers):
            x, skip = self.Transformer_layers[f'Transformer_layer_{layer}'](x, layer=layer)
            t.append(skip)

        x = torch.relu(x)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)

        return x, t

    def inference(self, x):
        #x: (batch, time, dmodel), FloatTensor
        x = x.unsqueeze(1)  #(batch, channel, time, dmodel)
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
        x = self.dropout3(x)    #(batch, channel, time, 1)
        x = x.transpose(1, 3).squeeze(1).contiguous()    #(batch, time, channel=dmodel)
        
        batch, time, dmodel = x.shape
        t = []
        attn = [torch.eye(time, device=x.device).repeat(batch, self.nhead, 1, 1)]
        for layer in range(self.nlayers):
            x, skip, layer_attn = self.Transformer_layers[f'Transformer_layer_{layer}'].inference(x, layer=layer)
            t.append(skip)
            attn.append(torch.matmul(attn[-1], layer_attn.transpose(-2, -1)))


        x = torch.relu(x)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)

        return x, t, attn
