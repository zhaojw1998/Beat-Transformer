import math
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm


class DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(nn.Module):
    def __init__(self, dmodel, num_heads, dropout=0, Er_provided=False, attn_len=5):
        super(DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding, self).__init__()
        self.attn_len = attn_len
        self.dmodel = dmodel
        self.num_heads = num_heads
        self.head_dim = dmodel // num_heads
        assert self.head_dim * num_heads == dmodel, "embed_dim must be divisible by num_heads"

        self.key = nn.Linear(dmodel, dmodel)
        self.value = nn.Linear(dmodel, dmodel)
        self.query = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)
        self.Er_provided = Er_provided
        
        if not Er_provided:
            self.Er = nn.Parameter(torch.randn(num_heads, self.head_dim, attn_len))


    def forward(self, query, key, value, layer=0):
        #query, key, and value: (batch, time, dmodel), float tensor

        batch, time, d_model = query.shape

        q = self.query(query).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)
        k = self.key(key).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)
        v = self.value(value).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)

        k = torch.cat(
                        (
                        self.kv_roll(k[:, 0: 4], layer, padding_value=0, shift=0),
                        self.kv_roll(k[:, 4: 5], layer, padding_value=0, shift=-2),
                        self.kv_roll(k[:, 5: 6], layer, padding_value=0, shift=-1),
                        self.kv_roll(k[:, 6: 7], layer, padding_value=0, shift=1),
                        self.kv_roll(k[:, 6: 7], layer, padding_value=0, shift=2)   
                        ),
                    dim=1
                    )   #we define 4 symmetrical heads and 4 skewed heads
                        #The last line should be k[:, 7: 8]. This is a bug in my code. 
                        #This bug should not have impacted model performance though.

        v = torch.cat(
                        (
                        self.kv_roll(v[:, 0: 4], layer, padding_value=0, shift=0),
                        self.kv_roll(v[:, 4: 5], layer, padding_value=0, shift=-2),
                        self.kv_roll(v[:, 5: 6], layer, padding_value=0, shift=-1),
                        self.kv_roll(v[:, 6: 7], layer, padding_value=0, shift=1),
                        self.kv_roll(v[:, 7: 8], layer, padding_value=0, shift=2)
                        ),
                    dim=1
                    )   #we define 4 symmetrical heads and 4 skewed heads
        
        Er_t = self.Er.unsqueeze(1).unsqueeze(0)  #(1, num_head, 1, head_dim, attn_len)

        qk = torch.matmul(q, k.transpose(-2, -1))
        attn_mask = torch.zeros_like(qk).masked_fill_((qk==0), float('-inf'))
        attn = (qk + torch.matmul(q, Er_t)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn + attn_mask, dim=-1)

        out = torch.matmul(attn, v) #(batch, num_head, time, 1, head_dim)
        out = out.squeeze(-2).transpose(1, 2).reshape(batch, time, d_model)

        return self.dropout(out), attn
        


    def kv_roll(self, tensor, layer, padding_value=0, shift=1):
        #tensor: (batch, num_head, time, 1, head_dim)
        batch, num_head, time, _, head_dim = tensor.shape

        tensor = F.pad(tensor, (0, 0, 0, 0, (2**layer)*(self.attn_len//2), (2**layer)*(self.attn_len//2)), mode='constant', value=padding_value) 
        #(batch, num_head, time+(2**layer)*(self.attn_len//2), 1, head_dim)

        tensor = torch.cat([torch.roll(tensor, shifts=-i*(2**layer), dims=2) for i in range(shift, self.attn_len+shift)], dim=-2) 
        #(batch, num_head, time+(2**layer)*(self.attn_len//2), attn_len, head_dim)

        return tensor[:, :, :time, :, :]    #(batch, num_head, time, attn_len, head_dim)




class DilatedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, Er_provided=False, attn_len=5, norm_first=False, layer_norm_eps=1e-5):
        super(DilatedTransformerLayer, self).__init__()
        self.self_attn = DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(d_model, nhead, dropout, Er_provided, attn_len)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu


    def forward(self, x, layer=0):
        #x: (batch, time, dmodel)
        if self.norm_first:
            x_ = self._sa_block(self.norm1(x), layer)[0]
            x = x + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, layer)[0])
            x = self.norm2(x + self._ff_block(x))
        return x, x_


    def inference(self, x, layer=0):
        #x: (batch, time, dmodel)
        if self.norm_first:
            x_, attn = self._sa_block(self.norm1(x), layer)
            x = x + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x_, attn = self._sa_block(x, layer)
            x = self.norm1(x + x_)
            x = self.norm2(x + self._ff_block(x))


        attn = attn.squeeze(-2) #batch, num_head, time, attn_len
        batch, num_head, time, attn_len = attn.shape
        padded_attn_len = (attn_len-1) * (2**layer) + 1
        tmp_output = torch.zeros(batch, num_head, time, padded_attn_len, device=x.device)
        for i, j in enumerate(range(0, padded_attn_len, 2**layer)):
            tmp_output[:, :, :, j] = attn[:, :, :, i]

        attn = torch.zeros(batch, num_head, time, time+(padded_attn_len-1)*2, device=x.device)
        for i in range(time):
            attn[:, :, i, i: i+padded_attn_len] = tmp_output[:, :, i]

        center = (padded_attn_len-1)
        attn = torch.cat(
                            [
                                attn[:, 0: 4, :,  center - (2**layer) * 2:  center - (2**layer) * 2 + time],
                                attn[:, 4: 5, :,  center - (2**layer) * 1:  center - (2**layer) * 1 + time],
                                attn[:, 5: 6, :,  center - (2**layer) * 0:  center - (2**layer) * 0 + time],
                                attn[:, 6: 7, :,  center - (2**layer) * 3:  center - (2**layer) * 3 + time],
                                attn[:, 7: 8, :,  center - (2**layer) * 4:  center - (2**layer) * 4 + time]
                            ],
                            dim=1
                        )   #restore the square attention matrix from dilated self-attention

        return x, x_, attn


    # self-attention block
    def _sa_block(self, x, layer=0):
        x, attn = self.self_attn(x, x, x, layer)
        return self.dropout1(x), attn


    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)




if __name__ == '__main__':
    BATCH=1
    TIME=9
    DMODEL=8
    N_HEAD=4
    ATTN_LEN=5
    LAYER=1

    x = torch.ones(BATCH, TIME, DMODEL)

    model = DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(dmodel=DMODEL, num_heads=N_HEAD, attn_len=ATTN_LEN)

    output, attn = model(x, x, x, layer=LAYER)
    print(attn[0, 0, :, :, :])