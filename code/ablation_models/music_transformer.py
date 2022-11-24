import math
from numpy import transpose
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm
from torch.nn import Transformer
import copy


class MultiheadAttentionwithRelativePositionalEmbedding(nn.Module):
    def __init__(self, dmodel, num_heads, dropout=0, Er_provided=False, max_len=3):
        super(MultiheadAttentionwithRelativePositionalEmbedding, self).__init__()
        self.L = 2 * max_len - 1
        self.num_heads = num_heads
        self.max_len = max_len
        self.head_dim = dmodel // num_heads
        assert self.head_dim * num_heads == dmodel, "embed_dim must be divisible by num_heads"

        self.key = nn.Linear(dmodel, dmodel)
        self.value = nn.Linear(dmodel, dmodel)
        self.query = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)
        self.Er_provided = Er_provided
        self.num_heads = num_heads
        
        if not Er_provided:
            self.Er = nn.Parameter(torch.randn(num_heads, self.L, self.head_dim))

    def forward(self, query, key, value, Er=None, layer=0, attn_mask=None):
        #x: (batch, len, dmodel)
        #Srel: (num_head, tgt_len, src_len)
        #attn_mask:  (batch, num_head, tgt_len, src_len): float tensor
        bs, tgt_len, d_model = query.shape
        _, src_len, _ = key.shape

        q = self.query(query).reshape(bs, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  #(batch, num_head, tgt_len, head_dim)
        k = self.key(key).reshape(bs, src_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  #(batch, num_head, head_dim, src_len)
        v = self.value(value).reshape(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)  #(batch, num_head, src_len, head_dim)

        Er_t = torch.zeros(self.num_heads, 2*src_len-1, self.head_dim, device=query.device)
        dilation_len = min(1 + (src_len-1)//(2**layer), self.max_len)
        if not self.Er_provided:
            Er_t[:, [src_len-1 + i*(2**layer) for i in range(-dilation_len+1, dilation_len)], :] = self.Er[:, self.max_len-dilation_len: self.max_len+dilation_len-1, :]
        else:
            Er_t[:, [src_len-1 + i*(2**layer) for i in range(-dilation_len+1, dilation_len)], :] = Er[:, self.max_len-dilation_len: self.max_len+dilation_len-1, :]
        Er_t = Er_t.transpose(-2, -1)   #(num_head, head_dim, src_L)

        QEr = torch.matmul(q, Er_t) #(num_head, num_head, tgt_len, src_L)
        #print(QEr[0, 0])
        Srel = self.skew(QEr, src_len) #(num_head, num_head, tgt_len, src_len)
        #print('Srel', Srel[1, 1])

        attn = (torch.matmul(q, k) + Srel) / math.sqrt(self.head_dim) #(batch, num_head, tgt_len, src_len)
        
        if attn_mask is not None:
            #print(attn.shape, attn_mask.shape)
            attn += attn_mask
            #for i in range(attn.shape[0]):
            #    print(attn_mask[i, 0])
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) #(batch, num_head, tgt_len, head_dim)
        out = out.transpose(1, 2).reshape(bs, tgt_len, d_model) #(batch, tgt_len, d_model)

        return self.dropout(out), attn
        
    def skew(self, QEr, src_len):
        #QEr: (batch, num_heads, tgt_len, src_L)
        bs, num_heads, tgt_len, src_L = QEr.shape
        QEr = F.pad(QEr, (0, 1))    #(batch, num_heads, tgt_len, src_L+1)
        QEr = QEr.reshape(bs, num_heads, -1)   #(batch, num_heads, tgt_len*(src_L+1))
        QEr = F.pad(QEr, (0, src_L-tgt_len))    #(batch, num_heads, (tgt_len+1)*src_L)
        QEr = QEr.reshape(bs, num_heads, tgt_len+1, src_L)
        QEr = QEr[:, :, :tgt_len, -src_len:]    #(batch, num_heads, tgt_len, src_len)
        return QEr


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, Er_provided=False, max_len=3, layer_norm_eps=1e-5, norm_first=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttentionwithRelativePositionalEmbedding(d_model, nhead, dropout, Er_provided, max_len)
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

    def forward(self, x, Er=None, layer=0, src_mask=None):
        #x: (batch, len, dmodel)
        #Er: (num_head, tgt_len, src_len)
        #key_padding_mask: (batch, num_head, tgt_len, src_len), bool tensor
        #attn_mask:  (batch, num_head, tgt_len, src_len): float tensor
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), Er, layer, src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, Er, layer, src_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, Er=None, layer=0, attn_mask=None):
        x = self.self_attn(x, x, x, Er, layer, attn_mask=attn_mask)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



def generate_dilation_self_attention_mask(batch, num_head, seq_len, max_len, layer):
        attn_mask = torch.eye(seq_len).repeat(batch, num_head, 1, 1)
        mask_temp = torch.eye(seq_len).repeat(batch, num_head, 1, 1)
        for i in range(1, max_len):
            attn_mask[:, :, : -i*(2**layer), i*(2**layer):] += mask_temp[:, :, i*(2**layer):, i*(2**layer):]
            attn_mask[:, :, i*(2**layer):, : -i*(2**layer)] += mask_temp[:, :, i*(2**layer):, i*(2**layer):]
        attn_mask = (1-attn_mask).masked_fill((attn_mask == 0), -float('inf'))
        return attn_mask




if __name__ == '__main__':
    MAX_LEN=3
    LAYER=0

    model = MultiheadAttentionwithRelativePositionalEmbedding(dmodel=12, num_heads=6, max_len=MAX_LEN)

    x = torch.ones(3, 8, 12)

    attn_mask = generate_dilation_self_attention_mask(3, 6, 8, MAX_LEN, LAYER)
    #print(attn_mask[1, 1, :, :].numpy())

    output, attn = model(x, x, x, attn_mask=attn_mask, layer=LAYER)
    #print(attn[1, 1])