# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 12:35:52 2021

@author: Ranak Roy Chowdhury
"""

import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import sys
from layer_norm import LayerNorm

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class LNargs(object):
    
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = False

class LNargsDetach(object):
    
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = True
        self.std_detach = True
        
class LNargsDetachNotMean(object):
    
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True

def make_p_layer(layer, gamma=0.01):
    player = copy.deepcopy(layer)
    player.weight = torch.nn.Parameter(layer.weight+gamma*layer.weight.clamp(min=0))
    player.bias   = torch.nn.Parameter(layer.bias +gamma*layer.bias.clamp(min=0))
    return player

class MultiheadAttention(nn.Module):
    
    def __init__(self, d_model, nheads, detach_mode, dropout):
        super().__init__()

        self.nheads = nheads
        self.attention_head_size = int(d_model / nheads)
        self.all_head_size = nheads * self.attention_head_size

        self.detach_mode = detach_mode

        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

        if self.detach_mode == 'yes':
            # assert self.detach_mode==
            print('Detach K-Q-softmax branch')

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.nheads, self.attention_head_size)
        x = x.view(*new_x_shape)
        X= x.permute(0, 2, 1, 3)
        return X
    
    def un_transpose_for_scores(self, x, old_shape):
        x = x.permute(0, 1, 2, 3)
        return x.reshape(old_shape)
    
    @staticmethod
    def pproc(layer, player, x):
        z = layer(x)
        zp = player(x)
        return zp * (z / zp).data
        
    def forward(self, x, attn_mask = None, key_padding_mask = None):
        
        pquery = make_p_layer(self.query, 0.01)
        pkey = make_p_layer(self.key, 0.01)
        pvalue = make_p_layer(self.value, 0.01)
                
        if self.detach_mode != 'yes':
            query_ = self.query(x) 
            key_ = self.key(x) 
            val_ = self.value(x)
        else:
            query_ = self.pproc(self.query, pquery, x) 
            key_ = self.pproc(self.key, pkey, x)
            val_ = self.pproc(self.value, pvalue, x)     

        # [1, senlen, 768] -> [1, 12, senlen, 64]
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)        
        
        # torch.Size([1, 12, 10, 64]) , torch.Size([1, 12, 64, 10]) -> torch.Size([B, H, L, L])
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))
     
        #if torch.isnan(attention_scores).any():
        #    import pdb;pdb.set_trace()

        if self.detach_mode == 'yes':
        #     assert self.config.detach_mode==False
            # print("Detached")
            attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        
        if self.detach_mode != 'yes':
            attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        output = context_layer.view(*new_context_layer_shape)

        # attention_probs = torch.mean(attention_probs, dim=1)

        return output, attention_probs


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 2048, detach_mode = 'yes', dropout = 0.1, activation = "relu", layer_norm_eps = 1e-5):
        super(TransformerEncoderLayer, self).__init__()
        # d_model is emb_size
        self.self_attn = MultiheadAttention(d_model, nhead, detach_mode, dropout=dropout)#, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if detach_mode != 'yes':
            largs = LNargs()
        else:
            largs = LNargsDetach()

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, args=largs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, args=largs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)


    def forward(self, src, src_mask = None, src_key_padding_mask = None): # Encoder Layer 이전에서부터 [L, B, D]의 형태로 들어옴
        
        # attention weight들은 batch_first랑 상관없이 똑같이 나옴
        # if torch.isnan(src).sum() != 0:
        #     print("Encoder -1")
        src2, attn = self.self_attn(src, attn_mask = src_mask,
                              key_padding_mask = src_key_padding_mask)
        # if (src2 != src2).sum() != 0:
        #     print("Encoder 0")
        src = src + self.dropout1(src2)
        # if (src != src).sum() != 0:
        #     print("Encoder 1")
        src = self.norm1(src)
        # if (src != src).sum() != 0:
        #     print("Encoder 2")
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        # if (src2 != src2).sum() != 0:
        #     print("Encoder 3")
        src = src + self.dropout3(src2)
        # if (src != src).sum() != 0:
        #     print("Encoder 4")
        src = self.norm2(src)
        # if (src != src).sum() != 0:
        #     print("Encoder 5")
        return src, attn

    def pforward(self, src):

        src2, attn = self.self_attn(src, attn_mask = None,
                              key_padding_mask = None)

        plinear1 = make_p_layer(self.linear1, 0.01)
        plinear2 = make_p_layer(self.linear2, 0.01)

        if self.detach_mode != 'yes':
            src = src + self.dropout1(src2)
        else:
            src = src + src2

        src = self.norm1(src)
        src2 = plinear1(src)
        src2 = self.activation(src2)
        
        if self.detach_mode != 'yes':
            src2 = self.dropout2(src2)

        src2 = plinear2(src2)

        if self.detach_mode != 'yes':
            src = src + self.dropout3(src2)
        else:
            src = src + src2
        src = self.norm2(src)

        return src, attn


class TransformerEncoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, device, norm = None):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, src, mask = None, src_key_padding_mask = None): 

        output = src # [B, L, D]
        # print(output.shape)
        attn_output = torch.zeros((src.shape[0], src.shape[1], src.shape[1]), device = self.device) # batch, seq_len, seq_len
        for num, mod in enumerate(self.layers):
            # print("Encoder Layer num!!!!!!!!!!!!!! :", num)
            if (output != output).sum() != 0:
                # print(output)
                # print("How many nans ? :", torch.isnan(output).sum())
                # print("output shape :", output.shape)
                nan_indices = torch.isnan(output)
                output[nan_indices] = 0
                # nan_indices = nan_indices.nonzero()
                # print("where? :", nan_indices)
                # print("transformer Encoder src")
            output, attn = mod(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
            # if (output != output).sum() != 0:
            #     print("transformer debug 1")
            attn = torch.mean(attn, dim=1)
            attn_output += attn
           
        if self.norm is not None:
            output = self.norm(output)
            if (output != output).sum() != 0:
                print("transformer debug 2")
        return output, attn_output