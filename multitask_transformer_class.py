# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:04:31 2020

@author: Ranak Roy Chowdhury
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformer
from transformer import make_p_layer, LNargs, LNargsDetach
from layer_norm import LayerNorm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 0)


class MultitaskTransformerModel(nn.Module):

    def __init__(self, task_type, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_tar, nhid_task, nlayers, detach_mode, dropout = 0.1):
        super(MultitaskTransformerModel, self).__init__()
        # from torch.nn import TransformerEncoder, TransformerEncoderLayer

        if detach_mode != 'yes':
            largs = LNargs()
        else:
            largs = LNargsDetach()

        self.nclasses = nclasses

        self.trunk_net = nn.Sequential(
            nn.Linear(input_size, emb_size),
            # nn.BatchNorm1d(batch),
            LayerNorm(emb_size, eps=1e-5, args=largs),
            PositionalEncoding(emb_size, max_len=seq_len),
            # nn.BatchNorm1d(batch)
            LayerNorm(emb_size, eps=1e-5, args=largs)
        )
        
        encoder_layers = transformer.TransformerEncoderLayer(emb_size, nhead, nhid, detach_mode, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers, device)
        
        self.layer_norm = LayerNorm(emb_size, eps=1e-5, args=largs)
        

        self.attention_probs = {i: [] for i in range(nlayers)}

        # Task-aware Reconstruction Layers
        self.tar_net = nn.Sequential(
            nn.Linear(emb_size, nhid_tar),
            # nn.BatchNorm1d(batch),
            LayerNorm(emb_size, eps=1e-5, args=largs),
            nn.Linear(nhid_tar, nhid_tar),
            # nn.BatchNorm1d(batch),
            LayerNorm(nhid_tar, eps=1e-5, args=largs),
            nn.Linear(nhid_tar, input_size),
        )

        if task_type == 'classification':
            # Classification Layers

            if detach_mode != 'yes':

                self.class_net = nn.Sequential(
                    nn.Linear(emb_size, nhid_task),
                    nn.ReLU(),
                    nn.Dropout(p = 0.3),
                    nn.Linear(nhid_task, nhid_task),
                    nn.ReLU(),
                    nn.Dropout(p = 0.3),
                    nn.Linear(nhid_task, nclasses)
                )
            else:
                self.class_net = nn.Sequential(
                    nn.Linear(emb_size, nhid_task),
                    nn.ReLU(),
                    nn.Dropout(p = 0),
                    nn.Linear(nhid_task, nhid_task),
                    nn.ReLU(),
                    nn.Dropout(p = 0),
                    nn.Linear(nhid_task, nclasses)
                )
        else:
            # Regression Layers
            self.reg_net = nn.Sequential(
                nn.Linear(emb_size, nhid_task),
                nn.ReLU(),
                nn.Linear(nhid_task, nhid_task),
                nn.ReLU(),
                nn.Linear(nhid_task, 1),
            )
            
    @staticmethod
    def pproc(layer,player,x):
        z = layer(x)
        zp = player(x)
        return zp * (z / zp).data
        
    def forward(self, x, task_type):
        x = self.trunk_net(x)
        x, attn = self.transformer_encoder(x)
        x = self.layer_norm(x)

        # X shape : B L D

        if task_type == 'reconstruction':
            output = self.tar_net(x)#.permute(1, 0, 2)

        elif task_type == 'classification':
            output = self.class_net(x[:,-1,:])

        elif task_type == 'regression':
            # B, L, D 에서 
            output = self.reg_net(x[:,-1,:])

        return output, attn


    def forward_and_explain(self, x, y, task_type):
        
        A = {}

        xdata = x.data
        xdata.requires_grad_(True)

        A['x_data'] = xdata
        A['x'] = x

        trunk_input = x

        for i, trunk in enumerate(self.trunk_net):

            trunk_inputdata = trunk_input.data
            trunk_inputdata.requires_grad_(True)

            A['trunk_input_{}_data'.format(i)] = trunk_inputdata
            A['trunk_input_{}'.format(i)] = trunk_input

            if i == 0:
                pembedding = make_p_layer(trunk)
                output = self.pproc(trunk, pembedding, A['trunk_input_{}_data'.format(i)])
            elif i == 2: # Positional Encoding이라면
                output = trunk(A['trunk_input_{}_data'.format(i)])
                pe = output - A['trunk_input_{}_data'.format(i)]
            else:
                output = trunk(A['trunk_input_{}_data'.format(i)])


            trunk_input = output

        hidden_states = output

        hidden_states_data = hidden_states.data
        hidden_states_data.requires_grad_(True)
        A['hidden_states_data'] = hidden_states_data
        A['hidden_states'] = hidden_states

        attn_input = hidden_states_data

        for i, block in enumerate(self.transformer_encoder.layers):
            
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            A['attn_input_{}_data'.format(i)] = attn_inputdata
            A['attn_input_{}'.format(i)] = attn_input

            output, attention_probs = block(A['attn_input_{}_data'.format(i)])

            self.attention_probs[i] = attention_probs
            attn_input = output

        outputdata = output.data
        outputdata.requires_grad_(True)

        norm_out = self.layer_norm(outputdata)

        norm_outdata = norm_out.data
        norm_outdata.requires_grad_(True)

        logits = self.class_net(norm_outdata[:,-1,:])

        A['logits'] = logits
        Rout = A['logits'] * F.one_hot(y, num_classes = self.nclasses)

        self.R0 = Rout.detach().cpu().numpy()

        Rout.sum().backward()
        
        print("Rout :", Rout.sum().item())

        ((norm_outdata.grad)*norm_out).sum().backward()

        Rpool = ((outputdata.grad)*output)
        print("Rpool :", Rpool.sum().item())
        R_ = Rpool

        for i, block in list(enumerate(self.transformer_encoder.layers))[::-1]:
            R_.sum().backward()
            
            R_grad = A['attn_input_{}_data'.format(i)].grad
            R_attn =  (R_grad)*A['attn_input_{}'.format(i)]
              
            R_ = R_attn
            print("attention", i, ":", R_.sum().item())

        R_.sum().backward()
        R_hidden_grad = A['hidden_states_data'].grad  # 얘가 값이 없음
        R_hidden = (R_hidden_grad) * A['hidden_states'] # 문제 발생
        print("R_hidden :", R_hidden.sum().item())
        # R_hidden.sum().backward()

        R_t = R_hidden

        for i, trunk in list(enumerate(self.trunk_net))[::-1]:
            R_t.sum().backward()

            R_t_grad = A['trunk_input_{}_data'.format(i)].grad
            if i == 3:
                R_t = (R_t_grad) * (A['trunk_input_{}'.format(i)] - pe)
            else:
                R_t = (R_t_grad) * A['trunk_input_{}'.format(i)]
            print("R_t at", i,":", R_t.sum().item())

        return logits, R_t, R_hidden
        