# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import argparse
from tqdm import tqdm

import warnings
from multitask_transformer_class_xai import MultitaskTransformerModel
from utils_mod import get_prop, data_loader, preprocess, test, preprocess_for_XAI, initialize_training, compute_joint_attention
from captum.attr import Saliency, InputXGradient, DeepLift, GuidedBackprop, GradientShap, DeepLiftShap, IntegratedGradients
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='uni_syn')
parser.add_argument('--batch', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--task_rate', type=float, default=0.25)
parser.add_argument('--masking_ratio', type=float, default=0.15)
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='macro')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--model_path', type=str, default='./check_points/uni_syn_1024_0.001_3_128_4_1.000.mdl', help='model path')
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_task', type=int, default=128)
parser.add_argument('--nhid_tar', type=int, default=128)
parser.add_argument('--task_type', type=str, default='classification', help='[classification, regression]')
parser.add_argument('--detach_mode', type=str, default='yes', help='[yes, no]')
parser.add_argument('--device', type=str, default='cuda:0', help='[cuda or cpu]')
parser.add_argument('--xai_method', type=str, default='GI', help='[GI, DeepLift, Saliency, LRP]')
parser.add_argument('--top_percent', type=int, default=90)


args = parser.parse_args()


prop = vars(args)
device = prop['device']

X_train, y_train, X_test, y_test = data_loader(args.dataset, prop['task_type'])
X_test = X_test[:-1]
X_train_task, y_train_task, X_test, y_test = preprocess(prop, X_train, y_train, X_test, y_test)

prop['nclasses'] = torch.max(y_train_task).item() + 1 if prop['task_type'] == 'classification' else None
prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_task.shape[1], X_train_task.shape[2]

if prop['xai_method'] == 'LRP':
    prop['detach_mode'] = 'yes'
else:
    prop['detach_mode'] = 'no'

model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = initialize_training(prop)
model_path = prop['model_path']
chkpt = torch.load(model_path)
best_model.load_state_dict(chkpt)

class LRP:

    def __init__(self, model):
        self.model = model

    def attribute(self, x, target):
        x = x
        y = target
        _, R, _ = self.model.attribute(x, y, task_type='classification')

        return R

class Attn_Last:

    def __init__(self, model):
        self.model = model

    def attribute(self, x, target):
        x = x
        y = target
        _, _, attention_probs = self.model.attribute(x, y, task_type='classification')
        layer_idxs = attention_probs.keys()

        R = np.expand_dims(np.mean([x_.sum(1) for x_ in attention_probs[max(layer_idxs)].detach().cpu().numpy()], 1), axis=-1) 
        return R

class Rollout:
    def __init__(self, model):
        self.model = model

    def attribute(self, x, target):
        x = x
        y = target
        _, _, attention_probs = self.model.attribute(x, y, task_type='classification')

        attns = [attention_probs[k].detach().cpu().numpy() for k in sorted(attention_probs.keys())]
        attentions_mat = np.stack(attns,axis=0) # (num_layer, B, H, L, L)
        attentions_mat = attentions_mat.swapaxes(0, 1)
        res_att_mat = attentions_mat.sum(axis=2)/attentions_mat.shape[2] # head를 기준으로 해야하니 axis를 2로 변경
        joint_attentions = compute_joint_attention(res_att_mat, add_residual=True) # B N L L
        attribution = joint_attentions[:,-1, :, :].sum(1, keepdims=True)
        R = attribution.swapaxes(1, 2)

        return R

class Random_attr:

    def __init__(self, model):
        self.model = model

    def attribute(self, x, target):
        x = x
        y = target
        R = torch.rand(x.shape)

        return R


def expand_mask(mask):
    expanded_mask = np.copy(mask)
    expanded_mask[:, 1:, :] = np.logical_or(expanded_mask[:, 1:, :], mask[:, :-1, :])
    expanded_mask[:, :-1, :] = np.logical_or(expanded_mask[:, :-1, :], mask[:, 1:, :])
    return expanded_mask

xai_dict = {'LRP': LRP,
           'Saliency': Saliency,
           'GI': InputXGradient,
           'DeepLift': DeepLift,
           'GuidedBackprop': GuidedBackprop,
           'GradientShap': GradientShap,
           'attn_last': Attn_Last,
           'Random': Random_attr,
           'rollout': Rollout,
           'IntegratedGradients': IntegratedGradients,
           'DeepShap': DeepLiftShap
           }



def evaluate(y_pred, y, nclasses, criterion, task_type, device, avg):
    results = []

    loss = criterion(y_pred.view(-1, nclasses), torch.as_tensor(y, device = device)).item()
    
    pred, target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
    pred = np.argmax(pred, axis = 1)
    acc = accuracy_score(target, pred)
    prec =  precision_score(target, pred, average = avg)
    rec = recall_score(target, pred, average = avg)
    f1 = f1_score(target, pred, average = avg)
    
    results.extend([loss, acc, prec, rec, f1])

    
    return results


def in_range_ratio(model, X, y, batch, nclasses, xai_method_1, criterion, task_type, device, avg, top_percent):
    
    if xai_method_1 != 'LRP':
        model.eval() # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)  # 7
    
    xai_method = xai_dict[xai_method_1](model)

    total = 0
    total2 = 0

    print("num_batches :", num_batches)

    for i in tqdm(range(num_batches)):
        start = int(i * batch)
        end = int((i + 1) * batch)

        num_inst = y[start : end].shape[0]
        
            
        input_x = torch.as_tensor(X[start : start + num_inst], device = device)
        input_y = torch.as_tensor(y[start : start + num_inst], device = device)

        # relevance_scores = torch.randn(input_x.shape)
        if xai_method_1 == 'GradientShap' or xai_method_1 == 'DeepShap' or xai_method_1=='IntegratedGradients':
            relevance_scores = xai_method.attribute(input_x, target = input_y,  additional_forward_args=['classification'], baselines = torch.zeros_like(input_x))
        elif xai_method_1 in ['LRP', 'attn_last', 'attn_max_last', 'rollout', 'Random'] :
            relevance_scores = xai_method.attribute(input_x, target = input_y)
        else:
            relevance_scores = xai_method.attribute(input_x, target = input_y,  additional_forward_args='classification')
        #else:
        #    relevance_scores = xai_method.attribute(input_x, target = input_y)
        if xai_method_1 in ['rollout', 'attn_last']:
            relevance_scores = relevance_scores
            percentiles = np.percentile(relevance_scores, q=top_percent, axis=1)
        else:
            relevance_scores = relevance_scores.detach().cpu()
            percentiles = np.percentile(relevance_scores.numpy(), q=top_percent, axis=1)            
            percentiles = torch.from_numpy(percentiles)
        
        
        mask = torch.from_numpy(np.where(relevance_scores > percentiles[:, np.newaxis], 1, 0))

        range_mask = np.where(fall_range[start : end] == 1, 1, 0)
        
        range_mask = range_mask.reshape(num_inst, -1, 1) # 실제 range
        
        in_range = range_mask * mask.numpy() # 얼마나 range 내에 존재하는가
        mask_sum = mask.sum()
        in_range_sum = in_range.sum()

        range_mask_sum = range_mask.sum()
        
        total += in_range_sum / mask_sum # 모델이 1이고 한 것 중에 실제로 1인 것들

        total2 += in_range_sum / range_mask_sum

    return total / num_batches, total2 / num_batches

fall_range = np.load('./uni_syn_range.npy')

batch_idx = fall_range.shape[0]

top_percent = prop['top_percent']

ap, ar = in_range_ratio(best_model, X_test[batch_idx:], y_test[batch_idx:], prop['batch'], prop['nclasses'], prop['xai_method'], criterion_task, prop['task_type'], prop['device'], prop['avg'], top_percent)

print("AP :", ap)
print("AR :", ar)
af = 2 * (ap * ar/(ap+ ar))
print('AF1 :', af)



# EXITT
# AP : tensor(0.3898)
# AR : 0.18685545224006753
# AF1 : tensor(0.2526)

# Saliency 
# AP : tensor(0.3629)
# AR : 0.1741994928148774
# AF1 : tensor(0.2354)

# GI
# AP : tensor(0.3348)
# AR : 0.16055114116652583
# AF1 : tensor(0.2170)

# GB
# AP : tensor(0.3301)
# AR : 0.15842603550295872
# AF1 : tensor(0.2141)

# DeepLift
# AP : tensor(0.3329)
# AR : 0.15962806424344894
# AF1 : tensor(0.2158)

# DeepSHAP
# AP : tensor(0.3329)
# AR : 0.15962806424344894
# AF1 : tensor(0.2158)

# Rollout
# AP : tensor(0.3478)
# AR : 0.16695519864750621
# AF1 : tensor(0.2256)

# A-Last
# AP : tensor(0.3104)
# AR : 0.1490143702451394
# AF1 : tensor(0.2014)
