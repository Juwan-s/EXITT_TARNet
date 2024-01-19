import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import argparse
import warnings
from multitask_transformer_class_xai import MultitaskTransformerModel
from utils_mod import get_prop, data_loader, preprocess, test, preprocess_for_XAI, initialize_training, compute_joint_attention
from captum.attr import Saliency, InputXGradient, DeepLift, ShapleyValues
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


seed = 9999 # 원하는 시드 값으로 설정

# 파이토치 랜덤 시드 고정
torch.manual_seed(seed)

# CUDA를 사용하고 있다면, CUDA를 위한 랜덤 시드도 고정
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# cuDNN을 사용하고 있다면, determinism을 확보해야 한다.
# Note: 이 옵션은 성능을 떨어뜨릴 수 있다.
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# numpy 랜덤 시드 고정
np.random.seed(seed)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ECG200')
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--emb_size', type=int, default=256)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--task_rate', type=float, default=0.20)
parser.add_argument('--masking_ratio', type=float, default=0.15)
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='macro')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_task', type=int, default=128)
parser.add_argument('--nhid_tar', type=int, default=128)
parser.add_argument('--task_type', type=str, default='classification', help='[classification, regression]')
parser.add_argument('--detach_mode', type=str, default='yes', help='[yes, no]')
parser.add_argument('--device', type=str, default='cuda:0', help='[cuda or cpu]')
parser.add_argument('--model_path', type=str, default='./check_points/ECG200_16_0.0001_3_256_0.89.mdl', help='model path')
parser.add_argument('--xai_method', type=str, default='GI', help='[GI, DeepLift, Saliency, LRP]')
parser.add_argument('--top_percent', type=int, default=5)

args = parser.parse_args()

#  ====================================================================================================
#  Moving Average Class For Relevance Smoothing
#  ====================================================================================================

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) # sefl.kernel_size - 1
        # self.kernel_size = config.seq_len//3
        # self.kernel_size // 2
        left = right = (self.kernel_size-1) // 2
        L_in = x.shape[1] + left + right
        L_out = L_in - self.kernel_size + 1
        if L_out == x.shape[1]:
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2 , 1)
        else:
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2+1, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class LRP:

    def __init__(self, model, task_type):
        self.model = model
        self.task_type = task_type

    def attribute(self, x, target):
        x = x
        y = target
        _, R, _ = self.model.attribute(x, y, task_type=self.task_type)

        return R

class Attn_Last:

    def __init__(self, model, task_type):
        self.model = model
        self.task_type = task_type

    def attribute(self, x, target):
        x = x
        y = target
        _, _, attention_probs = self.model.attribute(x, y, task_type=self.task_type)
        layer_idxs = attention_probs.keys()
        print(layer_idxs)

        R = np.expand_dims(np.mean([x_.sum(1) for x_ in attention_probs[max(layer_idxs)].detach().cpu().numpy()], 1), axis=-1) 
        return R

class Rollout:
    def __init__(self, model, task_type):
        self.model = model
        self.task_type = task_type

    def attribute(self, x, target):
        x = x
        y = target
        _, _, attention_probs = self.model.attribute(x, y, task_type=self.task_type)

        attns = [attention_probs[k].detach().cpu().numpy() for k in sorted(attention_probs.keys())]
        print("Attns shape :", attns[0].shape) # B H L L
        print("len attns :", len(attns)) # 3 -> Attention Map의 수
        attentions_mat = np.stack(attns,axis=0) # (num_layer, B, H, L, L)
        attentions_mat = attentions_mat.swapaxes(0, 1)
        print("attentions_mat :", attentions_mat.shape) # (3, 12, 13, 13) -> (B, num_layer, H, L, L)
        res_att_mat = attentions_mat.sum(axis=2)/attentions_mat.shape[2] # head를 기준으로 해야하니 axis를 2로 변경
        print("res_att_mat :", res_att_mat.shape) # (3, 13, 13) -> B, N, L, L
        joint_attentions = compute_joint_attention(res_att_mat, add_residual=True) # B N L L
        print("joint_attentions :", joint_attentions.shape)
        attribution = joint_attentions[:,-1, :, :].sum(1, keepdims=True)
        R = attribution.swapaxes(1, 2)
        print("attribution :", attribution.shape)

        return R

def expand_mask(mask):
    expanded_mask = np.copy(mask)
    expanded_mask[:, 1:, :] = np.logical_or(expanded_mask[:, 1:, :], mask[:, :-1, :])
    expanded_mask[:, :-1, :] = np.logical_or(expanded_mask[:, :-1, :], mask[:, 1:, :])
    return expanded_mask

#  ====================================================================================================
#  Load Dataset & Make torch dataloader
#  ====================================================================================================

prop = vars(args)
device = prop['device']

X_train, y_train, X_test, y_test = data_loader(args.dataset, prop['task_type'])
X_train_task, y_train_task, X_test, y_test = preprocess(prop, X_train, y_train, X_test, y_test)

prop['nclasses'] = torch.max(y_train_task).item() + 1 if prop['task_type'] == 'classification' else None
prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_task.shape[1], X_train_task.shape[2]

# mv = moving_avg(X_train.shape[1]//10, 1)


#  ====================================================================================================
#  Load model
#  ====================================================================================================

if prop['xai_method'] == 'LRP':
    prop['detach_mode'] = 'yes'
else:
    prop['detach_mode'] = 'no'

model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = initialize_training(prop)

model_path = prop['model_path']

chkpt = torch.load(model_path)
best_model.load_state_dict(chkpt)


xai_dict = {'LRP': LRP,
           'Saliency': Saliency,
           'GI': InputXGradient,
           'DeepLift': DeepLift,
           'Shapely'
           'attn_last': Attn_Last,
           'rollout': Rollout
           }


def evaluate(y_pred, y, nclasses, criterion, task_type, device, avg):
    results = []

    if task_type == 'classification':
        loss = criterion(y_pred.view(-1, nclasses), torch.as_tensor(y, device = device)).item()
        
        pred, target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
        pred = np.argmax(pred, axis = 1)
        acc = accuracy_score(target, pred)
        prec =  precision_score(target, pred, average = avg)
        rec = recall_score(target, pred, average = avg)
        f1 = f1_score(target, pred, average = avg)
        
        results.extend([loss, acc, prec, rec, f1])
    else:
        y_pred = y_pred.squeeze()
        y = torch.as_tensor(y, device = device)
        rmse = math.sqrt( ((y_pred - y) * (y_pred - y)).sum().data / y_pred.shape[0] )
        mae = (torch.abs(y_pred - y).sum().data / y_pred.shape[0]).item()
        results.extend([rmse, mae])
    # per_class_results = precision_recall_fscore_support(target, pred, average = None, labels = list(range(0, nclasses)))
    
    return results


def perturb_zero(model, X, y, batch, nclasses, xai_method_1, criterion, task_type, device, avg, top_percent):
    model.eval() # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)  # 7
    
    xai_method = xai_dict[xai_method_1](model, task_type)

    output_arr = []

    for i in range(num_batches):
        start = int(i * batch)
        end = int((i + 1) * batch)

        num_inst = y[start : end].shape[0]

        input_x = torch.as_tensor(X[start : start + num_inst], device = device)
        input_y = torch.as_tensor(y[start : start + num_inst], device = device)

        # relevance_scores = torch.randn(input_x.shape)

        if xai_method_1 != 'LRP':
            relevance_scores = xai_method.attribute(input_x, target = input_y,  additional_forward_args='classification')
        else:
            relevance_scores = xai_method.attribute(input_x, target = input_y)
        relevance_scores = relevance_scores.detach().cpu()
        # relevance_scores = mv(relevance_scores)
        
        percentiles = np.percentile(relevance_scores.numpy(), q=top_percent, axis=1)
        percentiles = torch.from_numpy(percentiles)
        mask = torch.from_numpy(np.where(relevance_scores > percentiles[:, np.newaxis], 0, 1)).to(device)
        x_perturbed = input_x * mask.to(device)
        out = model(x_perturbed, task_type)

        output_arr.append(out[ : num_inst])

    with torch.no_grad():
        
        test_metrics = evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)

    return test_metrics

def perturb_zero_random(model, X, y, batch, nclasses, xai_method_1, criterion, task_type, device, avg, top_percent):
    model.eval() # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)  # 7
    
    # xai_method = xai_dict[xai_method_1](model)

    output_arr = []

    for i in range(num_batches):
        start = int(i * batch)
        end = int((i + 1) * batch)

        num_inst = y[start : end].shape[0]

        input_x = torch.as_tensor(X[start : start + num_inst], device = device)
        input_y = torch.as_tensor(y[start : start + num_inst], device = device)

        relevance_scores = torch.rand(input_x.shape)

        percentiles = np.percentile(relevance_scores.numpy(), q=top_percent, axis=1)
        percentiles = torch.from_numpy(percentiles)
        mask = torch.from_numpy(np.where(relevance_scores > percentiles[:, np.newaxis], 0, 1)).to(device)
        x_perturbed = input_x * mask.to(device)
        out = model(x_perturbed, task_type)

        output_arr.append(out[ : num_inst])


    with torch.no_grad():
        
        test_metrics = evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)

    return test_metrics



def perturb_mean(model, X, y, batch, nclasses, xai_method_1, criterion, task_type, device, avg, top_percent):

    model.eval() # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)  # 7
    
    xai_method = xai_dict[xai_method_1](model)

    output_arr = []

    for i in range(num_batches):
        start = int(i * batch)
        end = int((i + 1) * batch)

        num_inst = y[start : end].shape[0]

        input_x = torch.as_tensor(X[start : start + num_inst], device = device)
        input_y = torch.as_tensor(y[start : start + num_inst], device = device)


        mean_val = torch.mean(input_x, dim =1).detach().cpu()
        mean_val = mean_val.unsqueeze(1).expand(-1, input_x.shape[1], -1)


        if xai_method_1 != 'LRP':
            relevance_scores = xai_method.attribute(input_x, target = input_y,  additional_forward_args='classification')
        else:
            relevance_scores = xai_method.attribute(input_x, target = input_y)
        relevance_scores = relevance_scores.detach().cpu()
        # relevance_scores = mv(relevance_scores)
        
        # percentiles = np.percentile(relevance_scores.numpy(), q=top_percent, axis=1)
        percentiles = relevance_scores.numpy().mean(axis = 1)
        percentiles = torch.from_numpy(percentiles)
        mask = torch.from_numpy(np.where(relevance_scores > percentiles[:, np.newaxis], 0, 1))

        x_perturbed = torch.from_numpy(np.where(mask == 1, mean_val, input_x.detach().cpu())).to(device)
        out = model(x_perturbed, task_type)

        output_arr.append(out[ : num_inst])


    with torch.no_grad():
        
        test_metrics = evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)

    return test_metrics




def perturb_inverse(model, X, y, batch, nclasses, xai_method_1, criterion, task_type, device, avg, top_percent):

    model.eval() # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)  # 7
    
    xai_method = xai_dict[xai_method_1](model)

    output_arr = []

    for i in range(num_batches):
        start = int(i * batch)
        end = int((i + 1) * batch)

        num_inst = y[start : end].shape[0]

        input_x = torch.as_tensor(X[start : start + num_inst], device = device)
        input_y = torch.as_tensor(y[start : start + num_inst], device = device)

        max_val = torch.max(input_x, dim =1).values.detach().cpu()
        max_val = max_val.unsqueeze(1).expand(-1, input_x.shape[1], -1)

        if xai_method_1 != 'LRP':
            relevance_scores = xai_method.attribute(input_x, target = input_y,  additional_forward_args='classification')
        else:
            relevance_scores = xai_method.attribute(input_x, target = input_y)
        relevance_scores = relevance_scores.detach().cpu()
        
        percentiles = np.percentile(relevance_scores.numpy(), q=top_percent, axis=1)
        percentiles = torch.from_numpy(percentiles)
        mask = torch.from_numpy(np.where(relevance_scores > percentiles[:, np.newaxis], 0, 1))

        x_perturbed = torch.from_numpy(np.where(mask == 1, max_val - input_x.detach().cpu(), input_x.detach().cpu())).to(device)
        out = model(x_perturbed, task_type)

        output_arr.append(out[ : num_inst])


    with torch.no_grad():
        
        test_metrics = evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)

    return test_metrics



top_percent = prop['top_percent']

print(prop['dataset'])

res_zero = perturb_zero(best_model, X_test, y_test, prop['batch'], prop['nclasses'], prop['xai_method'], criterion_task, prop['task_type'], prop['device'], prop['avg'], top_percent)

# print("Accuracy :", res_zero[1], "F1 :", res_zero[-1])


# res_mean = perturb_mean(best_model, X_test, y_test, prop['batch'], prop['nclasses'], prop['xai_method'], criterion_task, prop['task_type'], prop['device'], prop['avg'], top_percent)

# print("Accuracy :", res_mean[1], "F1 :", res_mean[-1])

# res_inv = perturb_inverse(best_model, X_test, y_test, prop['batch'], prop['nclasses'], prop['xai_method'], criterion_task, prop['task_type'], prop['device'], prop['avg'], top_percent)

# print("Accuracy :", res_inv[1], "F1 :", res_inv[-1])

# res_zero = perturb_zero_random(best_model, X_test, y_test, prop['batch'], prop['nclasses'], prop['xai_method'], criterion_task, prop['task_type'], prop['device'], prop['avg'], top_percent)

print("Accuracy :", res_zero[1], "F1 :", res_zero[-1])

df = pd.DataFrame()

# res_avg = (res_zero[1] + res_mean[1] + res_inv[1]) / 3

temp = {'val': {'Acc': res_zero[1], 'F1': res_zero[-1]}}

df = pd.DataFrame(temp)

# df.to_csv('./results/' + prop['dataset'] + '_' + prop['xai_method'] + '.csv')
df.to_csv('./results/' + prop['dataset'] + '_' + 'random' + '.csv')