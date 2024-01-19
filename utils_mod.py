# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 01:05:24 2021

@author: Ranak Roy Chowdhury
"""
import sys

import warnings, pickle, torch, math, os, random, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets
import multitask_transformer_class_xai
import pandas as pd
import os
warnings.filterwarnings("ignore")

seed = 9999  # 원하는 시드 값으로 설정

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

def compute_joint_attention(att_mat, add_residual=True):

    if add_residual: #  B, N, L, L
        residual_att = np.eye(att_mat.shape[2])[None, ...] # identity matrix 생성
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat
    
    joint_attentions = np.zeros(aug_att_mat.shape) # B, N, L, L
    # print("joint attn :", joint_attentions.shape)
    layers = joint_attentions.shape[1]
    joint_attentions[:,0,:,:] = aug_att_mat[:, 0, :, :]
    for i in np.arange(1, layers):
        # joint_attentions[:,i, :, :] = aug_att_mat[:, i, :, :].dot(joint_attentions[:, i-1, :, :])
        joint_attentions[:,i, :, :] = np.einsum('bij,bjk->bik', aug_att_mat[:, i, :, :], joint_attentions[:, i-1, :, :])
        
    return joint_attentions

# loading optimized hyperparameters
def get_optimized_hyperparameters(dataset):

    path = './hyperparameters.pkl'
    with open(path, 'rb') as handle:
        all_datasets = pickle.load(handle)
        if dataset in all_datasets:
            prop = all_datasets[dataset]
    return prop


# loading user-specified hyperparameters
def get_user_specified_hyperparameters(args):

    return vars(args)


# loading fixed hyperparameters
def get_fixed_hyperparameters(prop, args):

    prop['lamb'], prop['epochs'], prop['ratio_highest_attention'], prop['avg'] = args.lamb, args.epochs, args.ratio_highest_attention, args.avg
    prop['dropout'], prop['nhid'], prop['nhid_task'], prop['nhid_tar'], prop['dataset'] = args.dropout, args.nhid, args.nhid_task, args.nhid_tar, args.dataset
    return prop


def get_prop(args):
    
    prop = get_user_specified_hyperparameters(args)
    return prop


def data_loader(dataset, task_type):

    ucr = UCR_UEA_datasets()
    mul_list = ['AtrialFibrillation', 'CharacterTrajectories', 'DuckDuckGeese', 'ERing', \
                'JapaneseVowels', 'AppliancesEnergy', 'IEEEPPG', 'LiveFuelMoistureContent', \
                    'FloodModeling1', 'Covid3Month', 'BenzeneConcentration', 'BeijingPM10Quality']

    print("dataset :", dataset)
    
    if dataset in mul_list:
            
        X_train = np.load('./data/'+dataset+'_train.npy', allow_pickle=True)       
        y_train = np.load('./data/'+dataset+'_train_label.npy', allow_pickle=True)
        X_test = np.load('./data/'+dataset+'_test.npy', allow_pickle=True)
        y_test = np.load('./data/'+dataset+'_test_label.npy', allow_pickle=True)

        print("First loaded shape............")
        print("X_train shape :", X_train.shape)
        print("y_train shape :", y_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_test shape :", y_test.shape)
    

    elif dataset != 'uni_syn' and dataset != 'synthetic_multi' and dataset != 'multi_syn':

        X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
    
    elif dataset == 'uni_syn':

        X_train = np.load('./uni_syn_train.npy')
        
        y_train = np.load('./uni_syn_train_label.npy')
        X_test = np.load('./uni_syn_test.npy')
        y_test = np.load('./uni_syn_test_label.npy')

    elif dataset == 'multi_syn':

        X_train = np.load('./multi_syn_train.npy')
        
        y_train = np.load('./multi_syn_train_label.npy')
        X_test = np.load('./multi_syn_test.npy')
        y_test = np.load('./multi_syn_test_label.npy')
    
    # elif dataset == 'synthetic_multi':
    #     print("Loaded")

    #     X_train = np.load('./synthetic_multi_train.npy')
        
    #     y_train = np.load('./synthetic_multi_train_label.npy')
    #     X_test = np.load('./synthetic_multi_test.npy')
    #     y_test = np.load('./synthetic_multi_test_label.npy')

    elif dataset == 'SmartFall':

        X_train = np.load('./fall_train.npy')
        y_train = np.load('./fall_train_label.npy')
        X_test = np.load('./fall_test.npy')
        y_test = np.load('./fall_test_label.npy')


    else:
        print("No dataset")

    X_train = X_train.astype(np.float)
    X_test = X_test.astype(np.float)

    train_has_nan = np.isnan(X_train).any()
    test_has_nan = np.isnan(X_test).any()

    

    if train_has_nan:    
        print("train set has Nan val.. Preprocessing start")
        #X_train_minval = np.nanmin(X_train, axis = (0, 1))
        X_train_mask = np.isnan(X_train)
        X_train[X_train_mask] = 0
        print('Done')

    if test_has_nan:
        print("test set has Nan val.. Preprocessing start")
        #X_test_minval = np.nanmin(X_test, axis = (0, 1))
        X_test_mask = np.isnan(X_test)
        X_test[X_test_mask] = 0
        print('Done')

    if task_type == 'classification':
        # print("Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        train_len = y_train.shape[0]
        y = np.concatenate((y_train, y_test))

        if dataset == 'HandMovementDirection': # this one has special labels
            y = [yy[0] for yy in y]
        y = LabelEncoder().fit_transform(y) # sometimes labels are strings or start from 1
        assert(y.min() == 0) # assert labels are integers and start from 0
        
        y_train = y[:train_len]
        y_test = y[train_len:]

    elif task_type == 'regression':
        y_train = y_train.astype(np.float)
        y_test = y_test.astype(np.float)

    # print("Ended Dataloader shape............")
    # print("X_train shape :", X_train.shape)
    # print("y_train shape :", y_train.shape)
    # print("X_test shape :", X_test.shape)
    # print("y_test shape :", y_test.shape)
        
    return X_train, y_train, X_test, y_test
    
def make_perfect_batch(X, num_inst, num_samples):

    extension = np.zeros((num_samples - num_inst, X.shape[1], X.shape[2]))
    X = np.concatenate((X, extension), axis = 0)
    return X

def mean_standardize_fit(X):
    m1 = np.mean(X, axis = 1)
    mean = np.mean(m1, axis = 0)
    
    s1 = np.std(X, axis = 1)
    std = np.mean(s1, axis = 0)
    
    return mean, std

def mean_standardize_transform(X, mean, std):
    return (X - mean) / std

def preprocess(prop, X_train, y_train, X_test, y_test):
    
    mean, std = mean_standardize_fit(X_train) # 평균과 표준편차 구함
    X_train, X_test = mean_standardize_transform(X_train, mean, std), mean_standardize_transform(X_test, mean, std)

    num_train_inst, num_test_inst = X_train.shape[0], X_test.shape[0] # Train size
    num_train_samples = math.ceil(num_train_inst / prop['batch']) * prop['batch'] # 만약 전체 수가 Batch보다 작으면 Bath만큼 데이터가 늘어남
    num_test_samples = math.ceil(num_test_inst / prop['batch']) * prop['batch']
    
    X_train = make_perfect_batch(X_train, num_train_inst, num_train_samples) # X_train 300 640 2
    X_test = make_perfect_batch(X_test, num_test_inst, num_test_samples)

    X_train_task = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()

    if prop['task_type'] == 'classification':
        y_train_task = torch.as_tensor(y_train)
        y_test = torch.as_tensor(y_test)
    else:
        y_train_task = torch.as_tensor(y_train).float()
        y_test = torch.as_tensor(y_test).float()

    return X_train_task, y_train_task, X_test, y_test


def preprocess_for_XAI(prop, X_train, y_train, X_test, y_test):
    
    mean, std = mean_standardize_fit(X_train) # 평균과 표준편차 구함
    X_train, X_test = mean_standardize_transform(X_train, mean, std), mean_standardize_transform(X_test, mean, std)

    if prop['task_type'] == 'classification':
        y_train = torch.as_tensor(y_train)
        y_test = torch.as_tensor(y_test)
    else:
        y_train = torch.as_tensor(y_train).float()
        y_test = torch.as_tensor(y_test).float()

    return X_train, y_train, X_test, y_test


def initialize_training(prop):
    model = multitask_transformer_class_xai.MultitaskTransformerModel(prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'], \
        prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'], prop['nhid_task'], prop['nlayers'], prop['detach_mode'], prop['dropout']).to(prop['device'])
    best_model = multitask_transformer_class_xai.MultitaskTransformerModel(prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'], \
        prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'], prop['nhid_task'], prop['nlayers'], prop['detach_mode'], prop['dropout']).to(prop['device'])

    criterion_tar = torch.nn.MSELoss()
    criterion_task = torch.nn.CrossEntropyLoss() if prop['task_type'] == 'classification' else torch.nn.MSELoss() # nn.L1Loss() for MAE
    optimizer = torch.optim.Adam(model.parameters(), lr = prop['lr'])
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr = prop['lr']) # get new optimiser

    return model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer


def attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights):

    res, index = instance_weights.topk(int(math.ceil(ratio_highest_attention * X.shape[1])))
    index = index.cpu().data.tolist()
    index2 = [random.sample(index[i], int(math.ceil(masking_ratio * X.shape[1]))) for i in range(X.shape[0])]
    return np.array(index2)

def random_instance_masking(X, masking_ratio, ratio_highest_attention, instance_weights):
    # 전체 데이터 X에 대한 Masking을 진행한다.
    # 결론적으로 Batch indexing 방법을 변경하기 어려울 것으로 보임.
    # 현재 BatchNorm에 해당하는 부분을 모두 LayerNorm으로 교체하였음
    # 입력 파라미터는 Train 전체, 랜덤 Masking, 상위 Masking, [Full batch, Length]의 텐서

    indices = attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights)
    # print("indices :", indices.shape)
    boolean_indices = np.array([[True if i in index else False for i in range(X.shape[1])] for index in indices])
    # print("boolean_indices :", boolean_indices.shape)  # (112, 96) -> (FB, L)

    boolean_indices_masked = np.repeat(boolean_indices[ : , : , np.newaxis], X.shape[2], axis = 2)
    # print("boolean_indices_masked :", boolean_indices_masked.shape)  # (FB, L, D)
    boolean_indices_unmasked =  np.invert(boolean_indices_masked)
    
    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = np.copy(X), np.copy(X), np.copy(X)
    X_train_tar = np.where(boolean_indices_unmasked, X, 0.0)
    y_train_tar_masked = y_train_tar_masked[boolean_indices_masked].reshape(X.shape[0], -1)  # Task type이 아마 regression인 경우 사용
    y_train_tar_unmasked = y_train_tar_unmasked[boolean_indices_unmasked].reshape(X.shape[0], -1)
    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = torch.as_tensor(X_train_tar).float(), torch.as_tensor(y_train_tar_masked).float(), torch.as_tensor(y_train_tar_unmasked).float()

    return X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked

    

def compute_tar_loss(model, device, criterion_tar, y_train_tar_masked, y_train_tar_unmasked, batched_input_tar, \
                    batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start):

    model.train()
    out_tar = model(torch.as_tensor(batched_input_tar, device = device), 'reconstruction')[0]

    # print("batched_input_tar :", batched_input_tar.shape) # torch.Size([16, 96, 1])

    out_tar_masked = torch.as_tensor(out_tar[torch.as_tensor(batched_boolean_indices_masked)].reshape(out_tar.shape[0], -1), device = device)
    # print("out_tar_masked :", out_tar_masked.shape) # torch.Size([16, 29])
    out_tar_unmasked = torch.as_tensor(out_tar[torch.as_tensor(batched_boolean_indices_unmasked)].reshape(out_tar.shape[0], -1), device = device)

    # print("out_tar_unmasked :", out_tar_unmasked.shape) # torch.Size([16, 67])

    # loss tar masked -> shape이 있는 것이 아니라 단순 scalar 값임
    loss_tar_masked = criterion_tar(out_tar_masked[ : num_inst], torch.as_tensor(y_train_tar_masked[start : start + num_inst], device = device))

    loss_tar_unmasked = criterion_tar(out_tar_unmasked[ : num_inst], torch.as_tensor(y_train_tar_unmasked[start : start + num_inst], device = device))
    # print("loss_tar_unmasked :", loss_tar_unmasked.shape)

    # sys.exit()
    
    return loss_tar_masked, loss_tar_unmasked


def compute_task_loss(nclasses, model, device, criterion_task, y_train_task, batched_input_task, task_type, num_inst, start):
    model.train()
    out_task, attn = model(torch.as_tensor(batched_input_task, device = device), task_type)
    out_task = out_task.view(-1, nclasses) if task_type == 'classification' else out_task.squeeze()
    loss_task = criterion_task(out_task[ : num_inst], torch.as_tensor(y_train_task[start : start + num_inst], device = device)) # dtype = torch.long
    return attn, loss_task


def multitask_train(model, criterion_tar, criterion_task, optimizer, X_train_tar, X_train_task, y_train_tar_masked, y_train_tar_unmasked, \
                    y_train_task, boolean_indices_masked, boolean_indices_unmasked, prop):
    
    model.train() # Turn on the train mode

    # y_train_tar_masked와 y_train_tar_unmasked는 tar loss를 계산하는데에 사용된다.
    # 


    total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task = 0.0, 0.0, 0.0
    num_batches = math.ceil(X_train_tar.shape[0] / prop['batch'])
    output, attn_arr = [], []
    
    for i in range(num_batches):
        start = int(i * prop['batch'])
        end = int((i + 1) * prop['batch'])
        num_inst = y_train_task[start : end].shape[0]
        
        optimizer.zero_grad()
        
        batched_input_tar = X_train_tar[start : end] # 배치 입력
        batched_input_task = X_train_task[start : end] # 배치 입력
        
        batched_boolean_indices_masked = boolean_indices_masked[start : end] # DataLoader에서 불러오는 방식으로 해야함
        batched_boolean_indices_unmasked = boolean_indices_unmasked[start : end] # (15, 640, 2)

        loss_tar_masked, loss_tar_unmasked = compute_tar_loss(model, prop['device'], criterion_tar, y_train_tar_masked, y_train_tar_unmasked, \
            batched_input_tar, batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start)
        
        attn, loss_task = compute_task_loss(prop['nclasses'], model, prop['device'], criterion_task, y_train_task, \
            batched_input_task, prop['task_type'], num_inst, start)

        total_loss_tar_masked += loss_tar_masked.item() 
        total_loss_tar_unmasked += loss_tar_unmasked.item()
        total_loss_task += loss_task.item() * num_inst
                
        loss = prop['task_rate'] * (prop['lamb'] * loss_tar_masked + (1 - prop['lamb']) * loss_tar_unmasked) + (1 - prop['task_rate']) * loss_task
        loss.backward()
        
        optimizer.step()
        
        attn_arr.append(torch.sum(attn, axis = 1) - torch.diagonal(attn, offset = 0, dim1 = 1, dim2 = 2))
    instance_weights = torch.cat(attn_arr, axis = 0)
    return total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task / y_train_task.shape[0], instance_weights


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


def test(model, X, y, batch, nclasses, criterion, task_type, device, avg):
    model.eval() # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)
    
    output_arr = []
    with torch.no_grad():
        for i in range(num_batches):
            start = int(i * batch)
            end = int((i + 1) * batch)
            num_inst = y[start : end].shape[0]
            
            out = model(torch.as_tensor(X[start : end], device = device), task_type)[0]
            output_arr.append(out[ : num_inst])

    return evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)



def training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop):
    tar_loss_masked_arr, tar_loss_unmasked_arr, tar_loss_arr, task_loss_arr, min_task_loss = [], [], [], [], math.inf  # 결론적으로 math.inf밖에 안씀
    acc, rmse, mae = 0, math.inf, math.inf
    saving_acc_best = None
    instance_weights = torch.as_tensor(torch.rand(X_train_task.shape[0], prop['seq_len']), device = prop['device']) # shape : Train_size, L
    # 처음에는 Random한 weight를 넣어줌
    # 데이터 전체를 의미함
        
    for epoch in range(1, prop['epochs'] + 1):  # epoch 단위로 돌아감 
        
        X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked = \
            random_instance_masking(X_train_task, prop['masking_ratio'], prop['ratio_highest_attention'], instance_weights)  # X_train 전체, 랜덤 마스킹 비율, 상위 마스킹 비율, 어텐션 가중치

        tar_loss_masked, tar_loss_unmasked, task_loss, instance_weights = multitask_train(model, criterion_tar, criterion_task, optimizer, 
                                            X_train_tar, X_train_task, y_train_tar_masked, y_train_tar_unmasked, y_train_task, 
                                            boolean_indices_masked, boolean_indices_unmasked, prop)
        
        tar_loss_masked_arr.append(tar_loss_masked)
        tar_loss_unmasked_arr.append(tar_loss_unmasked)
        tar_loss = tar_loss_masked + tar_loss_unmasked
        tar_loss_arr.append(tar_loss)
        task_loss_arr.append(task_loss)
        print('Epoch: ' + str(epoch) + ', TAR Loss: ' + str(tar_loss), ', TASK Loss: ' + str(task_loss))

        # save model and optimizer for lowest training loss on the end task
        if task_loss < min_task_loss:
            min_task_loss = task_loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer.load_state_dict(optimizer.state_dict())
            
        # Saved best model state at the lowest training loss is evaluated on the official test set
        test_metrics = test(best_model, X_test, y_test, prop['batch'], prop['nclasses'], criterion_task, prop['task_type'], prop['device'], prop['avg'])

        
        if prop['task_type'] == 'classification' and test_metrics[1] > acc:
            acc = test_metrics[1]

           
            save_path = './check_points/'
            # Read the CSV file into a DataFrame
            mdl_name = prop['dataset'] +"_"+ str(prop['batch']) +"_"+ str(prop['lr']) +"_"+ str(prop['nlayers']) +"_"+ str(prop['emb_size'])+"_"+ str(prop['nhead']) +"_"+ f"{acc:.3f}" + '.mdl'
            df = pd.read_csv(save_path+'results.csv')
            # Filter the DataFrame based on the value in the first column

            if df['dataset'].isin([prop['dataset']]).any():
                print("exists", prop['dataset'])
                if acc > df.loc[df['dataset'] == prop['dataset'], 'acc'].item():
                    os.remove(save_path + df.loc[df['dataset'] == prop['dataset'], 'mdl_name'].item())
                    # Modify the value in the next column
                    df.loc[df['dataset'] == prop['dataset'], 'acc'] = acc 
                    df.loc[df['dataset'] == prop['dataset'], 'mdl_name'] = mdl_name 
                    df.to_csv(save_path + 'results.csv', index=False)
                    
                    print('Save model because it outperforms past model')
                    saving_acc_best = best_model
                    torch.save(saving_acc_best.state_dict(), save_path + mdl_name )
                else:
                    print('Saving is not conducted because it is worse than past model')
            else:
                print("doesn't exist")
                # Create a dictionary with the values for the new row
                new_row = {'dataset': prop['dataset'], 'acc': acc, 'mdl_name': mdl_name}
                
                # Add the new row to the DataFrame
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(save_path + 'results.csv', index=False)
                saving_acc_best = best_model
                torch.save(saving_acc_best.state_dict(), save_path + mdl_name )
                    


            # saving_acc_best = best_model
            # torch.save(saving_acc_best.state_dict(), './check_points/' + prop['dataset'] +"_"+ str(prop['batch']) +"_"+ str(prop['lr']) +"_"+ str(prop['nlayers']) +"_"+ str(prop['emb_size'])+"_"+ str(prop['nhead']) +"_"+ f"{acc:.3f}" + '.mdl' )
        
        
        
        elif prop['task_type'] == 'regression' and test_metrics[0] < rmse:
            rmse = test_metrics[0]
            mae = test_metrics[1]

    
    if prop['task_type'] == 'classification':
        print('Dataset: ' + prop['dataset'] + ', Acc: ' + str(acc))
    elif prop['task_type'] == 'regression':
        print('Dataset: ' + prop['dataset'] + ', RMSE: ' + str(rmse) + ', MAE: ' + str(mae))
    acc *= 100
    

    del model
    torch.cuda.empty_cache()

