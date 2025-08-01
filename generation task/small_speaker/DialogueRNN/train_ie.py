import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np


import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from model import BiModel, Model, MaskedNLLLoss
from dataloader import MELDDataset, IEMOCAPDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings
warnings.filterwarnings('ignore')

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_loaders(path, batch_size=32, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path, split='train')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    validset = IEMOCAPDataset(path=path, split='val')
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, split='test')
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return trainset.n_classes, train_loader, valid_loader, test_loader

def train_model(model, loss_function, dataloader, epoch, optimizer=None):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert optimizer!=None
    model.train()
    for data in dataloader:
        optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        textf, acouf, visuf, qmask, umask, label =\
                [d.cuda() for d in data[:-2]] if cuda else data[:-2]
        if feature_type == "audio":
            log_prob, alpha, alpha_f, alpha_b = model(acouf, qmask,umask) # seq_len, batch, n_classes
        elif feature_type == "text":
            log_prob, alpha, alpha_f, alpha_b = model(textf, qmask,umask) # seq_len, batch, n_classes
        else:
            log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask) # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        loss.backward()
#             if args.tensorboard:
#                 for param in model.named_parameters():
#                     writer.add_histogram(param[0], param[1].grad, epoch)
        optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='macro')*100,2)
    class_report = classification_report(labels,preds,sample_weight=masks,digits=4)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas, alphas_f, alphas_b, vids], class_report


def eval_model(model, loss_function, dataloader, epoch):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    model.eval()
    for data in dataloader:
        # import ipdb;ipdb.set_trace()
        textf, acouf, visuf, qmask, umask, label, smask =\
                [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if feature_type == "audio":
            log_prob, alpha, alpha_f, alpha_b = model(acouf, qmask,umask) # seq_len, batch, n_classes
        elif feature_type == "text":
            log_prob, alpha, alpha_f, alpha_b = model(textf, qmask,umask) # seq_len, batch, n_classes
        else:
            log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask) # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        # masks.append(umask.view(-1).cpu().numpy())
        masks.append(smask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        
        alphas += alpha
        alphas_f += alpha_f
        alphas_b += alpha_b
        vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)
    class_report = classification_report(labels,preds,sample_weight=masks,digits=4)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas, alphas_f, alphas_b, vids], class_report

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2021, help='random seed')

parser.add_argument('--fea_model', type=str, default='albert_chinese_small-UTT', help='feature model dir')

args = parser.parse_args()
np.random.seed(args.seed)# 1234

cuda = torch.cuda.is_available()
if cuda:
    print('Running on GPU')
else:
    print('Running on CPU')
    
tensorboard = False    
if tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

# choose between 'sentiment' or 'emotion'
classification_type = 'emotion'
feature_type = 'multimodal'

batch_size = 30
# n_classes = 3
n_epochs = 100
active_listener = False
attention = 'general'
class_weight = False
dropout = 0.1
rec_dropout = 0.1
l2 = 0.00001
lr = 0.0001

if feature_type == 'text':
    print("Running on the text features........")
    D_m = 5120
elif feature_type == 'audio':
    print("Running on the audio features........")
    D_m = 1024
else:
    print("Running on the multimodal features........")
    D_m = 5120 + 1024 + 768# T A V

d_path = '/home/jingran/MyBench/features-lianzheng/IEMOCAP/features_utt_all'
D_m = 4096
with open(os.path.join(d_path, 'whisper-base-UTT', 'Ses01F_impro01_F000.npy'), 'rb') as fl:
    a = np.load(fl)
D_m_audio = a.shape[0]
with open(os.path.join(d_path, 'manet_UTT', 'Ses01F_impro01_F000.npy'), 'rb') as fl:
    a = np.load(fl)
D_m_visual = a.shape[0]

D_m = D_m + D_m_audio + D_m_visual
D_g = 150
D_p = 150
D_e = 100
D_h = 100

D_a = 100 # concat attention

loss_weights = torch.FloatTensor([1.0,1.0,1.0,1.0,1.0,1.0])
n_classes, train_loader, valid_loader, test_loader =get_loaders(args.fea_model, batch_size=batch_size, num_workers=0)
        
model = BiModel(D_m, D_g, D_p, D_e, D_h,
                n_classes=n_classes,
                listener_state=active_listener,
                context_attention=attention,
                dropout_rec=rec_dropout,
                dropout=dropout)

if cuda:
    model.cuda()
if class_weight:
    loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
else:
    loss_function = MaskedNLLLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=l2)



best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None


for e in range(n_epochs):
    start_time = time.time()
    train_loss, train_acc, _,_,_,train_fscore,_,_= train_model(model, loss_function, train_loader, e, optimizer)
    valid_loss, valid_acc, _,_,_,val_fscore,_,_= eval_model(model, loss_function, valid_loader, e)
    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_class_report = eval_model(model, loss_function, test_loader, e)

    if best_fscore == None or best_fscore < val_fscore:
        best_fscore, best_loss, best_label, best_pred, best_mask, best_attn =\
                val_fscore, test_loss, test_label, test_pred, test_mask, attentions

#     if args.tensorboard:
#         writer.add_scalar('test: accuracy/loss',test_acc/test_loss,e)
#         writer.add_scalar('train: accuracy/loss',train_acc/train_loss,e)
    print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
            format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                    test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))
    print (test_class_report)
if tensorboard:
    writer.close()

print('Test performance..')
print('Fscore {} accuracy {}'.format(best_fscore,
                                 round(accuracy_score(best_label,best_pred,sample_weight=best_mask)*100,2)))
print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))