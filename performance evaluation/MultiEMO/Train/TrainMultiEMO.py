import sys
sys.path.append('Loss')
sys.path.append('Model')
sys.path.append('Dataset')
from SampleWeightedFocalContrastiveLoss import SampleWeightedFocalContrastiveLoss
from SoftHGRLoss import SoftHGRLoss
from IEMOCAPDataset import IEMOCAPDataset
from MELDDataset import MELDDataset
from MultiEMO_Model import MultiEMO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from optparse import OptionParser
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data.sampler import SubsetRandomSampler
import random




class TrainMultiEMO():

    def __init__(self, dataset, batch_size, num_epochs, learning_rate, weight_decay, 
                 num_layers, model_dim, num_heads, hidden_dim, dropout_rate, dropout_rec,
                 temp_param, focus_param, sample_weight_param, SWFC_loss_param, 
                 HGR_loss_param, CE_loss_param, multi_attn_flag, device, fea_model):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.dropout_rec = dropout_rec
        self.temp_param = temp_param
        self.focus_param = focus_param
        self.sample_weight_param = sample_weight_param
        self.SWFC_loss_param = SWFC_loss_param
        self.HGR_loss_param = HGR_loss_param
        self.CE_loss_param = CE_loss_param
        self.multi_attn_flag = multi_attn_flag
        self.device = device
        self.fea_model = fea_model
        self.best_test_f1 = 0.0
        self.best_epoch = 1
        self.best_test_report = None

        self.get_dataloader()
        self.get_model()
        self.get_loss()
        self.get_optimizer()
    

    def get_train_valid_sampler(self, train_dataset, valid = 0.1):
        size = len(train_dataset)
        idx = list(range(size))
        split = int(valid * size)
        np.random.shuffle(idx)
        return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


    def get_dataloader(self, valid = 0.05):
        if self.dataset == 'IEMOCAP':
            train_dataset = IEMOCAPDataset(self.fea_model, split = 'train')
            valid_dataset = IEMOCAPDataset(self.fea_model, split = 'valid')
            test_dataset = IEMOCAPDataset(self.fea_model, split = 'test')
        elif self.dataset == 'MELD':
            train_dataset = MELDDataset(self.fea_model, split = 'train')
            valid_dataset = MELDDataset(self.fea_model, split = 'valid')
            test_dataset = MELDDataset(self.fea_model, split = 'test')

        # train_sampler, valid_sampler = self.get_train_valid_sampler(train_dataset, valid)
        self.train_dataloader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, 
                                            collate_fn = train_dataset.collate_fn, num_workers = 0)
        self.valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = self.batch_size, 
                                            collate_fn = valid_dataset.collate_fn, num_workers = 0)
        self.test_dataloader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, 
                                            collate_fn = test_dataset.collate_fn, shuffle = False, num_workers = 0)
    

    def get_class_counts(self):
        class_counts = torch.zeros(self.num_classes).to(self.device)

        for _, data in enumerate(self.train_dataloader):
            _, _, _, _, _, padded_labels = [d.to(self.device) for d in data]
            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]
            class_counts += torch.bincount(labels, minlength = self.num_classes)

        return class_counts
    

    def get_model(self):

        if self.dataset == 'IEMOCAP':
            self.num_classes = 6
            self.n_speakers = 2

        elif self.dataset == 'MELD':
            self.num_classes = 7
            self.n_speakers = 9
        import os
        D_m = 4096
        d_path = f'/data/jingran/MyBench/features-lianzheng/{self.dataset}/features_utt_all'
        a_model = 'chinese-hubert-large-UTT' if self.dataset == 'IEMOCAP' else 'wavlm-base-UTT'
        v_model = 'clip-vit-large-patch14-UTT'
        sample = 'Ses01F_impro01_F000.npy' if self.dataset == 'IEMOCAP' else 'test_dia0_utt0.npy'
        with open(os.path.join(d_path, a_model, sample), 'rb') as fl:
            a = np.load(fl)
        D_m_audio = a.shape[0]
        with open(os.path.join(d_path, v_model, sample), 'rb') as fl:
            a = np.load(fl)
        D_m_visual = a.shape[0]
        roberta_dim = D_m
        # D_m_audio, D_m_visual = D_m, D_m
        listener_state = False
        D_e = self.model_dim 
        D_p = self.model_dim
        D_g = self.model_dim
        D_h = self.model_dim
        D_a = self.model_dim 
        context_attention = 'simple'
        hidden_dim = self.hidden_dim
        dropout_rate = self.dropout_rate 
        num_layers = self.num_layers
        num_heads = self.num_heads

        self.model = MultiEMO(self.dataset, self.multi_attn_flag, roberta_dim, hidden_dim, dropout_rate, num_layers, 
                                    self.model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, 
                                    D_h, self.num_classes, self.n_speakers, 
                                    listener_state, context_attention, D_a, self.dropout_rec, self.device).to(self.device)

    def get_loss(self):
        class_counts = self.get_class_counts()
        self.SWFC_loss = SampleWeightedFocalContrastiveLoss(self.temp_param, self.focus_param, 
                                                            self.sample_weight_param, self.dataset, class_counts, self.device)
        self.HGR_loss = SoftHGRLoss()
        self.CE_loss = nn.CrossEntropyLoss()
    

    def get_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.95, patience = 10, threshold = 1e-6, verbose = True)
    

    def train_or_eval_model_per_epoch(self, dataloader, train = True):
        if train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        total_SWFC_loss, total_HGR_loss, total_CE_loss = 0.0, 0.0, 0.0
        all_labels, all_preds = [], []
        all_probs = []
        senti, totvid = {}, 0
        for _, data in enumerate(dataloader):
            if train:
                self.optimizer.zero_grad() 

            padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels = [d.to(self.device) for d in data]
            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]

            fused_text_features, fused_audio_features, fused_visual_features, fc_outputs, mlp_outputs = \
                self.model(padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels)
            
            soft_HGR_loss = self.HGR_loss(fused_text_features, fused_audio_features, fused_visual_features)
            SWFC_loss = self.SWFC_loss(fc_outputs, labels)
            CE_loss = self.CE_loss(mlp_outputs, labels)

            loss = soft_HGR_loss * self.HGR_loss_param + SWFC_loss * self.SWFC_loss_param + CE_loss * self.CE_loss_param

            total_loss += loss.item()

            total_HGR_loss += soft_HGR_loss.item()
            total_SWFC_loss += SWFC_loss.item()
            total_CE_loss += CE_loss.item()

            if train:
                loss.backward()
                self.optimizer.step()
            
            preds = torch.argmax(mlp_outputs, dim = -1)
            cnt = 0
            for i in range(len(padded_utterance_masks)):
                vid = totvid
                totvid += 1
                for uid in range(len(padded_utterance_masks[i])):
                    if padded_utterance_masks[i][uid] != 0:
                        if vid not in senti.keys():
                            senti[vid] = {}
                        senti[vid][uid] = {'pred': preds[cnt], 'gold': labels[cnt]}
                        cnt += 1
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(mlp_outputs.detach().cpu().numpy())
        
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        rd = [all_labels, all_probs]
        avg_f1 = round(f1_score(all_labels, all_preds, average = 'weighted') * 100, 4)
        avg_acc = round(accuracy_score(all_labels, all_preds) * 100, 4)
        report = classification_report(all_labels, all_preds, digits = 4)

        return round(total_loss, 4), round(total_HGR_loss, 4), round(total_SWFC_loss, 4), round(total_CE_loss, 4), avg_f1, avg_acc, report, senti# rd

    def result_analysis(self, senti):
        preds_shift, preds_length, preds_len = [[] for _ in range(2)], [[] for _ in range(3)], [[] for _ in range(1000)]
        labels_shift, labels_length, labels_len = [[] for _ in range(2)], [[] for _ in range(3)], [[] for _ in range(1000)]
        preds, labels, mx = [], [], 0
        blk = 11 if self.dataset == 'MELD' else 30
        for vid in senti.keys():
            rs = senti[vid]
            for uid in rs.keys():
                i = uid
                mx = max(mx, i+1)
                pred, label = rs[uid]['pred'].cpu().item(), rs[uid]['gold'].cpu().item()
                preds.append(pred)
                labels.append(label)
                flag_shift = 0
                if i > 0:
                    if label != rs[i - 1]['gold']:
                        flag_shift = 1
                preds_shift[flag_shift].append(pred)
                labels_shift[flag_shift].append(label)
                if i <= blk:
                    preds_length[0].append(pred)
                    labels_length[0].append(label)
                elif i <= blk * 2:
                    preds_length[1].append(pred)
                    labels_length[1].append(label)
                else:
                    preds_length[2].append(pred)
                    labels_length[2].append(label)
                preds_len[i].append(pred)
                labels_len[i].append(label)
        print('MAX length:', mx)
        preds = np.stack(preds)
        labels = np.stack(labels)
        preds_shift[0] = np.stack(preds_shift[0])
        labels_shift[0] = np.stack(labels_shift[0])
        preds_shift[1] = np.stack(preds_shift[1])
        labels_shift[1] = np.stack(labels_shift[1])
        for i in range(3):
            preds_length[i] = np.stack(preds_length[i])
            labels_length[i] = np.stack(labels_length[i])
        ret = {}
        ret['report'] = classification_report(labels, preds, digits=4)
        ret['WF1'] = round(f1_score(labels,preds, average='weighted')*100, 2)
        ret['nES'] = round(f1_score(labels_shift[0],preds_shift[0], average='weighted')*100, 2)
        ret['ES'] = round(f1_score(labels_shift[1],preds_shift[1], average='weighted')*100, 2)
        ret['short'] = round(f1_score(labels_length[0],preds_length[0], average='weighted')*100, 2)
        ret['medium'] = round(f1_score(labels_length[1],preds_length[1], average='weighted')*100, 2)
        ret['long'] = round(f1_score(labels_length[2],preds_length[2], average='weighted')*100, 2)
        ret['len'] = [round(f1_score(labels_len[i],preds_len[i], average='weighted')*100, 2) for i in range(mx)]
        return ret


    def train_or_eval_linear_model(self):
        # best_rd = None
        best_senti = None
        for e in range(self.num_epochs):
            train_loss, train_HGR_loss, train_SWFC_loss, train_CE_loss, train_f1, train_acc, _, _ = self.train_or_eval_model_per_epoch(self.train_dataloader, train = True)
            with torch.no_grad():
                valid_loss, valid_HGR_loss, valid_SWFC_loss, valid_CE_loss,valid_f1, valid_acc, _, _ = self.train_or_eval_model_per_epoch(self.valid_dataloader, train = False)
                test_loss, test_HGR_loss, test_SWFC_loss, test_CE_loss, test_f1, test_acc, test_report, test_senti = self.train_or_eval_model_per_epoch(self.test_dataloader, train = False)
            print('Epoch {}, train loss: {}, train HGR loss: {}, train SWFC loss: {}, train CE loss: {}, train f1: {}, train acc: {}'.format(e + 1, train_loss, train_HGR_loss, train_SWFC_loss, train_CE_loss, train_f1, train_acc))
            print('Epoch {}, valid loss: {}, valid HGR loss: {}, valid SWFC loss: {}, valid CE loss: {}, valid f1: {}, valid acc: {}'.format(e + 1, valid_loss, valid_HGR_loss, valid_SWFC_loss, valid_CE_loss, valid_f1, valid_acc))
            print('Epoch {}, test loss: {}, test HGR loss: {}, test SWFC loss: {}, test CE loss: {}, test f1: {}, test acc: {}, '.format(e + 1, test_loss, test_HGR_loss, test_SWFC_loss, test_CE_loss, test_f1, test_acc))
            print(test_report)   

            self.scheduler.step(valid_loss)

            if test_f1 >= self.best_test_f1:
                self.best_test_f1 = test_f1
                self.best_epoch = e + 1
                self.best_test_report = test_report
                # best_rd = test_rd
                best_senti = test_senti
                
        print('Best test f1: {} at epoch {}'.format(self.best_test_f1, self.best_epoch))
        print(self.best_test_report)
        import json
        with open(f"analysis/{self.dataset}_{self.fea_model}_{args.seed}.json", 'w') as fl:
            json.dump(self.result_analysis(best_senti), fl, indent=4)



def get_args():
    parser = OptionParser()
    parser.add_option('--dataset', dest = 'dataset', default = 'MELD', type = 'str', help = 'MELD or IEMOCAP')
    parser.add_option('--batch_size', dest = 'batch_size', default = 64, type = 'int', help = '64 for IEMOCAP and 100 for MELD')
    parser.add_option('--num_epochs', dest = 'num_epochs', default = 100, type = 'int', help = 'number of epochs')
    parser.add_option('--learning_rate', dest = 'learning_rate', default = 0.0001, type = 'float', help = 'learning rate')
    parser.add_option('--weight_decay', dest = 'weight_decay', default = 0.00001, type = 'float', help = 'weight decay parameter')
    parser.add_option('--num_layers', dest = 'num_layers', default = 6, type = 'int', help = 'number of layers in MultiAttn')
    parser.add_option('--model_dim', dest = 'model_dim', default = 256, type = 'int', help = 'model dimension in MultiAttn')
    parser.add_option('--num_heads', dest = 'num_heads', default = 4, type = 'int', help = 'number of heads in MultiAttn')
    parser.add_option('--hidden_dim', dest = 'hidden_dim', default = 1024, type = 'int', help = 'hidden dimension in MultiAttn')
    parser.add_option('--dropout_rate', dest = 'dropout_rate', default = 0, type = 'float', help = 'dropout rate')
    parser.add_option('--dropout_rec', dest = 'dropout_rec', default = 0, type = 'float', help = 'dropout rec')
    parser.add_option('--temp_param', dest = 'temp_param', default = 0.8, type = 'float', help = 'temperature parameter of SWFC loss')
    parser.add_option('--focus_param', dest = 'focus_param', default = 2.0, type = 'float', help = 'focusing parameter of SWFC loss')
    parser.add_option('--sample_weight_param', dest = 'sample_weight_param', default = 0.8, type = 'float', help = 'sample-weight parameter of SWFC loss')
    parser.add_option('--SWFC_loss_param', dest = 'SWFC_loss_param', default = 0.4, type = 'float', help = 'coefficient of SWFC loss')
    parser.add_option('--HGR_loss_param', dest = 'HGR_loss_param', default = 0.3, type = 'float', help = 'coefficient of Soft-HGR loss')
    parser.add_option('--CE_loss_param', dest = 'CE_loss_param', default = 0.3, type = 'float', help = 'coefficient of Cross Entropy loss')
    parser.add_option('--multi_attn_flag', dest = 'multi_attn_flag', default = True, help = 'Multimodal fusion')
    parser.add_option('--seed', dest = 'seed', type=int, default=2021, help='random seed')
    parser.add_option('--fea_model', dest = 'fea_model', type=str, help='feature model dir')
    (options, _) = parser.parse_args()

    return options




def set_seed(seed):
    np.random.seed(seed) 
    random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    model_dim = args.model_dim
    num_heads = args.num_heads
    hidden_dim = args.hidden_dim
    dropout_rate = args.dropout_rate
    dropout_rec = args.dropout_rec
    temp_param = args.temp_param
    focus_param = args.focus_param
    sample_weight_param = args.sample_weight_param
    SWFC_loss_param = args.SWFC_loss_param
    HGR_loss_param = args.HGR_loss_param
    CE_loss_param = args.CE_loss_param
    multi_attn_flag = args.multi_attn_flag
    seed = args.seed
    fea_model = args.fea_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # seed = 2023 
    set_seed(seed)

    multiemo_train = TrainMultiEMO(dataset, batch_size, num_epochs, learning_rate, 
                                   weight_decay, num_layers, model_dim, num_heads, hidden_dim, 
                                   dropout_rate, dropout_rec,temp_param, focus_param, sample_weight_param, 
                                   SWFC_loss_param, HGR_loss_param, CE_loss_param, multi_attn_flag, device, fea_model)
    multiemo_train.train_or_eval_linear_model()

        