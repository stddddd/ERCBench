import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import os
import numpy as np

class IEMOCAPDataset(Dataset):

    def __init__(self, path, split):
        
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        _, self.trainVid, self.testVid, self.devVid = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        
        self.roberta1 = {vid: np.array(self.roberta1[vid]) for vid in self.roberta1.keys()}
        self.roberta2 = {vid: np.array(self.roberta2[vid]) for vid in self.roberta2.keys()}
        self.roberta3 = {vid: np.array(self.roberta3[vid]) for vid in self.roberta3.keys()}
        self.roberta4 = {vid: np.array(self.roberta4[vid]) for vid in self.roberta4.keys()}
        self.videoText = {vid: np.concatenate((self.roberta1[vid], self.roberta2[vid], self.roberta3[vid], self.roberta4[vid]),axis=1) for vid in self.roberta1.keys()}
        
        for vid in self.videoIDs.keys():
            self.videoAudio[vid], self.videoVisual[vid] = [], []
            for uid in self.videoIDs[vid]:
                data_path = '/home/jingran/MyBench/features-lianzheng/IEMOCAP/features_utt_all'
                audio_path = os.path.join(data_path, 'chinese-hubert-large-UTT', f'{uid}.npy')
                video_path = os.path.join(data_path, 'clip-vit-large-patch14-UTT', f'{uid}.npy')

                with open(audio_path, 'rb') as fl:
                    audio_feature = np.load(fl)
                self.videoAudio[vid].append(audio_feature)
                with open(video_path, 'rb') as fl:
                    video_feature = np.load(fl)
                self.videoVisual[vid].append(video_feature)

        if 'T' not in path:
            self.videoText = {vid: np.zeros_like(self.videoText[vid]) for vid in self.videoText.keys()}
        if 'A' not in path:
            self.videoAudio = {vid: np.zeros_like(self.videoAudio[vid]) for vid in self.videoAudio.keys()}
        if 'V' not in path:
            self.videoVisual = {vid: np.zeros_like(self.videoVisual[vid]) for vid in self.videoVisual.keys()}
        if split == 'train':
            self.keys = [x for x in self.trainVid]
        elif split == 'val':
            self.keys = [x for x in self.devVid]
        else:
            self.keys = [x for x in self.testVid]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path, split):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open('/home/jingran/MyBench/features/MELD_features/MELD_features_raw1.pkl', 'rb'))

        _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            _, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open("/home/jingran/MyBench/features/MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')
        
        self.roberta1 = {vid: np.array(self.roberta1[vid]) for vid in self.roberta1.keys()}
        self.roberta2 = {vid: np.array(self.roberta2[vid]) for vid in self.roberta2.keys()}
        self.roberta3 = {vid: np.array(self.roberta3[vid]) for vid in self.roberta3.keys()}
        self.roberta4 = {vid: np.array(self.roberta4[vid]) for vid in self.roberta4.keys()}
        self.videoText = {vid: np.concatenate((self.roberta1[vid], self.roberta2[vid], self.roberta3[vid], self.roberta4[vid]),axis=1) for vid in self.roberta1.keys()}

        for vid in self.videoIDs.keys():
            self.videoAudio[vid], self.videoVisual[vid] = [], []
            for i in self.videoIDs[vid]:
                if vid < 1039:
                    uid = f'train_dia{vid}_utt{i}'
                elif vid < 1153:
                    uid = f'val_dia{vid-1039}_utt{i}'
                else:
                    uid = f'test_dia{vid-1153}_utt{i}'

                data_path = '/home/jingran/MyBench/features-lianzheng/MELD/features_utt_all'
                audio_path = os.path.join(data_path, 'wavlm-base-UTT', f'{uid}.npy')
                video_path = os.path.join(data_path, 'clip-vit-large-patch14-UTT', f'{uid}.npy')

                with open(audio_path, 'rb') as fl:
                    audio_feature = np.load(fl)
                self.videoAudio[vid].append(audio_feature)
                with open(video_path, 'rb') as fl:
                    video_feature = np.load(fl)
                self.videoVisual[vid].append(video_feature)

        if 'T' not in path:
            self.videoText = {vid: np.zeros_like(self.videoText[vid]) for vid in self.videoText.keys()}
        if 'A' not in path:
            self.videoAudio = {vid: np.zeros_like(self.videoAudio[vid]) for vid in self.videoAudio.keys()}
        if 'V' not in path:
            self.videoVisual = {vid: np.zeros_like(self.videoVisual[vid]) for vid in self.videoVisual.keys()}

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'val':
            self.keys = [x for x in self.validIds]
        else:
            self.keys = [x for x in self.testIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.LongTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid
    
    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return  torch.FloatTensor(self.Features[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                conv

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]


class DailyDialogueDataset2(Dataset):

    def __init__(self, split, path):

        self.Speakers, self.Features, _, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.FloatTensor(self.Features[conv]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.EmotionLabels[conv])), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]
