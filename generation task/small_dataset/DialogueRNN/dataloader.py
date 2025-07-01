import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np

import os

IEe2s = [1, 2, 0, 2, 1, 2]
class MELDDataset(Dataset):

    def __init__(self, path, split):

        if not split == 'test':
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
            self.testVid, _ = pickle.load(open('/home/jingran/MyBench/features/MELD_features/MELD_features_raw1.pkl', 'rb'))

            _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
                _, self.trainIds, self.testIds, self.validIds \
                = pickle.load(open("/home/jingran/MyBench/features/MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')
            self.videoLabels = pickle.load(open('/home/jingran/MyBench/features/MELD/meld_sentiment.pkl', 'rb'))
        else:
            self.videoIDs, IESpeakers, IELabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
            self.testVid = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

            _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
            _, self.trainIds, self.testIds, self.validIds = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
            self.videoSpeakers, self.videoLabels = {}, {}
            for vid in self.videoIDs.keys():
                self.videoSpeakers[vid] = []
                self.videoLabels[vid] = []
                for i in range(len(IESpeakers[vid])):
                    self.videoLabels[vid].append(IEe2s[IELabels[vid][i]])
                    sp = IESpeakers[vid][i]
                    if sp == 'M':
                        self.videoSpeakers[vid].append([1,0,0,0,0,0,0,0,0])
                    else:
                        self.videoSpeakers[vid].append([0,1,0,0,0,0,0,0,0])
        
        self.roberta1 = {vid: np.array(self.roberta1[vid]) for vid in self.roberta1.keys()}
        self.roberta2 = {vid: np.array(self.roberta2[vid]) for vid in self.roberta2.keys()}
        self.roberta3 = {vid: np.array(self.roberta3[vid]) for vid in self.roberta3.keys()}
        self.roberta4 = {vid: np.array(self.roberta4[vid]) for vid in self.roberta4.keys()}
        self.videoText = {vid: np.concatenate((self.roberta1[vid], self.roberta2[vid], self.roberta3[vid], self.roberta4[vid]),axis=1) for vid in self.roberta1.keys()}

        for vid in self.videoIDs.keys():
            self.videoAudio[vid], self.videoVisual[vid] = [], []
            for i in self.videoIDs[vid]:
                if not split == 'test':
                    if vid < 1039:
                        uid = f'train_dia{vid}_utt{i}'
                    elif vid < 1153:
                        uid = f'val_dia{vid-1039}_utt{i}'
                    else:
                        uid = f'test_dia{vid-1153}_utt{i}'

                    data_path = '/home/jingran/MyBench/features-lianzheng/MELD/features_utt_all'
                else:
                    uid = i
                    data_path = '/home/jingran/MyBench/features-lianzheng/IEMOCAP/features_utt_all'
                audio_path = os.path.join(data_path, 'whisper-base-UTT', f'{uid}.npy')
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
        self.n_classes = 3
        self.len = len(self.keys)


    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.LongTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class IEMOCAPDataset(Dataset):

    def __init__(self, path, split):

        if not split == 'test':
            self.videoIDs, IESpeakers, IELabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
            self.testVid = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

            _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
            _, self.trainVid, self.testVid, self.devVid = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
            self.videoSpeakers, self.videoLabels = {}, {}
            for vid in self.videoIDs.keys():
                self.videoSpeakers[vid] = []
                self.videoLabels[vid] = []
                for i in range(len(IESpeakers[vid])):
                    self.videoLabels[vid].append(IEe2s[IELabels[vid][i]])
                    sp = IESpeakers[vid][i]
                    if sp == 'M':
                        self.videoSpeakers[vid].append([1,0,0,0,0,0,0,0,0])
                    else:
                        self.videoSpeakers[vid].append([0,1,0,0,0,0,0,0,0])
        else:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
            self.testVid, _ = pickle.load(open('/home/jingran/MyBench/features/MELD_features/MELD_features_raw1.pkl', 'rb'))

            _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
                _, self.trainIds, self.testIds, self.validIds \
                = pickle.load(open("/home/jingran/MyBench/features/MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')
            self.videoLabels = pickle.load(open('/home/jingran/MyBench/features/MELD/meld_sentiment.pkl', 'rb'))
        
        self.roberta1 = {vid: np.array(self.roberta1[vid]) for vid in self.roberta1.keys()}
        self.roberta2 = {vid: np.array(self.roberta2[vid]) for vid in self.roberta2.keys()}
        self.roberta3 = {vid: np.array(self.roberta3[vid]) for vid in self.roberta3.keys()}
        self.roberta4 = {vid: np.array(self.roberta4[vid]) for vid in self.roberta4.keys()}
        self.videoText = {vid: np.concatenate((self.roberta1[vid], self.roberta2[vid], self.roberta3[vid], self.roberta4[vid]),axis=1) for vid in self.roberta1.keys()}
        
        for vid in self.videoIDs.keys():
            self.videoAudio[vid], self.videoVisual[vid] = [], []
            for i in self.videoIDs[vid]:
                if not split == 'test':
                    uid = i
                    data_path = '/home/jingran/MyBench/features-lianzheng/IEMOCAP/features_utt_all'
                else:
                    if vid < 1039:
                        uid = f'train_dia{vid}_utt{i}'
                    elif vid < 1153:
                        uid = f'val_dia{vid-1039}_utt{i}'
                    else:
                        uid = f'test_dia{vid-1153}_utt{i}'
                    data_path = '/home/jingran/MyBench/features-lianzheng/MELD/features_utt_all'
                audio_path = os.path.join(data_path, 'whisper-base-UTT', f'{uid}.npy')
                video_path = os.path.join(data_path, 'manet_UTT', f'{uid}.npy')

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
        self.n_classes = 3
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

