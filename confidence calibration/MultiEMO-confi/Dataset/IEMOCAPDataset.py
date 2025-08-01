import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F



'''
label index mapping = {'happiness': 0, 'sadness': 1, 'neutral': 2, 'anger': 3, 'excitement': 4, 'frustration': 5}
'''
class IEMOCAPDataset(Dataset):

    def __init__(self, fea_model, split):
        # _, self.videoSpeakers, self.videoLabels, _, _, _, _, self.trainVid,\
        # self.testVid = pickle.load(open('Data/IEMOCAP/Speakers.pkl', 'rb'), encoding='latin1')

        # '''
        # Textual features are extracted using pre-trained EmoBERTa. If you want to extract textual
        # features on your own, please visit https://github.com/tae898/erc
        # '''
        # self.videoText = pickle.load(open('Data/IEMOCAP/TextFeatures.pkl', 'rb'))
        # self.videoAudio = pickle.load(open('Data/IEMOCAP/AudioFeatures.pkl', 'rb'))
        # self.videoVisual = pickle.load(open('Data/IEMOCAP/VisualFeatures.pkl', 'rb'))

        # self.trainVid = sorted(self.trainVid)
        # self.testVid = sorted(self.testVid)

        # self.keys = [x for x in (self.trainVid if train else self.testVid)]
        # self.len = len(self.keys)
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('/data/jingran/MyBench/features/IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        _, self.trainVid, self.testVid, self.devVid = pickle.load(open('/data/jingran/MyBench/features/IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        
        self.roberta1 = {vid: np.array(self.roberta1[vid]) for vid in self.roberta1.keys()}
        self.roberta2 = {vid: np.array(self.roberta2[vid]) for vid in self.roberta2.keys()}
        self.roberta3 = {vid: np.array(self.roberta3[vid]) for vid in self.roberta3.keys()}
        self.roberta4 = {vid: np.array(self.roberta4[vid]) for vid in self.roberta4.keys()}
        self.videoText = {vid: np.concatenate((self.roberta1[vid], self.roberta2[vid], self.roberta3[vid], self.roberta4[vid]),axis=1) for vid in self.roberta1.keys()}
        
        for vid in self.videoIDs.keys():
            self.videoAudio[vid], self.videoVisual[vid] = [], []
            for uid in self.videoIDs[vid]:
                data_path = '/data/jingran/MyBench/features-lianzheng/IEMOCAP/features_utt_all'
                audio_path = os.path.join(data_path, 'chinese-hubert-large-UTT', f'{uid}.npy')
                video_path = os.path.join(data_path, 'clip-vit-large-patch14-UTT', f'{uid}.npy')

                with open(audio_path, 'rb') as fl:
                    audio_feature = np.load(fl)
                self.videoAudio[vid].append(audio_feature)
                with open(video_path, 'rb') as fl:
                    video_feature = np.load(fl)
                self.videoVisual[vid].append(video_feature)

        if 'T' not in fea_model:
            self.videoText = {vid: np.zeros_like(self.videoText[vid]) for vid in self.videoText.keys()}
        if 'A' not in fea_model:
            self.videoAudio = {vid: np.zeros_like(self.videoAudio[vid]) for vid in self.videoAudio.keys()}
        if 'V' not in fea_model:
            self.videoVisual = {vid: np.zeros_like(self.videoVisual[vid]) for vid in self.videoVisual.keys()}
        if split == 'train':
            self.keys = [x for x in self.trainVid]
        elif split == 'val':
            self.keys = [x for x in self.devVid]
        else:
            self.keys = [x for x in self.testVid]

        self.len = len(self.keys)
        self.uttr_length = sum([len(_) for _ in self.videoIDs.values()])
        self.keys_dict = {}
        self.uttrs = [_2 for _1 in list([_ for _ in self.videoIDs.values()]) for _2 in _1]
        for i in range(self.uttr_length):
            self.keys_dict[self.uttrs[i]] = i
        pass


    def __getitem__(self, index):
        vid = self.keys[index]
        vid_uttr = [self.keys_dict[_] for _ in self.videoIDs[vid]]
        return torch.FloatTensor(self.videoText[vid]),\
            torch.FloatTensor(self.videoAudio[vid]),\
                torch.FloatTensor(self.videoVisual[vid]),\
                    torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]),\
                        torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                            torch.LongTensor(self.videoLabels[vid]),\
                                vid_uttr


    def __len__(self):
        return self.len


    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        output = []
        for i in dat:
            temp = dat[i].values
            if i <= 3:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], padding_value = 0)) 
            elif i <= 4:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], True, padding_value = 0))
            elif i <= 5:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], True, padding_value = -1))
            else:
                output.append([temp[i] for i in range(len(temp))])

        return output