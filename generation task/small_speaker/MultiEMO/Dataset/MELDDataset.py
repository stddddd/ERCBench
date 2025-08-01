import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F

main_speakers = ['Joey', 'Ross', 'Rachel', 'Phoebe', 'Monica', 'Chandler']
seen_speakers = ['Joey', 'Ross', 'Rachel' ,'Phoebe']
unseen_speakers = []
for speaker in main_speakers:
	if speaker not in seen_speakers:
		unseen_speakers.append(speaker)
'''
label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
'''
class MELDDataset(Dataset):

	def __init__(self, fea_model, split):
		# _, self.videoSpeakers, self.videoLabels, _, _, _, _, self.trainVid,\
		# self.testVid, _ = pickle.load(open('Data/MELD/Speakers.pkl', 'rb'))

		# '''
		# Textual features are extracted using pre-trained EmoBERTa. If you want to extract textual
		# features on your own, please visit https://github.com/tae898/erc
		# '''
		# self.videoText = pickle.load(open('Data/MELD/TextFeatures.pkl', 'rb'))
		# self.videoAudio = pickle.load(open('Data/MELD/AudioFeatures.pkl', 'rb'))
		# self.videoVisual = pickle.load(open('Data/MELD/VisualFeatures.pkl', 'rb'))

		# self.keys = [x for x in (self.trainVid if train else self.testVid)]
		# self.len = len(self.keys)
		self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
		self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
		self.testVid, _ = pickle.load(open('/data/jingran/MyBench/features/MELD_features/MELD_features_raw1.pkl', 'rb'))

		_, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
			_, self.trainIds, self.testIds, self.validIds \
			= pickle.load(open("/data/jingran/MyBench/features/MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')
		
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

				data_path = '/data/jingran/MyBench/features-lianzheng/MELD/features_utt_all'
				audio_path = os.path.join(data_path, 'wavlm-base-UTT', f'{uid}.npy')
				video_path = os.path.join(data_path, 'clip-vit-large-patch14-UTT', f'{uid}.npy')

				with open(audio_path, 'rb') as fl:
					audio_feature = np.load(fl)
				self.videoAudio[vid].append(audio_feature)
				with open(video_path, 'rb') as fl:
					video_feature = np.load(fl)
				self.videoVisual[vid].append(video_feature)

		meld_speaker = pickle.load(open('/data/jingran/MyBench/features/MELD/meld_speaker_new.pkl', 'rb'))
		newtrainIds, newvalidIds, newtestIds = [], [], []
		for vid in self.trainIds:
			flag = True
			for speaker in meld_speaker[vid]:
				if speaker in unseen_speakers:
					flag = False
					break
			if flag:
				newtrainIds.append(vid)
			else:
				newtestIds.append(vid)
		for vid in self.validIds:
			flag = True
			for speaker in meld_speaker[vid]:
				if speaker in unseen_speakers:
					flag = False
					break
			if flag:
				newvalidIds.append(vid)
			else:
				newtestIds.append(vid)
		for vid in self.testIds:
			flag = True
			for speaker in meld_speaker[vid]:
				if speaker in unseen_speakers:
					flag = False
					break
			if flag:
				newtrainIds.append(vid)
			else:
				newtestIds.append(vid)
		self.smask = {}
		if split == 'train':
			self.keys = [x for x in newtrainIds]
			for vid in self.keys:
				self.smask[vid] = [1]*len(meld_speaker[vid])
		elif split == 'val':
			self.keys = [x for x in newvalidIds]
			for vid in self.keys:
				self.smask[vid] = [1]*len(meld_speaker[vid])
		else:
			self.keys = [x for x in newtestIds]
			for vid in self.keys:
				self.smask[vid] = []
				for sp in meld_speaker[vid]:
					if fea_model == 'seen':
						if sp in seen_speakers:
							self.smask[vid].append(1)
						else:
							self.smask[vid].append(0)
					elif fea_model == 'unseen':
						if sp in unseen_speakers:
							self.smask[vid].append(1)
						else:
							self.smask[vid].append(0)
					elif fea_model == 'seenO':
						if sp not in unseen_speakers:
							self.smask[vid].append(1)
						else:
							self.smask[vid].append(0)
					elif fea_model == 'unseenO':
						if sp not in seen_speakers:
							self.smask[vid].append(1)
						else:
							self.smask[vid].append(0)

		self.len = len(self.keys)
		video_ids = list(self.videoIDs.values())
		self.uttr_length = sum([len(_) for _ in video_ids])
		self.keys_dict = {}

		self.uttrs = ['{}_{}'.format(_1, _2) for _1 in self.videoIDs.keys() for _2 in self.videoIDs[_1]]
		for i in range(self.uttr_length):
			self.keys_dict[self.uttrs[i]] = i
		pass

	def __getitem__(self, index):
		vid = self.keys[index]
		vid_uttr = [self.keys_dict['{}_{}'.format(vid, _)] for _ in self.videoIDs[vid]]
		return torch.FloatTensor(np.array(self.videoText[vid])),\
			torch.FloatTensor(np.array(self.videoAudio[vid])),\
				torch.FloatTensor(np.array(self.videoVisual[vid])),\
					torch.FloatTensor(np.array(self.videoSpeakers[vid])),\
						torch.FloatTensor(np.array([1] * len(self.videoLabels[vid]))),\
							torch.LongTensor(np.array(self.videoLabels[vid])),\
			   					torch.FloatTensor(self.smask[vid]),\
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
			elif i <= 6:
				output.append(pad_sequence([temp[i] for i in range(len(temp))], True, padding_value = -1))
			else:
				output.append([temp[i] for i in range(len(temp))])

		return output


