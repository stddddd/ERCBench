import argparse

from tqdm import tqdm
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import corect


log = corect.utils.get_logger()

def get_iemocap(path):
	corect.utils.set_seed(args.seed)

	video_ids, video_speakers, video_labels, videoText,\
	videoAudio, videoVisual, video_sentence, trainVid,\
	testVid = pickle.load(open('/data/jingran/MyBench/features/IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

	_, _, roberta1, roberta2, roberta3, roberta4,\
	_, train_vids, test_vids, dev_vids = pickle.load(open('/data/jingran/MyBench/features/IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
	
	for vid in video_ids.keys():
		videoAudio[vid], videoVisual[vid], videoText[vid] = [], [], []
		for uid in video_ids[vid]:
			data_path = '/data/jingran/MyBench/features-lianzheng/IEMOCAP/features_utt_all'
			audio_path = os.path.join(data_path, 'wavlm-base-UTT', f'{uid}.npy')
			video_path = os.path.join(data_path, 'manet_UTT', f'{uid}.npy')
			text_path = os.path.join(data_path, 'paraphrase-multilingual-mpnet-base-v2-UTT', f'{uid}.npy')

			with open(audio_path, 'rb') as fl:
				audio_feature = np.load(fl)
			videoAudio[vid].append(audio_feature)
			with open(video_path, 'rb') as fl:
				video_feature = np.load(fl)
			videoVisual[vid].append(video_feature)
			with open(text_path, 'rb') as fl:
				text_feature = np.load(fl)
			videoText[vid].append(text_feature)
	
	with open("/data/jingran/MyBench/lab_topic/IEMOCAP_topics.pkl", 'rb') as fl:
		IEMOCAP_topic = pickle.load(fl)
	newtrainIds, newvalidIds, newtvIds, newtestIds, tag = [], [], [], [], 0
	smask = {}
	for vid in video_ids.keys():
		topic = IEMOCAP_topic[vid]
		if topic == 2 or topic == 3:
			newtvIds.append(vid)
			smask[vid] = [1]*len(video_labels[vid])
		elif topic == 4:
			newtestIds.append(vid)
			if path == 'seen':
				smask[vid] = [0]*len(video_labels[vid])
			else:
				smask[vid] = [1]*len(video_labels[vid])
		else:
			if tag == 1:
				newtvIds.append(vid)
				smask[vid] = [1]*len(video_labels[vid])
				tag = 0
			else:
				newtestIds.append(vid)
				if path == 'seen':
					smask[vid] = [1]*len(video_labels[vid])
				else:
					smask[vid] = [0]*len(video_labels[vid])
				tag += 1
	for vid in newtvIds:
		if tag == 9:
			newvalidIds.append(vid)
			tag = 0
		else:
			newtrainIds.append(vid)
			tag += 1

	train, dev, test = [], [], []

	for vid in tqdm(newtrainIds, desc="train"):
		train.append(
			{
				"vid" : vid,
				"speakers" : video_speakers[vid],
				"labels" : video_labels[vid],
				"audio" : videoAudio[vid],
				"visual" : videoVisual[vid],
				"text": videoText[vid],
				"sentence" : video_sentence[vid],
				"smask" : smask[vid],
			}
		)
	for vid in tqdm(newvalidIds, desc="dev"):
		dev.append(
			{
				"vid" : vid,
				"speakers" : video_speakers[vid],
				"labels" : video_labels[vid],
				"audio" : videoAudio[vid],
				"visual" : videoVisual[vid],
				"text": videoText[vid],
				"sentence" : video_sentence[vid],
				"smask" : smask[vid],
			}
		)
	for vid in tqdm(newtestIds, desc="test"):
		test.append(
			{
				"vid" : vid,
				"speakers" : video_speakers[vid],
				"labels" : video_labels[vid],
				"audio" : videoAudio[vid],
				"visual" : videoVisual[vid],
				"text": videoText[vid],
				"sentence" : video_sentence[vid],
				"smask" : smask[vid],
			}
		)
	# log.info("train vids:")
	# log.info(sorted(train_vids))
	# log.info("dev vids:")
	# log.info(sorted(dev_vids))
	# log.info("test vids:")
	# log.info(sorted(test_vids))

	return train, dev, test

import torch
def get_meld(path):
	corect.utils.set_seed(args.seed)

	videoIDs, video_speakers, video_labels, videoText,\
	videoAudio, videoVisual, video_sentence, trainVid,\
	testVid, _ = pickle.load(open('/data/jingran/MyBench/features/MELD_features/MELD_features_raw1.pkl', 'rb'))

	_, _, _, roberta1, roberta2, roberta3, roberta4, \
		_, train_vids, test_vids, dev_vids \
		= pickle.load(open("/data/jingran/MyBench/features/MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')
	
	for vid in videoIDs.keys():
		videoAudio[vid], videoVisual[vid], videoText[vid] = [], [], []
		for i in videoIDs[vid]:
			if vid < 1039:
				uid = f'train_dia{vid}_utt{i}'
			elif vid < 1153:
				uid = f'val_dia{vid-1039}_utt{i}'
			else:
				uid = f'test_dia{vid-1153}_utt{i}'

			data_path = '/data/jingran/MyBench/features-lianzheng/MELD/features_utt_all'
			audio_path = os.path.join(data_path, 'whisper-base-UTT', f'{uid}.npy')
			video_path = os.path.join(data_path, 'clip-vit-large-patch14-UTT', f'{uid}.npy')
			text_path = os.path.join(data_path, 'paraphrase-multilingual-mpnet-base-v2-UTT', f'{uid}.npy')

			with open(audio_path, 'rb') as fl:
				audio_feature = np.load(fl)
			videoAudio[vid].append(audio_feature)
			with open(video_path, 'rb') as fl:
				video_feature = np.load(fl)
			videoVisual[vid].append(video_feature)
			with open(text_path, 'rb') as fl:
				text_feature = np.load(fl)
			videoText[vid].append(text_feature)
	
	with open("/data/jingran/MyBench/lab_topic/MELD_topics.pkl", 'rb') as fl:
		MELD_topic = pickle.load(fl)
	newtrainIds, newvalidIds, newtvIds, newtestIds, tag = [], [], [], [], 0
	smask = {}
	for vid in videoIDs.keys():
		topic = MELD_topic[vid]
		if topic == 0:
			newtestIds.append(vid)
			if path == 'seen':
				smask[vid] = [0]*len(video_labels[vid])
			else:
				smask[vid] = [1]*len(video_labels[vid])
		else:
			if tag == 1:
				newtvIds.append(vid)
				smask[vid] = [1]*len(video_labels[vid])
				tag = 0
			else:
				newtestIds.append(vid)
				if path == 'seen':
					smask[vid] = [1]*len(video_labels[vid])
				else:
					smask[vid] = [0]*len(video_labels[vid])
				tag += 1
	for vid in newtvIds:
		if tag == 9:
			newvalidIds.append(vid)
			tag = 0
		else:
			newtrainIds.append(vid)
			tag += 1

	train, dev, test = [], [], []

	for vid in tqdm(newtrainIds, desc="train"):
		train.append(
			{
				"vid" : vid,
				"speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
				"labels" : video_labels[vid],
				"audio" : videoAudio[vid],
				"visual" : videoVisual[vid],
				"text": videoText[vid],
				"sentence" : video_sentence[vid],
				"smask" : smask[vid],
			}
		)
	for vid in tqdm(newvalidIds, desc="dev"):
		dev.append(
			{
				"vid" : vid,
				"speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
				"labels" : video_labels[vid],
				"audio" : videoAudio[vid],
				"visual" : videoVisual[vid],
				"text": videoText[vid],
				"sentence" : video_sentence[vid],
				"smask" : smask[vid],
			}
		)
	for vid in tqdm(newtestIds, desc="test"):
		test.append(
			{
				"vid" : vid,
				"speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
				"labels" : video_labels[vid],
				"audio" : videoAudio[vid],
				"visual" : videoVisual[vid],
				"text": videoText[vid],
				"sentence" : video_sentence[vid],
				"smask" : smask[vid],
			}
		)
	# log.info("train vids:")
	# log.info(sorted(train_vids))
	# log.info("dev vids:")
	# log.info(sorted(dev_vids))
	# log.info("test vids:")
	# log.info(sorted(test_vids))

	return train, dev, test


def main(args):
	if args.dataset == "iemocap":
		train, dev, test = get_iemocap(args.fea_model)
		data = {"train": train, "dev": dev, "test": test}
		corect.utils.save_pkl(data, args.data_root + "/data/iemocap/data_iemocap_m3.pkl")
	if args.dataset == "meld":
		train, dev, test = get_meld(args.fea_model)
		data = {"train": train, "dev": dev, "test": test}
		corect.utils.save_pkl(data, args.data_root + "/data/meld/data_meld_m3.pkl")


	log.info("number of train samples: {}".format(len(train)))
	log.info("number of dev samples: {}".format(len(dev)))
	log.info("number of test samples: {}".format(len(test)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="preprocess.py")
	parser.add_argument(
		"--dataset",
		type=str,
		required=True,
		choices=["iemocap", "meld"],
		help="Dataset name.",
	)

	parser.add_argument(
		"--data_dir", type=str, default="./data", help="Dataset directory"
	)

	parser.add_argument(
		"--data_root",
		type=str,
		default=".",
	)
	parser.add_argument("--seed", type=int, default=24, help="Random seed.")
	parser.add_argument('--fea_model', type=str, default='albert_chinese_small-UTT', help='feature model dir')

	args = parser.parse_args()

	main(args)
