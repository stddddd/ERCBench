import argparse

from tqdm import tqdm
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import corect


log = corect.utils.get_logger()

def get_iemocap(fea_model):
    corect.utils.set_seed(args.seed)
    IEe2s = [1, 2, 0, 2, 1, 2]
    video_ids, IESpeakers, IELabels, videoText,\
    videoAudio, videoVisual, video_sentence, _,\
    _ = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

    _, _, _, _, _, _,\
    _, train_vids, _, dev_vids = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
    video_speakers, videoLabels = {}, {}
    for vid in video_ids.keys():
        video_speakers[vid] = []
        videoLabels[vid] = []
        for i in range(len(IESpeakers[vid])):
            videoLabels[vid].append(IEe2s[IELabels[vid][i]])
            sp = IESpeakers[vid][i]
            if sp == 'M':
                video_speakers[vid].append([1,0,0,0,0,0,0,0,0])
            else:
                video_speakers[vid].append([0,1,0,0,0,0,0,0,0])
    MELDvideoIDs, MELDvideo_speakers, _, _,\
    _, _, MELDvideo_sentence, _,\
    _, _ = pickle.load(open('/home/jingran/MyBench/features/MELD_features/MELD_features_raw1.pkl', 'rb'))
    MELDvideoLabels = pickle.load(open('/home/jingran/MyBench/features/MELD/meld_sentiment.pkl', 'rb'))
        
    _, _, _, _, _, _, _, \
        _, _, test_vids, _ \
        = pickle.load(open("/home/jingran/MyBench/features/MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')


    for vid in video_ids.keys():
        videoAudio[vid], videoVisual[vid], videoText[vid] = [], [], []
        for uid in video_ids[vid]:
            data_path = '/home/jingran/MyBench/features-lianzheng/IEMOCAP/features_utt_all'
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
        
    for vid in test_vids:
        videoAudio[vid], videoVisual[vid], videoText[vid] = [], [], []
        video_speakers[vid] = MELDvideo_speakers[vid]
        video_sentence[vid] = MELDvideo_sentence[vid]
        videoLabels[vid] = MELDvideoLabels[vid]
        for i in MELDvideoIDs[vid]:
            if vid < 1039:
                uid = f'train_dia{vid}_utt{i}'
            elif vid < 1153:
                uid = f'val_dia{vid-1039}_utt{i}'
            else:
                uid = f'test_dia{vid-1153}_utt{i}'

            data_path = '/home/jingran/MyBench/features-lianzheng/MELD/features_utt_all'
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

    if 'T' not in fea_model:
        videoText = {vid: np.zeros_like(videoText[vid]) for vid in videoText.keys()}
    if 'A' not in fea_model:
        videoAudio = {vid: np.zeros_like(videoAudio[vid]) for vid in videoAudio.keys()}
    if 'V' not in fea_model:
        videoVisual = {vid: np.zeros_like(videoVisual[vid]) for vid in videoVisual.keys()}

    train, dev, test = [], [], []

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            {
                "vid" : vid,
                "speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
                "labels" : videoLabels[vid],
                "audio" : videoAudio[vid],
                "visual" : videoVisual[vid],
                "text": videoText[vid],
                "sentence" : video_sentence[vid],
            }
        )
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            {
                "vid" : vid,
                "speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
                "labels" : videoLabels[vid],
                "audio" : videoAudio[vid],
                "visual" : videoVisual[vid],
                "text": videoText[vid],
                "sentence" : video_sentence[vid],
            }
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            {
                "vid" : vid,
                "speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
                "labels" : videoLabels[vid],
                "audio" : videoAudio[vid],
                "visual" : videoVisual[vid],
                "text": videoText[vid],
                "sentence" : video_sentence[vid],
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
def get_meld(fea_model):
    corect.utils.set_seed(args.seed)

    IEe2s = [1, 2, 0, 2, 1, 2]
    video_ids, IESpeakers, IELabels, videoText,\
    videoAudio, videoVisual, video_sentence, _,\
    _ = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

    _, _, _, _, _, _,\
    _, _, test_vids, _ = pickle.load(open('/home/jingran/MyBench/features/IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
    video_speakers, videoLabels = {}, {}
    for vid in test_vids:
        video_speakers[vid] = []
        videoLabels[vid] = []
        for i in range(len(IESpeakers[vid])):
            videoLabels[vid].append(IEe2s[IELabels[vid][i]])
            sp = IESpeakers[vid][i]
            if sp == 'M':
                video_speakers[vid].append([1,0,0,0,0,0,0,0,0])
            else:
                video_speakers[vid].append([0,1,0,0,0,0,0,0,0])
    MELDvideoIDs, MELDvideo_speakers, _, _,\
    _, _, MELDvideo_sentence, _,\
    _, _ = pickle.load(open('/home/jingran/MyBench/features/MELD_features/MELD_features_raw1.pkl', 'rb'))
    MELDvideoLabels = pickle.load(open('/home/jingran/MyBench/features/MELD/meld_sentiment.pkl', 'rb'))
        
    _, _, _, _, _, _, _, \
        _, train_vids, _, dev_vids \
        = pickle.load(open("/home/jingran/MyBench/features/MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')


    for vid in test_vids:
        videoAudio[vid], videoVisual[vid], videoText[vid] = [], [], []
        for uid in video_ids[vid]:
            data_path = '/home/jingran/MyBench/features-lianzheng/IEMOCAP/features_utt_all'
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
        
    for vid in MELDvideoIDs.keys():
        videoAudio[vid], videoVisual[vid], videoText[vid] = [], [], []
        video_speakers[vid] = MELDvideo_speakers[vid]
        video_sentence[vid] = MELDvideo_sentence[vid]
        videoLabels[vid] = MELDvideoLabels[vid]
        for i in MELDvideoIDs[vid]:
            if vid < 1039:
                uid = f'train_dia{vid}_utt{i}'
            elif vid < 1153:
                uid = f'val_dia{vid-1039}_utt{i}'
            else:
                uid = f'test_dia{vid-1153}_utt{i}'

            data_path = '/home/jingran/MyBench/features-lianzheng/MELD/features_utt_all'
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

    if 'T' not in fea_model:
        videoText = {vid: np.zeros_like(videoText[vid]) for vid in videoText.keys()}
    if 'A' not in fea_model:
        videoAudio = {vid: np.zeros_like(videoAudio[vid]) for vid in videoAudio.keys()}
    if 'V' not in fea_model:
        videoVisual = {vid: np.zeros_like(videoVisual[vid]) for vid in videoVisual.keys()}

    train, dev, test = [], [], []

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            {
                "vid" : vid,
                "speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
                "labels" : videoLabels[vid],
                "audio" : videoAudio[vid],
                "visual" : videoVisual[vid],
                "text": videoText[vid],
                "sentence" : video_sentence[vid],
            }
        )
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            {
                "vid" : vid,
                "speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
                "labels" : videoLabels[vid],
                "audio" : videoAudio[vid],
                "visual" : videoVisual[vid],
                "text": videoText[vid],
                "sentence" : video_sentence[vid],
            }
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            {
                "vid" : vid,
                "speakers" : torch.argmax(torch.Tensor(video_speakers[vid]),-1).numpy().tolist(),
                "labels" : videoLabels[vid],
                "audio" : videoAudio[vid],
                "visual" : videoVisual[vid],
                "text": videoText[vid],
                "sentence" : video_sentence[vid],
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
