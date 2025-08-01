import pickle
import os
import argparse
import torch
import os

import numpy as np

from sklearn import metrics
from tqdm import tqdm

import corect

log = corect.utils.get_logger()


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def main(args):
    data = load_pkl(f"data/{args.dataset}/data_{args.dataset}.pkl")

    model_dict = torch.load(
        f"model_checkpoints/{args.dataset}/"
        + "model"
        + ".pt",
    )
    stored_args = model_dict["args"]
    model = model_dict["state_dict"]
    dataset_label_dict = {
            "iemocap": ["hap", "sad", "neu", "ang", "exc", "fru"],
            "iemocap_4": ["hap", "sad", "neu", "ang"],
            "mosei": ["Negative", "Positive"],
        }
    
    testset = corect.Dataset(data["test"], stored_args)
    idx_labels = dataset_label_dict[args.dataset]

    test = True
    with torch.no_grad():
        golds = []
        preds = []
        for idx in tqdm(range(len(testset)), desc="test" if test else "dev"):
            data = testset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(stored_args.device)
            y_hat = model(data)

            preds.append(y_hat.detach().to("cpu"))

        golds = torch.cat(golds, dim=-1).numpy()
        preds = torch.cat(preds, dim=-1).numpy()
        f1 = metrics.f1_score(golds, preds, average="weighted")

        if test:
            print(metrics.classification_report(golds, preds, digits=4))

            print(f"F1 Score: {f1}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="eval.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="iemocap",
        choices=["iemocap", "meld"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    parser.add_argument("--device", type=str, default="cpu", help="Computing device.")

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "at", "atv"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    args = parser.parse_args()
    main(args)
