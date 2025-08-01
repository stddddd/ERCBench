from comet_ml import Experiment, Optimizer

import argparse
import torch
import os
import corect
import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
from sklearn.metrics import f1_score, classification_report
log = corect.utils.get_logger()

def result_analysis(senti):
    preds_shift, preds_length, preds_len = [[] for _ in range(2)], [[] for _ in range(3)], [[] for _ in range(1000)]
    labels_shift, labels_length, labels_len = [[] for _ in range(2)], [[] for _ in range(3)], [[] for _ in range(1000)]
    preds, labels, mx = [], [], 0
    blk = 11 if args.dataset == 'meld' else 30
    for vid in senti.keys():
        rs = senti[vid]
        for uid in rs.keys():
            i = uid
            mx = max(mx, i+1)
            pred, label = rs[uid]['pred'], rs[uid]['gold']
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

def main(args):
    corect.utils.set_seed(args.seed)

    if args.emotion:
        args.data = os.path.join(
            args.data_root,
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + "_" + args.emotion + ".pkl",
        )
    else:
        args.data = os.path.join(
            args.data_root, args.data_dir_path, args.dataset, "data_" + args.dataset + "_m3.pkl"
        )

    # load data
    log.info("Loading data from '%s'." % args.data)

    data = corect.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = corect.Dataset(data["train"], args)
    devset = corect.Dataset(data["dev"], args)
    testset = corect.Dataset(data["test"], args)

    log.debug("Building model...")
    
    model_file = args.data_root + f"/model_checkpoints/{args.dataset}/model.pt"
    model = corect.CORECT(args).to(args.device)
    opt = corect.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = corect.Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()
    best_senti = ret[-1]
    with open(f"analysis/{args.dataset}_{args.fea_model}_{args.seed}.json", 'w') as fl:
        json.dump(result_analysis(best_senti), fl, indent=4)
    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="iemocap",
        choices=["iemocap", "meld"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    # Training parameters
    parser.add_argument(
        "--from_begin", action="store_true", help="Training from begin.", default=False
    )

    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.00025, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""",
    )
    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate.")

    # Model parameters
    parser.add_argument(
        "--wp",
        type=int,
        default=11,
        help="Past context window size. Set wp to -1 to use all the past context.",
    )
    parser.add_argument(
        "--wf",
        type=int,
        default=9,
        help="Future context window size. Set wp to -1 to use all the future context.",
    )
    
    parser.add_argument(
        "--hidden_size", type=int, default=100, help="Hidden size."
    )
    parser.add_argument(
        "--rnn",
        type=str,
        default="transformer",
        choices=["lstm", "transformer", "ffn"],
        help="Type of RNN encoder cell.",
    )

    parser.add_argument(
        "--class_weight",
        action="store_true",
        default=False,
        help="Use class weights in nll loss.",
    )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in UnimodalEncoder.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "t", "v", "at", "tv", "av", "atv"],
        help="Modalities",
    )

    parser.add_argument(
        "--gcn_conv",
        type=str,
        default="rgcn",
        # required=True,
        choices=["rgcn"],
        help="Graph convolution layer",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    
    parser.add_argument("--encoder_nlayers", type=int, default=2)
    parser.add_argument("--graph_transformer_nheads", type=int, default=7)
    parser.add_argument("--use_highway", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    parser.add_argument(
        "--log_in_comet",
        action="store_true",
        default=False,
        help="Logs the experiment data to comet.ml",
    )
    parser.add_argument(
        "--comet_api_key",
        type=str,
        help="comet api key, required for logging experiments on comet.ml",
    )
    parser.add_argument(
        "--comet_workspace",
        type=str,
        help="comet comet_workspace, required for logging experiments on comet.ml",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="data root of folder",
    )

    parser.add_argument(
        "--edge_type",
        default="temp_multi",
        choices=("temp_multi", "multi", "temp"),
        help="Choices edge contruct type",
    )

    parser.add_argument(
        "--use_speaker",
        action="store_true",
        default=False,
        help="Use speakers attribute",
    )

    parser.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_graph_transformer",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_crossmodal",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--crossmodal_nheads",
        type=int,
        default=2,
        help="number attention heads in crossmodal attention block"
    )

    parser.add_argument(
        "--num_crossmodal",
        type=int,
        default=2,
        help="number crossmodal block"
    )

    parser.add_argument(
        "--self_att_nheads",
        type=int,
        default=2,
        help="number attention heads in self attention block",
    )

    parser.add_argument(
        "--num_self_att",
        type=int,
        default=3,
        help="number self attention block"
    )

    parser.add_argument("--tag", type=str, default="normalexperiment")
    parser.add_argument('--fea_model', type=str, default='albert_chinese_small-UTT', help='feature model dir')

    args = parser.parse_args()
    import os
    import numpy as np
    dataset = 'IEMOCAP' if args.dataset == 'iemocap' else 'MELD'
    d_path = f'/home/jingran/MyBench/features-lianzheng/{dataset}/features_utt_all'
    a_model = 'wavlm-base-UTT' if dataset == 'IEMOCAP' else 'whisper-base-UTT'
    v_model = 'manet_UTT' if dataset == 'IEMOCAP' else 'clip-vit-large-patch14-UTT'
    t_model = 'paraphrase-multilingual-mpnet-base-v2-UTT'
    sample = 'Ses01F_impro01_F000.npy' if dataset == 'IEMOCAP' else 'test_dia0_utt0.npy'
    with open(os.path.join(d_path, t_model, sample), 'rb') as fl:
        a = np.load(fl)
    D_m = a.shape[0]
    with open(os.path.join(d_path, a_model, sample), 'rb') as fl:
        a = np.load(fl)
    D_audio = a.shape[0]
    with open(os.path.join(d_path, v_model, sample), 'rb') as fl:
        a = np.load(fl)
    D_visual = a.shape[0]
    args.dataset_embedding_dims = {
        args.dataset: {
            "a": D_audio,
            "t": D_m,
            "v": D_visual,
        }
    }
    log.debug(args)

    if args.log_in_comet:
        experiment = corect.Logger(
                api_key=args.comet_api_key,
                project_name="rtgraph",
                workspace=args.comet_workspace,
                auto_param_logging=False,
                auto_metric_logging=False,
                )

        experiment.add_tag(args.tag)
        experiment.log_parameters(args)
        print(experiment)

    main(args)
