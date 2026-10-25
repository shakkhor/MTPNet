import argparse
import os
from argparse import Namespace

import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.tools import string_split
import json
from exp.exp_main import wandb


def main():
    parser = argparse.ArgumentParser(description='TSMAE')
    parser.add_argument('--data', type=str, required=False, default='ETTh1',
                        help='name of dataset')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='SVPNet_Linear', help='model name, options: [SVPNet, SVPNet_Linear, Dozerformer, Dozerformer_Linear]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

    parser.add_argument('--root_path', type=str, default='./data/dataset/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file, ignored for the named datasets below')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length for encoder, look back window')
    parser.add_argument('--label_len', type=int, default=96, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length, horizon')
    parser.add_argument('--features', type=str, default='M', choices=['S', 'M'],
                        help='features S is univariate, M is multivariate')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--data_split', type=str, default='0.6,0.2,0.2',
                        help='train/val/test split, can be ratio or number, ignored for the named datasets below')

    parser.add_argument('--use_gpu', type=int, default=1, help='1 uses the gpu when one is available, 0 forces cpu')
    parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D), ignored for the named datasets below')
    parser.add_argument('--embed_dim', type=int, default=8, help='encoder input size, also used for the decoder')


    parser.add_argument('--n_heads', type=int, default=4, help='number of multihead attention')
    parser.add_argument('--encoder_depth', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--decoder_depth', type=int, default=1, help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of MLP in transformer')
    parser.add_argument('--patch_size', type=str, default='12, 24', help='patch sizes of hierarchical architecture, one level per size')
    parser.add_argument('--trend_patch_size', type=str, default='24', help='patch sizes of Trend hierarchical architecture')
    parser.add_argument('--moving_avg', type=str, default='13, 17', help='window size of moving average')

    parser.add_argument('--seed', type=int, default=2023, help='Random Seed')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--loss', type=str, default='L1', help='dropout')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0', help='multiple gpu')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')

    # DozerAttention parameters
    parser.add_argument('--local_window', type=int, default=7, help='The size of local window')
    # this should be determined with considering the total lengths of the patches
    parser.add_argument('--stride', type=int, default=24,
                        help='The stride interval sparse attention. If set to 24, interval will be 24.')
    parser.add_argument('--vary_len', type=int, default=1, help='The start varying length, if 1 input equals output')

    # Cross-variate attention (iTransformer-style) parameters, always on for SVPNet and SVPNet_Linear
    parser.add_argument('--variate_d_model', type=int, default=128, help='token dimension of variate attention')
    parser.add_argument('--variate_d_ff', type=int, default=256, help='dimension of MLP in variate attention')
    parser.add_argument('--variate_layers', type=int, default=1, help='number of variate attention layers')


    args = parser.parse_args()

    # Fixed settings, not command-line flags
    args.mode = 'finetune'
    args.output_attention = False
    args.use_amp = False

    args.patch_size = [int(i) for i in args.patch_size.split(', ')]
    args.trend_patch_size = [int(i) for i in args.trend_patch_size.split(', ')]
    args.H_depth = len(args.patch_size)
    args.moving_avg = [int(i) for i in args.moving_avg.split(', ')]
    args.decoder_embed_dim = args.embed_dim
    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1': {'data': 'ETT-data/ETTh1.csv', 'data_dim': 7, 'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
        'ETTh2': {'data': 'ETT-data/ETTh2.csv', 'data_dim': 7, 'split': [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24]},
        'ETTm1': {'data': 'ETT-data/ETTm1.csv', 'data_dim': 7, 'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},
        'ETTm2': {'data': 'ETT-data/ETTm2.csv', 'data_dim': 7, 'split': [4 * 12 * 30 * 24, 4 * 4 * 30 * 24, 4 * 4 * 30 * 24]},
        'electricity': {'data': 'electricity/electricity.csv', 'data_dim': 321, 'split': [0.7, 0.1, 0.2]},
        'Weather': {'data': 'weather/weather.csv', 'data_dim': 21, 'split': [0.7, 0.1, 0.2]},
        'ILI': {'data': 'illness/national_illness.csv', 'data_dim': 7, 'split': [0.7, 0.1, 0.2]},
        'Traffic': {'data': 'traffic/traffic.csv', 'data_dim': 862, 'split': [0.7, 0.1, 0.2]},
        'Exchange': {'data': 'exchange_rate/exchange_rate.csv', 'data_dim': 8, 'split': [0.7, 0.1, 0.2]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.data_dim = data_info['data_dim']
        args.data_split = data_info['split']
    else:
        args.data_split = string_split(args.data_split)

    # # run exp Dozerformer (paper settings for ETTh1, from upstream scripts/ETTh1.sh)
    # args.model = "Dozerformer_Linear"
    # args.patch_size = 24
    # args.stride = 7
    # args.local_window = 3
    # args.seq_len = 720
    # args.learning_rate = 5e-5

    # Model, local_window, stride, patch_size and seq_len come from the command line.
    # Dozerformer takes a single patch size, SVPNet takes a list
    if args.model.startswith('Dozerformer'):
        args.patch_size = args.patch_size[0]


    print('Args in experiment:')
    print(json.dumps(vars(args), indent=4, sort_keys=True, default=str))
    #print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = get_settings(args)
            run = start_wandb(args, setting)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if run is not None:
                run.finish()

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_settings(args)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


def start_wandb(args: Namespace, setting: str):
    # All runs go to one wandb project; without wandb or a login the run trains as before, just unlogged.
    # Account and project come from WANDB_ENTITY / WANDB_PROJECT, on/off from WANDB_MODE (online, offline, disabled)
    if wandb is None:
        print('wandb not installed, run is not logged to wandb')
        return None
    try:
        return wandb.init(project=os.environ.get('WANDB_PROJECT', 'SVPNet'), entity=os.environ.get('WANDB_ENTITY'),
                          name=setting, config=vars(args))
    except Exception as e:
        print('wandb not started ({}), run is not logged to wandb'.format(e))
        return None


def get_settings(args: Namespace) -> str:
    return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_segl{}_dm{}_Hlvl{}_nh{}_el{}_dl{}_df{}_lw{}_st{}_seed{}'.format(
        args.mode,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.patch_size,
        args.embed_dim,
        args.H_depth,
        args.n_heads,
        args.encoder_depth,
        args.decoder_depth,
        1,
        args.local_window,
        args.stride,
        args.seed)


if __name__ == "__main__":
    main()
    print('Finished Training')