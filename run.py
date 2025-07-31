import argparse
import os
import torch
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
import random
import numpy as np
from ipdb import set_trace
 
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser()

parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
parser.add_argument('--continue_training', type=int, default=0, help='Flag to continue training')
parser.add_argument('--train_path', type=str, required=False, default='', help='the path of train file')
# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--load_prompt_path', type=str, default='', help="Path to load the prompt from previous task")
parser.add_argument('--n_ucr', type=int, default=1)
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--freeze_layers', type=int, default=1, help="Whether to freeze layers during training")
parser.add_argument('--d_ff', type=int, default=16, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--atten_dim', type=int, default=64, help='dimension of attention mechanism')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--ffn_dim', type=int, default=256, help='dimension of Feedforward')
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument('--in_channel', type=int, default=7, help='number of input channels')
parser.add_argument('--out_channel', type=int, default=7, help='number of input channels')
parser.add_argument('--num_nodes', type=int, default=100, help='number of nodes')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='')
parser.add_argument('--gpu', type=int, default=2, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='3,2,1,0', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# patching
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument("--patch_len", type=int, default=16)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--dataset_types', type=str, default='SMD,SMAP,MSL,PSM,SWAT', help='Comma-separated list of dataset types to train on')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    print('devices:',device_ids)
    args.device_ids = [int(id_) for id_ in device_ids]
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # args.gpu = args.device_ids[3]
    # print(f"Using  GPU with device ID: {args.gpu}")

print('Args in experiment:')
print(args)


Exp = Exp_Anomaly_Detection


exp = Exp(args)

dataset_types=['SMAP','PSM','MSL','SWAT','SMD','UCR']
exp.train_on_multiple_datasets(dataset_types)

torch.cuda.empty_cache()
print("Complete!")



