"""
TUAB layer-wise feature extraction + linear probe (integrated, single-run-per-method).

Running this file once will, for a single method selected via
NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL:
  1. Extract layer-wise features for train / valid / test splits, caching them
     under a temp directory (features are never held fully in CPU/GPU memory).
  2. Run a linear-classifier probe on each of the 8 layers, one layer at a time.
  3. Delete every temp feature file once all layers have been probed.

This merges the previous feature_extraction.py (TUAB) and linear_probe.py
(originally written for KaggleERN) into one pipeline; the probe/metric logic
follows the format of the KaggleERN linear_probe.py, adapted for TUAB's
binary ("multilabel", num_outputs=1) classification setting.
"""

import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning_EEGPT import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier

import shutil
import re  # 정규표현식 모듈
from tqdm import tqdm
import gc
import pickle
import random

from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from einops import rearrange


seed = 0 + utils.get_rank()
print(f"SEED = {seed}")
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


from Modules.models.EEGPT_mcae_finetune_change_for_linear_wise_invest import EEGPTClassifier, EEGPTClassifier_rep_conv, EEGPTClassifier_rep_MULTI_SCALE_DYNAMIC_CONV


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 0. 실험 설정
# ==========================================

NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL = 'PEARL'

POOLING = []
# POOLING = [1, 2]

TARGET_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8]

FRONT = 397
KERNEL = 10
STRIDE = 5
HEADS = 1
CONV_SOFTMAX = False
CONV_BIAS = True
CONV_DROP = 0.5

KERNEL_SIZES = (5, 7, 9, 11)
REDUCTION = 2

LORA_RANK = 8
VERA_RANK = 256

if NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL == 'DORA':
    USE_DORA = True
else:
    USE_DORA = False

# feature 추출 시 배치 크기
EXTRACT_BATCH_SIZE = 500

# linear probe 학습 관련 설정
DATE = datetime.datetime.now().strftime('%y%m%d_%H%M%S') + '_TUAB_LAYER_WISE_LINEAR_PROBE'
OUTPUT_TYPE = "multilabel"  # TUAB은 이진 분류(정상/이상)라 multilabel(=binary) 형식 사용
NUM_OUTPUTS = 1
METRICS = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "cohen_kappa"]
EPOCHS = 120
PROBE_BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
HIDDEN_DIM = 16
DROPOUT_P = 0.5

TEMP_ROOT = f"/workspace/EEGPT/temp/TUAB/{NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL}"
LOG_ROOT = f"/workspace/EEGPT/log/TUAB_layerwise_probe_log/{DATE}/{NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL}_model/SEED_0"


from peft import LoraConfig, get_peft_model, VeraConfig


def apply_lora_to_eeg_encoder(model, lora_alpha=8, lora_dropout=0.05):
    """
    LitEEGPTCausal 모델의 target_encoder.blocks 내부에만 제한적으로 LoRA를 적용합니다.
    """
    target_regex = r".*blocks.*(?:qkv)"

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=lora_alpha,
        target_modules=target_regex,
        lora_dropout=lora_dropout,
        bias="none",
        use_dora=USE_DORA
    )

    model.target_encoder = get_peft_model(model.target_encoder, lora_config)

    for name, param in model.target_encoder.named_parameters():
        if "base_layer" in name or "original_module" in name:
            param.requires_grad = False
        elif any(keyword in name for keyword in ["proj", "fc1", "fc2", "norm"]):
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model


def apply_vera_to_eeg_encoder(model, vera_rank=128, vera_dropout=0.0):
    """
    LitEEGPTCausal 모델의 target_encoder.blocks 내부에만 제한적으로 VeRA를 적용합니다.
    """
    target_regex = r".*blocks.*(?:qkv)"

    vera_config = VeraConfig(
        r=VERA_RANK,
        target_modules=target_regex,
        vera_dropout=vera_dropout,
        bias="none",
    )

    model.target_encoder = get_peft_model(model.target_encoder, vera_config)

    for name, param in model.target_encoder.named_parameters():
        if "blocks" in name:
            if "vera_lambda_b" in name or "vera_lambda_d" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            if any(keyword in name for keyword in ["base_layer", "original_module", "vera_A", "vera_B"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model


def get_args():
    parser = argparse.ArgumentParser('fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')

    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module|state_dict', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_models(args):
    use_channels_names = [
        'FP1', 'FPZ', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'O2']

    if NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL == 'FINETUNE':
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    else:
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T7-REF', 'EEG T8-REF', 'EEG P7-REF', 'EEG P8-REF', 'EEG TP7-REF', 'EEG TP8-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG FT7-REF', 'EEG FT8-REF']

    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

    if NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL == 'PEARL':
        model = EEGPTClassifier_rep_conv(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(ch_names), 2368],
            use_channels_names=ch_names,
            use_chan_conv=False,
            use_mean_pooling=args.use_mean_pooling,
            kernel_size=KERNEL, conv_num_heads=HEADS, conv_weight_softmax=CONV_SOFTMAX, conv_bias=CONV_BIAS, conv_dropout=CONV_DROP, stride=STRIDE,
        )
    elif NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL == 'DYNAMICPEARL':
        model = EEGPTClassifier_rep_MULTI_SCALE_DYNAMIC_CONV(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(ch_names), 2368],
            use_channels_names=ch_names,
            use_chan_conv=False,
            use_mean_pooling=args.use_mean_pooling,
            channel_num=len(ch_names),
            kernel_sizes=KERNEL_SIZES,
            num_head=HEADS,
            conv_weight_softmax=CONV_SOFTMAX,
            conv_bias=CONV_BIAS,
            conv_dropout=CONV_DROP,
            stride=STRIDE,
            reduction=REDUCTION,
        )
    elif NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL == 'FINETUNE':
        model = EEGPTClassifier(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(use_channels_names), 2000],
            use_channels_names=use_channels_names,
            use_chan_conv=True,
            use_mean_pooling=args.use_mean_pooling,)
    else:
        model = EEGPTClassifier(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(ch_names), 2000],
            use_channels_names=ch_names,
            use_chan_conv=False,
            use_mean_pooling=args.use_mean_pooling,)

    return model


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        subject_id = file_name.split('_')[0]

        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, subject_id


def prepare_TUAB_dataset(root):
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)

    return train_dataset, test_dataset, val_dataset


def get_dataset(args):
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = prepare_TUAB_dataset("/workspace/EEGPT/dataset/TUAB/edf/processed")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "cohen_kappa"]
    return train_dataset, test_dataset, val_dataset


def load_model_checkpoint(model, opts):
    method = NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL

    if method == 'NOTRAIN':
        checkpoint = torch.load('/workspace/EEGPT/EEGPT-main/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt', map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['state_dict']
        utils.load_state_dict(model, checkpoint_model, prefix=opts.model_prefix)
    elif method == 'LINEAR':
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260808_1_SEED_0_LP_NO_CHAN_CONV/checkpoint-best.pth', map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model']
        utils.load_state_dict(model, checkpoint_model, prefix=opts.model_prefix)
    elif method == 'FINETUNE':
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260809_2_SEED_0_FT/checkpoint-best.pth', map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model']
        utils.load_state_dict(model, checkpoint_model, prefix=opts.model_prefix)
    elif method == 'PEARL':
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260809_3_SEED_0_PEARL/checkpoint-best_45_epoch.pth', map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"PEARL Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    elif method == 'DYNAMICPEARL':
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260812_2_SEED_0_PEARL_MULTI_SCALE_DYNAMIC_CONV_KERNEL_5,7,9,11_REDUCTION_2/checkpoint-best_48_epoch.pth', map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"DYNAMICPEARL Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    elif method == 'LORA':
        model = apply_lora_to_eeg_encoder(model)
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260809_4_SEED_0_LORA/checkpoint-best_13_epoch.pth', map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        lora_missing = [k for k in missing if 'lora_A' in k or 'lora_B' in k or 'lora_magnitude' in k]
        print(f"LoRA/DoRA 관련 missing keys 개수: {len(lora_missing)}")
    elif method == 'DORA':
        model = apply_lora_to_eeg_encoder(model)
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260809_5_SEED_0_DORA_only_qkv_RANK_8_NO_CHAN_CONV/checkpoint-best_5_epoch.pth', map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"DORA Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    elif method == 'VERA':
        model = apply_vera_to_eeg_encoder(model)
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260809_6_SEED_0_VERA_ONLY_QKV_RANK_256_NO_CHAN_CONV/checkpoint-best_47_epoch.pth', map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"VERA Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    else:
        print("ENTER LINEAR_OR_FINETUNE_OR_PEARL CORRECTLY")
        exit()

    return model


# ==========================================
# 1. Feature extraction (per split, cached to temp dir)
# ==========================================

def extract_and_merge_split(split_name, dataset, model, temp_dir, device, batch_size=EXTRACT_BATCH_SIZE):
    """
    한 split(train/valid/test)에 대해 모든 layer의 feature를 추출한다.
    배치 단위 feature는 temp_dir/{split}/batches 에 임시 저장했다가 layer별로
    병합한 뒤 temp_dir/{split}/layer_{n}.pt 로 저장하고 배치 파일은 삭제한다.
    (전체 feature를 CPU/GPU 메모리에 한 번에 올리지 않기 위함)
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    split_temp_dir = os.path.join(temp_dir, split_name)
    batch_temp_dir = os.path.join(split_temp_dir, "batches")
    os.makedirs(batch_temp_dir, exist_ok=True)

    y_list = []
    s_list = []

    with torch.no_grad():
        for step, data in enumerate(tqdm(loader, desc=f"[{split_name}] Extracting Features to Temp")):
            inputs, labels, subject_ids = data
            inputs = inputs.to(device)
            B = inputs.shape[0]

            features = model(inputs)

            batch_features_dict = {}
            for layer_num in TARGET_LAYERS:
                layer_idx = layer_num - 1
                feat = features[layer_idx]

                feat_reshaped = rearrange(feat, '(B N) dim1 dim2 -> B N dim1 dim2', B=B)
                feat_reshaped = feat_reshaped[..., -4:, :]

                if len(POOLING) != 0:
                    batch_features_dict[f'layer_{layer_idx}'] = feat_reshaped.mean(dim=POOLING)
                else:
                    batch_features_dict[f'layer_{layer_idx}'] = feat_reshaped

            torch.save(batch_features_dict, os.path.join(batch_temp_dir, f"batch_{step}.pt"))

            y_list.append(labels)
            s_list.extend(subject_ids)

            if step % 2 == 0:
                torch.cuda.empty_cache()

    entire_dataset_length = len(dataset)
    print(f"[{split_name}] dataset length = {entire_dataset_length}")

    y_tensor = torch.cat(y_list, dim=0)
    del y_list, batch_features_dict, feat_reshaped
    torch.cuda.empty_cache()
    gc.collect()

    num_batches = len(loader)
    layer_paths = {}

    # 배치 파일 하나에 모든 layer가 함께 저장되어 있으므로, layer마다 배치 파일을
    # 반복해서 다시 읽지 않도록 배치를 한 번만 순회하면서 모든 layer 텐서를 동시에 채운다.
    # (layer별로 두 번씩 읽던 기존 방식은 디스크 I/O가 8배 이상 불필요하게 발생했다)
    print(f"[{split_name}] Pre-allocating tensors for {len(TARGET_LAYERS)} layers...")
    first_batch = torch.load(os.path.join(batch_temp_dir, "batch_0.pt"), map_location='cpu', weights_only=False)
    layer_tensors = {}
    for layer_num in TARGET_LAYERS:
        layer_idx = layer_num - 1
        feat = first_batch[f'layer_{layer_idx}']
        layer_tensors[layer_num] = torch.empty((entire_dataset_length, *feat.shape[1:]), dtype=feat.dtype)
    del first_batch
    gc.collect()

    current_idx = 0
    for step in tqdm(range(num_batches), desc=f"[{split_name}] Merging all layers"):
        batch_data = torch.load(os.path.join(batch_temp_dir, f"batch_{step}.pt"), map_location='cpu', weights_only=False)

        rows = batch_data[f'layer_{TARGET_LAYERS[0] - 1}'].shape[0]
        for layer_num in TARGET_LAYERS:
            layer_idx = layer_num - 1
            layer_tensors[layer_num][current_idx: current_idx + rows] = batch_data[f'layer_{layer_idx}']
        current_idx += rows

        del batch_data

    gc.collect()
    torch.cuda.empty_cache()

    for layer_num in TARGET_LAYERS:
        layer_tensor = layer_tensors.pop(layer_num)
        print(f"[{split_name}] Final shape for Layer {layer_num} = {layer_tensor.shape}")

        save_dict = {"x": layer_tensor, "y": y_tensor, "s": s_list}
        layer_path = os.path.join(split_temp_dir, f"layer_{layer_num}.pt")
        torch.save(save_dict, layer_path)
        layer_paths[layer_num] = layer_path

        del layer_tensor, save_dict
        gc.collect()

    torch.cuda.empty_cache()

    for step in range(num_batches):
        batch_file = os.path.join(batch_temp_dir, f"batch_{step}.pt")
        if os.path.exists(batch_file):
            os.remove(batch_file)

    print(f"[{split_name}] feature extraction & merge complete.")
    return layer_paths


# ==========================================
# 2. Linear probe (StructuredProbe, same format as KaggleERN linear_probe.py)
# ==========================================

class StructuredProbe(nn.Module):
    """
    main 실험에서 실제로 학습된 linear_probe1/2와 동일한 factorized 구조를 갖되,
    가중치는 불러오지 않고 매번 무작위 초기화해서 새로 학습하는 "공정한" probe.
    """
    def __init__(self, embed_num, embed_dim, num_time_patch, hidden_dim, num_outputs, dropout_p=0.5):
        super().__init__()
        self.probe1 = nn.Linear(embed_num * embed_dim, hidden_dim)
        self.probe2 = nn.Linear(num_time_patch * hidden_dim, num_outputs)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # x: (B, N_time_patch, EMBED_NUM, D)
        B, N, E, D = x.shape
        h = self.dropout(x.reshape(B, N, E * D))
        h = self.probe1(h)      # (B, N_time_patch, hidden_dim)
        h = h.reshape(B, -1)    # (B, N_time_patch * hidden_dim)
        h = self.probe2(h)      # (B, num_outputs)
        return h


def compute_loss(logits, y, output_type):
    if output_type == "multiclass":
        loss = F.cross_entropy(logits, y, reduction="mean")
    elif output_type == "multilabel":
        loss = F.binary_cross_entropy_with_logits(logits, y.float().view_as(logits), reduction="mean")
    elif output_type == "regression":
        loss = F.mse_loss(logits, y, reduction="mean")
    else:
        raise NotImplementedError(f"Unknown output_type: {output_type}")

    return loss


def calculate_metrics(all_logits, all_targets, output_type, metrics):
    y_true = all_targets.cpu().numpy()

    if output_type == "multiclass":
        y_pred = F.softmax(all_logits, dim=1).cpu().numpy()
        metrics = multiclass_metrics_fn(y_true, y_pred, metrics=metrics)
    elif output_type == "multilabel":
        y_pred = torch.sigmoid(all_logits).cpu().numpy()
        if y_pred.ndim > 1 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        metrics = binary_metrics_fn(y_true, y_pred, metrics=metrics)
    else:
        raise NotImplementedError(f"Metrics for {output_type} are not configured.")

    return metrics


def load_layer_dataset(feature_path):
    """
    extract_and_merge_split()이 temp_dir에 저장해둔 layer별 feature(.pt)를 불러와
    TensorDataset으로 반환한다. feature는 추출 단계에서 이미 마지막 summary_token
    (EMBED_NUM=4)만 슬라이싱된 상태로 저장되어 있으므로 추가 슬라이싱은 필요 없다.
    """
    if not os.path.exists(feature_path):
        print(f"Error: {feature_path} 경로에 파일이 존재하지 않습니다. 스킵합니다.")
        exit()

    print(f"Loading features from {feature_path}...")
    data = torch.load(feature_path, map_location='cpu', weights_only=False)

    x = data['x'].float()
    y = data['y']

    return TensorDataset(x, y)


def run_layer_experiment(
    layer_num,
    train_path,
    valid_path,
    test_path,
    log_root,
    output_type=OUTPUT_TYPE,
    num_outputs=NUM_OUTPUTS,
    metrics=METRICS,
    epochs=EPOCHS,
    batch_size=PROBE_BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P,
):
    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> [Layer {layer_num}] 실험 시작 (Device: {probe_device})")

    train_dataset = load_layer_dataset(train_path)
    val_dataset = load_layer_dataset(valid_path)
    test_dataset = load_layer_dataset(test_path)
    print(f"Train: {len(train_dataset)}, Valid: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    _, num_time_patch, embed_num, embed_dim = train_dataset.tensors[0].shape
    print(f"[Layer {layer_num}] feature shape = (N_time_patch={num_time_patch}, "
        f"EMBED_NUM={embed_num}, D={embed_dim})")
    probe = StructuredProbe(
        embed_num=embed_num,
        embed_dim=embed_dim,
        num_time_patch=num_time_patch,
        hidden_dim=hidden_dim,
        num_outputs=num_outputs,
        dropout_p=dropout_p,
    ).to(probe_device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    log_dir = os.path.join(log_root, f"layer_{layer_num}")
    writer = SummaryWriter(log_dir=log_dir)

    best_val_balanced_acc = float("-inf")
    early_stopping_counter = 0

    for epoch in range(epochs):
        probe.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(probe_device), batch_y.to(probe_device)

            optimizer.zero_grad()
            logits = probe(batch_X)
            loss = compute_loss(logits, batch_y, output_type)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        probe.eval()
        val_loss = 0.0
        val_logits_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(probe_device), batch_y.to(probe_device)
                logits = probe(batch_X)
                loss = compute_loss(logits, batch_y, output_type)
                val_loss += loss.item() * batch_X.size(0)

                val_logits_list.append(logits)
                val_targets_list.append(batch_y)

        val_loss /= len(val_loader.dataset)
        writer.add_scalar("Loss/Valid", val_loss, epoch)

        val_logits_all = torch.cat(val_logits_list, dim=0)
        val_targets_all = torch.cat(val_targets_list, dim=0)
        val_metrics = calculate_metrics(val_logits_all, val_targets_all, output_type, metrics)

        writer.add_scalar("Balanced_Accuracy/Valid", val_metrics["balanced_accuracy"], epoch)
        writer.add_scalar("Cohen_Kappa/Valid", val_metrics["cohen_kappa"], epoch)
        writer.add_scalar("AUROC/Valid", val_metrics["roc_auc"], epoch)
        if "pr_auc" in val_metrics:
            writer.add_scalar("PR_AUC/Valid", val_metrics["pr_auc"], epoch)

        current_val_balanced_acc = val_metrics["balanced_accuracy"]
        if current_val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = current_val_balanced_acc
            early_stopping_counter = 0

            checkpoint_dict = {
                "epoch": epoch + 1,
                "model_state_dict": probe.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            ckpt_save_path = os.path.join(log_dir, "best_model.ckpt")
            torch.save(checkpoint_dict, ckpt_save_path)
            print(f"--> Epoch {epoch+1}: 최고 Balanced Accuracy 갱신! ({best_val_balanced_acc:.4f}) -> 체크포인트 저장")
        else:
            early_stopping_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | "
                f"Valid B_Acc: {val_metrics['balanced_accuracy']:.4f} | "
                f"Valid Kappa: {val_metrics['cohen_kappa']:.4f} | "
                f"Valid AUROC: {val_metrics['roc_auc']:.4f}")

        if early_stopping_counter >= patience:
            print(f"\n[Early Stopping Triggered] {patience} 에폭 동안 성능 개선이 없어 학습을 조기 종료합니다. (현재 Epoch: {epoch+1})")
            break

    final_checkpoint_dict = {
        "epoch": epochs,
        "model_state_dict": probe.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    final_ckpt_path = os.path.join(log_dir, "final_model.ckpt")
    torch.save(final_checkpoint_dict, final_ckpt_path)
    print(f"===> [Layer {layer_num}] 학습 완료! 최종 모델 저장 완료 -> {final_ckpt_path}")

    print(f"===> [Layer {layer_num}] 최적의 가중치 복원 후 Test 세트 평가 시작...")
    best_ckpt = torch.load(os.path.join(log_dir, "best_model.ckpt"), map_location=probe_device)
    probe.load_state_dict(best_ckpt["model_state_dict"])
    probe.eval()

    test_loss = 0.0
    test_logits_list = []
    test_targets_list = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(probe_device), batch_y.to(probe_device)
            logits = probe(batch_X)
            loss = compute_loss(logits, batch_y, output_type)
            test_loss += loss.item() * batch_X.size(0)

            test_logits_list.append(logits)
            test_targets_list.append(batch_y)

    test_loss /= len(test_loader.dataset)
    test_logits_all = torch.cat(test_logits_list, dim=0)
    test_targets_all = torch.cat(test_targets_list, dim=0)
    test_metrics = calculate_metrics(test_logits_all, test_targets_all, output_type, metrics)

    print(f"=====> [Layer {layer_num}] Test 결과 | Test Loss: {test_loss:.4f} | "
        f"Test B_Acc: {test_metrics['balanced_accuracy']:.4f} | "
        f"Test Kappa: {test_metrics['cohen_kappa']:.4f} | "
        f"Test AUROC: {test_metrics['roc_auc']:.4f}")

    hparams_dict = {
        "layer_num": layer_num,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    metrics_dict = {f"hparam/test_{k}": v for k, v in test_metrics.items()}
    metrics_dict["hparam/test_loss"] = test_loss
    writer.add_hparams(
        hparam_dict=hparams_dict,
        metric_dict=metrics_dict,
        run_name="."
    )

    writer.close()

    del probe, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return test_metrics


def existing_layer_paths(split_temp_dir):
    """
    이미 temp_dir/{split}/layer_{n}.pt 가 전부 존재하면 그 경로들을 반환하고,
    하나라도 없으면 None을 반환한다. (중단된 실행을 재개할 때 이미 끝난 split의
    feature 추출을 다시 하지 않기 위함)
    """
    layer_paths = {}
    for layer_num in TARGET_LAYERS:
        layer_path = os.path.join(split_temp_dir, f"layer_{layer_num}.pt")
        if not os.path.exists(layer_path):
            return None
        layer_paths[layer_num] = layer_path
    return layer_paths


def get_split_layer_paths(split_name, dataset, model, temp_dir, device):
    """
    temp_dir/{split}/layer_{n}.pt 가 이미 모두 있으면(예: resume_merge.py로 미리
    병합해둔 경우) 추출을 건너뛰고 그 경로를 그대로 쓰고, 없으면 처음부터 추출한다.
    """
    split_temp_dir = os.path.join(temp_dir, split_name)
    cached = existing_layer_paths(split_temp_dir)
    if cached is not None:
        print(f"[{split_name}] layer_*.pt가 이미 존재합니다. feature 추출을 건너뜁니다.")
        return cached
    return extract_and_merge_split(split_name, dataset, model, temp_dir, device)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==========================================
# 3. Main
# ==========================================

if __name__ == "__main__":
    seed_torch(7)

    opts, ds_init = get_args()
    train_dataset, test_dataset, val_dataset = get_dataset(opts)

    print("-" * 100)
    print("*" * 100)
    print(f"NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL = {NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL}")

    model = get_models(opts)
    model = load_model_checkpoint(model, opts)

    print(f"DEVICE = {device}")
    model.to(device)
    model.eval()

    os.makedirs(TEMP_ROOT, exist_ok=True)

    # --- 1) 모든 split에 대해 layer-wise feature 추출 (temp 폴더에만 저장) ---
    # 이미 temp_dir/{split}/layer_*.pt 가 있으면(resume_merge.py로 미리 병합해둔 경우
    # 등) 해당 split의 추출은 건너뛴다.
    train_layer_paths = get_split_layer_paths("train", train_dataset, model, TEMP_ROOT, device)
    valid_layer_paths = get_split_layer_paths("valid", val_dataset, model, TEMP_ROOT, device)
    test_layer_paths = get_split_layer_paths("test", test_dataset, model, TEMP_ROOT, device)

    # feature 추출이 끝났으므로 모델을 GPU에서 내려 probe 학습에 메모리를 확보
    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2) layer별로 하나씩 linear probe 수행 ---
    os.makedirs(LOG_ROOT, exist_ok=True)
    summary = {}

    for layer_num in TARGET_LAYERS:
        test_metrics = run_layer_experiment(
            layer_num=layer_num,
            train_path=train_layer_paths[layer_num],
            valid_path=valid_layer_paths[layer_num],
            test_path=test_layer_paths[layer_num],
            log_root=LOG_ROOT,
        )
        summary[f"layer_{layer_num}"] = test_metrics

    with open(os.path.join(LOG_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[레이어별 Test 결과 요약]")
    for layer_name, m in summary.items():
        print(f"{layer_name}: {m}")

    # --- 3) 실험이 모두 끝났으므로 temp 폴더(추출된 feature) 삭제 ---
    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT)
        print(f"Removed temp feature directory: {TEMP_ROOT}")

    print(f"\nTUAB {NOTRAIN_LINEAR_OR_FINETUNE_OR_PEARL_LORA_OR_DORA_VERA_OR_DYNAMICPEARL} LAYER-WISE FEATURE EXTRACTION + LINEAR PROBE END")
    print("터미널에 'tensorboard --logdir=<log_root>'를 입력해 결과를 확인하세요.")
