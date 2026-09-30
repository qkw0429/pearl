"""
TUAB layer-wise class separation R^2 (streaming, no feature caching).

Metric from "Why Do Better Loss Functions Lead to Less Transferable Features?"
(Kornblith et al., NeurIPS 2021), Eq. (11):

    R^2 = 1 - d_within_bar / d_total_bar

    d_within_bar = sum_k sum_m sum_n (1 - sim(x_{k,m}, x_{k,n})) / (K * N_k^2)
    d_total_bar  = sum_j sum_k sum_m sum_n (1 - sim(x_{j,m}, x_{k,n})) / (K^2 * N_j * N_k)

where sim(.,.) is cosine similarity, K is the number of classes, and N_k is the
number of samples in class k. Self-pairs (m == n) are included.

As in layer_wise_class_separation_BCIC2A.py, this is computed exactly (not
approximately) via the identity sum_m sum_n sim(x_m, x_n) = ||sum_m x_m_hat||^2,
which only requires accumulating, per class, the running sum of L2-normalized
feature vectors — never the O(N^2) pairwise similarity matrix, and never the
full dataset's features held in memory or written to disk:

    mu_k = (1/N_k) * sum_{m in class k} x_m_hat
    d_within_bar = 1 - (1/K) * sum_k ||mu_k||^2
    d_total_bar  = 1 - || (1/K) * sum_k mu_k ||^2

Unlike layer_wise_feature_distance_TUAB.py, this metric is intrinsic to a
single method's own features (it doesn't compare against a frozen reference),
so only one model needs to be resident on GPU at a time, and one dataset pass
per (split, method) covers all 8 layers.
"""

import os
import gc
import random

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from einops import rearrange
from torch.utils.data import DataLoader

from Modules.models.EEGPT_mcae_finetune_change_for_linear_wise_invest import (
    EEGPTClassifier,
    EEGPTClassifier_rep_conv,
    EEGPTClassifier_rep_MULTI_SCALE_DYNAMIC_CONV,
)
from peft import LoraConfig, get_peft_model, VeraConfig

import utils
from feature_extraction_and_linear_probe_TUAB import (
    get_args,
    TUABLoader,
    prepare_TUAB_dataset,
)


# ==========================================
# 0. 설정값
# ==========================================
TUAB_ROOT = "/workspace/EEGPT/dataset/TUAB/edf/processed"

SPLITS = ["train", "valid", "test"]

METHODS = {
    "frozen": "NOTRAIN",
    "linear_probe": "LINEAR",
    "finetune": "FINETUNE",
    "LORA": "LORA",
    "VERA": "VERA",
    "dynamic_pearl": "DYNAMICPEARL",
}

TARGET_LAYERS = list(range(1, 9))
EMBED_NUM = 4  # channel 축 뒤쪽 summary token 개수

SLICING_MODES = ["raw", "sliced"]
FEATURE_MODES = ["flatten", "pooling"]

BATCH_SIZE = 500

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

OUTPUT_DIR = "/workspace/EEGPT/dataset/TUAB/layer_wise"
DATE_TAG = "260930"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_csv_path(slicing_mode, feature_mode):
    return f"{OUTPUT_DIR}/{DATE_TAG}_layer_wise_class_separation_{slicing_mode}_{feature_mode}.csv"


# ==========================================
# 1. 모델 빌드 (layer_wise_feature_distance_TUAB.py와 동일)
# ==========================================
def apply_lora_to_eeg_encoder(model, use_dora, lora_alpha=8, lora_dropout=0.05):
    target_regex = r".*blocks.*(?:qkv)"
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=lora_alpha,
        target_modules=target_regex,
        lora_dropout=lora_dropout,
        bias="none",
        use_dora=use_dora,
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


def apply_vera_to_eeg_encoder(model, vera_dropout=0.0):
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


def build_model(method, args):
    use_channels_names = [
        'FP1', 'FPZ', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'O2']

    if method == 'FINETUNE':
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF',
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    else:
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T7-REF', 'EEG T8-REF', 'EEG P7-REF', 'EEG P8-REF', 'EEG TP7-REF', 'EEG TP8-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG FT7-REF', 'EEG FT8-REF']

    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

    if method == 'PEARL':
        model = EEGPTClassifier_rep_conv(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(ch_names), 2368],
            use_channels_names=ch_names,
            use_chan_conv=False,
            use_mean_pooling=args.use_mean_pooling,
            kernel_size=KERNEL, conv_num_heads=HEADS, conv_weight_softmax=CONV_SOFTMAX, conv_bias=CONV_BIAS, conv_dropout=CONV_DROP, stride=STRIDE,
        )
    elif method == 'DYNAMICPEARL':
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
    elif method == 'FINETUNE':
        model = EEGPTClassifier(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(use_channels_names), 2000],
            use_channels_names=use_channels_names,
            use_chan_conv=True,
            use_mean_pooling=args.use_mean_pooling)
    else:
        model = EEGPTClassifier(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(ch_names), 2000],
            use_channels_names=ch_names,
            use_chan_conv=False,
            use_mean_pooling=args.use_mean_pooling)

    return model


def load_model_checkpoint(model, opts, method):
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
        model.load_state_dict(checkpoint['model'], strict=False)
    elif method == 'DYNAMICPEARL':
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260812_2_SEED_0_PEARL_MULTI_SCALE_DYNAMIC_CONV_KERNEL_5,7,9,11_REDUCTION_2/checkpoint-best_48_epoch.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif method == 'LORA':
        model = apply_lora_to_eeg_encoder(model, use_dora=False)
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260809_4_SEED_0_LORA/checkpoint-best_13_epoch.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif method == 'DORA':
        model = apply_lora_to_eeg_encoder(model, use_dora=True)
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260809_5_SEED_0_DORA_only_qkv_RANK_8_NO_CHAN_CONV/checkpoint-best_5_epoch.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif method == 'VERA':
        model = apply_vera_to_eeg_encoder(model)
        checkpoint = torch.load('/workspace/EEGPT/log/TUAB_CHECKPOINT/EEGPT/260809_6_SEED_0_VERA_ONLY_QKV_RANK_256_NO_CHAN_CONV/checkpoint-best_47_epoch.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model


def build_and_load_model(method, args):
    model = build_model(method, args)
    model = load_model_checkpoint(model, args, method)
    model.to(DEVICE)
    model.eval()
    return model


# ==========================================
# 2. feature 변환 (layer_wise_feature_distance_TUAB.py와 동일한 정의)
# ==========================================
def apply_slicing(x, embed_num):
    # x: [B, T, C, D] -> [B, T, embed_num, D] (channel 축(dim=2)의 뒤쪽 embed_num개=summary token만 사용)
    return x[:, :, -embed_num:, :]


def to_flatten(x):
    return x.flatten(start_dim=1)


def to_pooling(x):
    return x.mean(dim=(1, 2))


def build_feature_vector(x, slicing_mode, feature_mode, embed_num):
    if slicing_mode == "sliced":
        x = apply_slicing(x, embed_num)

    if feature_mode == "flatten":
        return to_flatten(x)
    elif feature_mode == "pooling":
        return to_pooling(x)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")


# ==========================================
# 3. Class separation R^2 스트리밍 누적 (Kornblith et al., 2021, Eq. 11)
# ==========================================
class ClassSeparationAccumulator:
    """
    (layer, slicing_mode, feature_mode) 조합 하나에 대한 클래스별 누적 통계.
    sum_m sum_n sim(x_m, x_n) = || sum_m x_m_hat ||^2 라는 내적의 분배법칙을 이용해,
    클래스별 정규화 벡터의 누적합만으로 R^2를 정확히(근사 아님) 계산한다.
    """

    def __init__(self):
        self.class_sum = {}    # class_label -> running sum of x_hat (float64)
        self.class_count = {}  # class_label -> N_k

    def update(self, x_vec, y):
        x_hat = F.normalize(x_vec.double(), dim=1)
        y_cpu = y.detach().cpu()

        for c in torch.unique(y_cpu).tolist():
            mask = (y_cpu == c)
            idx = mask.nonzero(as_tuple=True)[0].to(x_hat.device)
            s = x_hat.index_select(0, idx).sum(dim=0)
            n = idx.numel()

            if c not in self.class_sum:
                self.class_sum[c] = s
                self.class_count[c] = n
            else:
                self.class_sum[c] += s
                self.class_count[c] += n

    def finalize(self):
        classes = sorted(self.class_sum.keys())
        if len(classes) < 2:
            return float("nan")

        mu_list = [self.class_sum[c] / self.class_count[c] for c in classes]
        mu_stack = torch.stack(mu_list, dim=0)  # [K, D']
        K = mu_stack.shape[0]

        d_within = 1.0 - (mu_stack.pow(2).sum(dim=1)).mean().item()

        mu_bar = mu_stack.mean(dim=0)
        d_total = 1.0 - mu_bar.pow(2).sum().item()

        if d_total == 0.0:
            return float("nan")

        return 1.0 - d_within / d_total


def run_split(split_name, dataset, model, method_name, results):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    accumulators = {
        (layer, slicing_mode, feature_mode): ClassSeparationAccumulator()
        for layer in TARGET_LAYERS
        for slicing_mode in SLICING_MODES
        for feature_mode in FEATURE_MODES
    }

    with torch.no_grad():
        for step, data in enumerate(loader):
            inputs, labels, subject_ids = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            B = inputs.shape[0]

            features = model(inputs)

            for layer_num in TARGET_LAYERS:
                layer_idx = layer_num - 1

                feat = rearrange(features[layer_idx], '(B N) dim1 dim2 -> B N dim1 dim2', B=B)

                for slicing_mode in SLICING_MODES:
                    for feature_mode in FEATURE_MODES:
                        x_vec = build_feature_vector(feat, slicing_mode, feature_mode, EMBED_NUM)
                        accumulators[(layer_num, slicing_mode, feature_mode)].update(x_vec, labels)

            if step % 5 == 0:
                torch.cuda.empty_cache()

    for (layer_num, slicing_mode, feature_mode), acc in accumulators.items():
        r2 = acc.finalize()
        print(
            f"[Layer {layer_num}][{split_name}][{method_name}][{slicing_mode}/{feature_mode}] R^2={r2:.4f}"
        )

        results.append({
            "layer": layer_num,
            "split": split_name,
            "method": method_name,
            "slicing_mode": slicing_mode,
            "feature_mode": feature_mode,
            "class_separation_r2": r2,
        })


def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==========================================
# 4. 메인
# ==========================================
def main():
    seed_torch(7)

    opts, _ = get_args()
    opts.nb_classes = 1

    train_dataset, test_dataset, val_dataset = prepare_TUAB_dataset(TUAB_ROOT)
    datasets = {"train": train_dataset, "valid": val_dataset, "test": test_dataset}

    results = []

    for method_name, method_token in METHODS.items():
        print(f"\n=== Method: {method_name} ({method_token}) ===")
        model = build_and_load_model(method_token, opts)

        for split_name in SPLITS:
            run_split(split_name, datasets[split_name], model, method_name, results)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    group_cols = ["layer", "method", "slicing_mode", "feature_mode"]
    metric_cols = ["class_separation_r2"]
    avg_df = df.groupby(group_cols)[metric_cols].mean().reset_index()
    avg_df["split"] = "average"

    final_df = pd.concat([df, avg_df], ignore_index=True)
    final_df = final_df[["layer", "split", "method", "slicing_mode", "feature_mode"] + metric_cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for slicing_mode in SLICING_MODES:
        for feature_mode in FEATURE_MODES:
            combo_df = final_df[
                (final_df["slicing_mode"] == slicing_mode)
                & (final_df["feature_mode"] == feature_mode)
            ]
            combo_csv = output_csv_path(slicing_mode, feature_mode)
            combo_df.to_csv(combo_csv, index=False)
            print(f"[완료] 결과 저장: {combo_csv}")


if __name__ == "__main__":
    main()
