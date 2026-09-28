"""
TUEV layer-wise encoder feature cosine distance (streaming, no feature caching).

Same approach as layer_wise_feature_distance_TUAB.py: TUEV is also too large to
cache every method's raw feature to disk before comparing. This script runs the
frozen encoder and one comparison-method encoder on the same batch
simultaneously and updates running accumulators per
(layer, slicing_mode, feature_mode), so all three distance metrics come out
mathematically identical to computing them over the full dataset at once,
without ever holding the whole dataset's features in memory or on disk:

  - same_index_pairwise_mean: accumulate sum((1 - cos_sim)) and a sample count.
  - full_pairwise_mean: uses sum_i sum_j (a_i_hat . b_j_hat)
        = (sum_i a_i_hat) . (sum_j b_j_hat)
    so only running sums of L2-normalized vectors are needed, never the full
    N x M pairwise matrix.
  - mean_vector_distance: accumulate the raw (non-normalized) vector sums,
    divide by count and take one cosine distance at the very end.

One pass over a split's dataloader (per method) is enough for all 8 layers,
since the model returns every layer's feature in a single forward call.
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

from Modules.models.EEGPT_mcae_finetune_change_tuev_for_linear_wise_invest import (
    EEGPTClassifier,
    EEGPTClassifier_rep_conv,
    EEGPTClassifier_rep_MULTI_SCALE_DYNAMIC_CONV,
)
from peft import LoraConfig, get_peft_model, VeraConfig

import utils
from feature_extraction_and_linear_probe_TUEV import (
    get_args,
    TUEVLoader,
    prepare_TUEV_dataset,
)


# ==========================================
# 0. 설정값
# ==========================================
TUEV_ROOT = "/workspace/EEGPT/dataset/TUEV/processed"

SPLITS = ["train", "valid", "test"]

REFERENCE_METHOD = "frozen"

# method 이름 -> get_models/load_model_checkpoint에서 쓰는 내부 토큰
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

BATCH_SIZE = 600

KERNEL = 100
STRIDE = 100
HEADS = 1
CONV_SOFTMAX = False
CONV_BIAS = True
CONV_DROP = 0.5
KERNEL_SIZES = (31, 41, 51, 61, 71, 81, 91, 101)
REDUCTION = 2

LORA_RANK = 8
VERA_RANK = 256

OUTPUT_DIR = "/workspace/EEGPT/dataset/TUEV/layer_wise"
DATE_TAG = "260928"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_csv_path(slicing_mode, feature_mode):
    return f"{OUTPUT_DIR}/{DATE_TAG}_layer_wise_feature_distance_{slicing_mode}_{feature_mode}.csv"


# ==========================================
# 1. 모델 빌드 (feature_extraction_and_linear_probe_TUEV.py의 get_models / load_model_checkpoint를
#    method 인자를 받는 형태로 재구성)
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
            img_size=[len(ch_names), 1000],
            use_channels_names=ch_names,
            use_chan_conv=False,
            use_mean_pooling=args.use_mean_pooling,
            kernel_size=KERNEL, conv_num_heads=HEADS, conv_weight_softmax=CONV_SOFTMAX, conv_bias=CONV_BIAS, conv_dropout=CONV_DROP, stride=STRIDE,
        )
    elif method == 'DYNAMICPEARL':
        model = EEGPTClassifier_rep_MULTI_SCALE_DYNAMIC_CONV(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(ch_names), 1000],
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
            img_size=[len(use_channels_names), 1000],
            use_channels_names=use_channels_names,
            use_chan_conv=True,
            use_mean_pooling=args.use_mean_pooling)
    else:
        model = EEGPTClassifier(
            num_classes=args.nb_classes,
            in_channels=len(ch_names),
            img_size=[len(ch_names), 1000],
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
        checkpoint = torch.load('/workspace/EEGPT/log/TUEV_CHECKPOINT/EEGPT/260813_1_SEED_0_LP_NO_CHAN_CONV/checkpoint-best_23_epoch.pth', map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model']
        utils.load_state_dict(model, checkpoint_model, prefix=opts.model_prefix)
    elif method == 'FINETUNE':
        checkpoint = torch.load('/workspace/EEGPT/log/TUEV_CHECKPOINT/EEGPT/260813_3_SEED_0_FT/checkpoint-best_15_epoch.pth', map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model']
        utils.load_state_dict(model, checkpoint_model, prefix=opts.model_prefix)
    elif method == 'PEARL':
        checkpoint = torch.load('/workspace/EEGPT/log/TUEV_CHECKPOINT/EEGPT/260813_2_SEED_0_PEARL/checkpoint-best_24_epoch.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif method == 'DYNAMICPEARL':
        checkpoint = torch.load('/workspace/EEGPT/log/TUEV_CHECKPOINT/EEGPT/260818_1_SEED_0_PEARL_MULTI_SCALE_DYNAMIC_CONV_KERNEL_31,41,51,61,71,81,91,101_REDUCTION_2/checkpoint-best_16_epoch.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif method == 'LORA':
        model = apply_lora_to_eeg_encoder(model, use_dora=False)
        checkpoint = torch.load('/workspace/EEGPT/log/TUEV_CHECKPOINT/EEGPT/260814_1_SEED_0_LORA_ONLY_QKV_RANK_8_NO_CHAN_CONV/checkpoint-best_16_epoch.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif method == 'DORA':
        model = apply_lora_to_eeg_encoder(model, use_dora=True)
        checkpoint = torch.load('/workspace/EEGPT/log/TUEV_CHECKPOINT/EEGPT/260814_2_SEED_0_DORA_ONLY_QKV_RANK_8_NO_CHAN_CONV/checkpoint-best_14_epoch.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif method == 'VERA':
        model = apply_vera_to_eeg_encoder(model)
        checkpoint = torch.load('/workspace/EEGPT/log/TUEV_CHECKPOINT/EEGPT/260814_3_SEED_0_VERA_ONLY_QKV_RANK_256_NO_CHAN_CONV/checkpoint-best_18_epoch.pth', map_location='cpu', weights_only=False)
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
# 2. feature 변환 (layer_wise_feature_distance_BCIC2A.py / TUAB.py와 동일한 정의)
# ==========================================
def apply_slicing(x, embed_num):
    # x: [B, T, C, D] -> [B, T, embed_num, D] (channel 축(dim=2)의 뒤쪽 embed_num개=summary token만 사용)
    return x[:, :, -embed_num:, :]


def crop_to_last(x, target_time_len):
    # x: [B, T, C, D] -> time 축(dim=1)에서 뒤쪽 target_time_len개만 사용
    return x[:, -target_time_len:, :, :]


def match_time_length(r, o):
    r_time_len = r.shape[1]
    o_time_len = o.shape[1]
    min_time_len = min(r_time_len, o_time_len)

    if r_time_len != min_time_len:
        r = crop_to_last(r, min_time_len)
    if o_time_len != min_time_len:
        o = crop_to_last(o, min_time_len)

    return r, o


def to_flatten(x):
    return x.flatten(start_dim=1)


def to_pooling(x):
    return x.mean(dim=(1, 2))


def build_feature_pair(ref_x, other_x, slicing_mode, feature_mode, embed_num):
    r, o = ref_x, other_x

    if slicing_mode == "sliced":
        r = apply_slicing(r, embed_num)
        o = apply_slicing(o, embed_num)

    if feature_mode == "flatten":
        r, o = match_time_length(r, o)
        if r.shape[2] != o.shape[2]:
            return None, None
        r_vec = to_flatten(r)
        o_vec = to_flatten(o)
    elif feature_mode == "pooling":
        r_vec = to_pooling(r)
        o_vec = to_pooling(o)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    return r_vec, o_vec


# ==========================================
# 3. 스트리밍 누적 통계
# ==========================================
class RunningAccumulator:
    """
    (layer, slicing_mode, feature_mode) 조합 하나에 대한 누적 통계.
    - full_pairwise_mean은 sum_i sum_j (a_i_hat . b_j_hat) = (sum_i a_i_hat) . (sum_j b_j_hat)
      라는 내적의 분배법칙을 이용해, N x M pairwise 행렬을 절대 만들지 않고
      정규화된 벡터의 누적합만으로 정확히(근사 아님) 계산한다.
    """

    def __init__(self):
        self.count = 0
        self.same_idx_sum = 0.0
        self.sum_r_hat = None
        self.sum_o_hat = None
        self.sum_r_raw = None
        self.sum_o_raw = None
        self.skipped = False

    def update(self, r_vec, o_vec):
        if r_vec is None:
            self.skipped = True
            return

        r_vec = r_vec.double()
        o_vec = o_vec.double()

        r_hat = F.normalize(r_vec, dim=1)
        o_hat = F.normalize(o_vec, dim=1)

        batch_same_idx = (1.0 - (r_hat * o_hat).sum(dim=1)).sum().item()
        self.same_idx_sum += batch_same_idx
        self.count += r_vec.shape[0]

        r_hat_sum = r_hat.sum(dim=0)
        o_hat_sum = o_hat.sum(dim=0)
        r_raw_sum = r_vec.sum(dim=0)
        o_raw_sum = o_vec.sum(dim=0)

        if self.sum_r_hat is None:
            self.sum_r_hat = r_hat_sum
            self.sum_o_hat = o_hat_sum
            self.sum_r_raw = r_raw_sum
            self.sum_o_raw = o_raw_sum
        else:
            self.sum_r_hat += r_hat_sum
            self.sum_o_hat += o_hat_sum
            self.sum_r_raw += r_raw_sum
            self.sum_o_raw += o_raw_sum

    def finalize(self):
        if self.skipped or self.count == 0:
            return {
                "same_index_pairwise_mean": float("nan"),
                "full_pairwise_mean": float("nan"),
                "mean_vector_distance": float("nan"),
            }

        same_idx_mean = self.same_idx_sum / self.count

        full_pairwise_mean = (self.sum_r_hat @ self.sum_o_hat).item() / (self.count * self.count)

        mean_r = self.sum_r_raw / self.count
        mean_o = self.sum_o_raw / self.count
        mean_r_hat = F.normalize(mean_r.unsqueeze(0), dim=1)
        mean_o_hat = F.normalize(mean_o.unsqueeze(0), dim=1)
        mean_vector_distance = (1.0 - (mean_r_hat * mean_o_hat).sum(dim=1)).item()

        return {
            "same_index_pairwise_mean": same_idx_mean,
            "full_pairwise_mean": full_pairwise_mean,
            "mean_vector_distance": mean_vector_distance,
        }


def run_split(split_name, dataset, ref_model, other_model, method_name, results):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    accumulators = {
        (layer, slicing_mode, feature_mode): RunningAccumulator()
        for layer in TARGET_LAYERS
        for slicing_mode in SLICING_MODES
        for feature_mode in FEATURE_MODES
    }

    with torch.no_grad():
        for step, data in enumerate(loader):
            inputs, labels, subject_ids = data
            inputs = inputs.to(DEVICE)
            B = inputs.shape[0]

            with torch.amp.autocast('cuda'):
                ref_features = ref_model(inputs)
                other_features = other_model(inputs)

            for layer_num in TARGET_LAYERS:
                layer_idx = layer_num - 1

                ref_feat = rearrange(ref_features[layer_idx], '(B N) dim1 dim2 -> B N dim1 dim2', B=B).float()
                other_feat = rearrange(other_features[layer_idx], '(B N) dim1 dim2 -> B N dim1 dim2', B=B).float()

                if ref_feat.shape[0] != other_feat.shape[0]:
                    print(f"[경고] Layer {layer_num} {split_name} {method_name}: 배치 sample 수 불일치. 스킵합니다.")
                    continue

                for slicing_mode in SLICING_MODES:
                    for feature_mode in FEATURE_MODES:
                        r_vec, o_vec = build_feature_pair(
                            ref_feat, other_feat, slicing_mode, feature_mode, EMBED_NUM
                        )
                        accumulators[(layer_num, slicing_mode, feature_mode)].update(r_vec, o_vec)

            if step % 5 == 0:
                torch.cuda.empty_cache()

    for (layer_num, slicing_mode, feature_mode), acc in accumulators.items():
        metrics = acc.finalize()
        if acc.skipped:
            print(
                f"[스킵] Layer {layer_num} {split_name} {method_name}[{slicing_mode}/{feature_mode}]: "
                f"channel 개수가 달라 flatten 비교 불가"
            )
        else:
            print(
                f"[Layer {layer_num}][{split_name}][{method_name}][{slicing_mode}/{feature_mode}] "
                f"same_idx={metrics['same_index_pairwise_mean']:.4f} "
                f"full_pairwise={metrics['full_pairwise_mean']:.4f} "
                f"mean_vec={metrics['mean_vector_distance']:.4f}"
            )

        results.append({
            "layer": layer_num,
            "split": split_name,
            "method": method_name,
            "slicing_mode": slicing_mode,
            "feature_mode": feature_mode,
            **metrics,
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
    opts.nb_classes = 6

    train_dataset, test_dataset, val_dataset = prepare_TUEV_dataset(TUEV_ROOT)
    datasets = {"train": train_dataset, "valid": val_dataset, "test": test_dataset}

    results = []

    print(f"Loading reference model ({METHODS[REFERENCE_METHOD]})...")
    ref_model = build_and_load_model(METHODS[REFERENCE_METHOD], opts)

    for method_name, method_token in METHODS.items():
        if method_name == REFERENCE_METHOD:
            continue

        print(f"\n=== Method: {method_name} ({method_token}) ===")
        other_model = build_and_load_model(method_token, opts)

        for split_name in SPLITS:
            run_split(split_name, datasets[split_name], ref_model, other_model, method_name, results)

        del other_model
        gc.collect()
        torch.cuda.empty_cache()

    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    group_cols = ["layer", "method", "slicing_mode", "feature_mode"]
    metric_cols = ["same_index_pairwise_mean", "full_pairwise_mean", "mean_vector_distance"]
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
