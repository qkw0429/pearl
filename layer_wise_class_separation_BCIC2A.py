"""
BCIC2A layer-wise class separation R^2.

Metric from "Why Do Better Loss Functions Lead to Less Transferable Features?"
(Kornblith et al., NeurIPS 2021), Eq. (11):

    R^2 = 1 - d_within_bar / d_total_bar

    d_within_bar = sum_k sum_m sum_n (1 - sim(x_{k,m}, x_{k,n})) / (K * N_k^2)
    d_total_bar  = sum_j sum_k sum_m sum_n (1 - sim(x_{j,m}, x_{k,n})) / (K^2 * N_j * N_k)

where sim(.,.) is cosine similarity, K is the number of classes, and N_k is the
number of samples in class k. Self-pairs (m == n) are included, not excluded.

Rather than materializing the O(N^2) pairwise similarity matrices, this is
computed exactly (not approximately) via the identity
sum_m sum_n sim(x_m, x_n) = || sum_m x_m_hat ||^2 (bilinearity of the dot
product), reducing the cost to O(N*D + K*D):

    mu_k = (1/N_k) * sum_{m in class k} x_m_hat      (x_hat = L2-normalized x)
    d_within_bar = 1 - (1/K) * sum_k ||mu_k||^2
    d_total_bar  = 1 - || (1/K) * sum_k mu_k ||^2
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd


# ==========================================
# 0. 설정값
# ==========================================
DATASET_PATH = "/workspace/EEGPT/dataset/BCIC-2A/layer_wise"
TRAIN_FILENAME = "train_features_labels_subjectids.pt"
VALID_FILENAME = "valid_features_labels_subjectids.pt"
TEST_FILENAME = "test_features_labels_subjectids.pt"

SUB_TAG = "SUB1_Model"

SPLITS = {
    "train": ("train", TRAIN_FILENAME),
    "valid": ("valid", VALID_FILENAME),
    "test": ("only_test", TEST_FILENAME),
}

METHODS = {
    "frozen": "NOTRAIN",
    "linear_probe": "linear_probe",
    "finetune": "finetune",
    "LORA": "LORA",
    "VERA": "VERA",
    "dynamic_pearl": "dynamic_pearl",
}

LAYERS = range(1, 9)
EMBED_NUM = 4  # sliced feature에서 사용할, channel 축 뒤쪽 summary token 개수

SLICING_MODES = ["raw", "sliced"]
FEATURE_MODES = ["flatten", "pooling"]

OUTPUT_DIR = DATASET_PATH
DATE_TAG = "260930"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_csv_path(slicing_mode, feature_mode):
    return f"{OUTPUT_DIR}/{DATE_TAG}_layer_wise_class_separation_{slicing_mode}_{feature_mode}.csv"


# ==========================================
# 1. 데이터 로드
# ==========================================
def load_raw_feature_and_label(dataset_path, split_dir, model_folder, layer_num, filename):
    """
    저장된 레이어별 feature(.pt)를 원본 shape 그대로 불러옵니다.
    반환: (x, y)
    x shape: [N, time_patch, channel(=실채널 수 + embed_num summary token), embed_dim]
    y shape: [N]
    """
    layer_dir = f"{dataset_path}/{split_dir}/{model_folder}_model/{SUB_TAG}/layer_{layer_num}"
    file_path = os.path.join(layer_dir, filename)

    if not os.path.exists(file_path):
        print(f"[경고] {file_path} 경로에 파일이 존재하지 않습니다. 스킵합니다.")
        return None, None

    data = torch.load(file_path)
    x = data["x"].float()
    y = data["y"]
    return x, y


# ==========================================
# 2. feature 변환 (layer_wise_feature_distance_BCIC2A.py와 동일한 정의)
# ==========================================
def apply_slicing(x, embed_num):
    # x: [N, T, C, D] -> [N, T, embed_num, D]
    # channel 축(dim=2, 크기 = 실채널 수 + embed_num)에서 뒤쪽 embed_num개(summary token)만 사용
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
# 3. Class separation R^2 (Kornblith et al., 2021, Eq. 11)
# ==========================================
def class_separation_r2(x_vec, y):
    """
    x_vec: [N, D'] feature 벡터
    y: [N] 클래스 레이블 (0, 1, ..., K-1)
    """
    x_hat = F.normalize(x_vec, dim=1)
    classes = torch.unique(y)
    K = classes.numel()

    mu_list = []
    for c in classes:
        mask = (y == c)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        mu_k = x_hat[mask].sum(dim=0) / n_k  # (1/N_k) * sum_{m in k} x_m_hat
        mu_list.append(mu_k)

    mu_stack = torch.stack(mu_list, dim=0)  # [K, D']

    d_within = 1.0 - (mu_stack.pow(2).sum(dim=1)).mean().item()

    mu_bar = mu_stack.mean(dim=0)  # (1/K) * sum_k mu_k
    d_total = 1.0 - mu_bar.pow(2).sum().item()

    if d_total == 0.0:
        return float("nan")

    r2 = 1.0 - d_within / d_total
    return r2


# ==========================================
# 4. 메인 실험 루프
# ==========================================
def main():
    results = []

    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")

        for split_name, (split_dir, filename) in SPLITS.items():
            for method_name, method_folder in METHODS.items():
                x, y = load_raw_feature_and_label(DATASET_PATH, split_dir, method_folder, layer, filename)
                if x is None:
                    continue

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                for slicing_mode in SLICING_MODES:
                    for feature_mode in FEATURE_MODES:
                        x_vec = build_feature_vector(x, slicing_mode, feature_mode, EMBED_NUM)
                        r2 = class_separation_r2(x_vec, y)

                        results.append({
                            "layer": layer,
                            "split": split_name,
                            "method": method_name,
                            "slicing_mode": slicing_mode,
                            "feature_mode": feature_mode,
                            "class_separation_r2": r2,
                        })

                        print(
                            f"[Layer {layer}][{split_name}][{method_name}]"
                            f"[{slicing_mode}/{feature_mode}] R^2={r2:.4f}"
                        )

                del x, y

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
