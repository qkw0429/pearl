"""
BCIC2A layer-wise class-distance normalized variance (CDNV).

Metric from "On the Role of Neural Collapse in Transfer Learning"
(Galanti et al., ICLR 2022):

    V_f(Q_i, Q_j) = (Var_f(Q_i) + Var_f(Q_j)) / (2 * ||mu_f(Q_i) - mu_f(Q_j)||^2)

    mu_f(Q_c)  = E_{x~Q_c}[f(x)]
    Var_f(Q_c) = E_{x~Q_c}[||f(x) - mu_f(Q_c)||^2]

For finite sample sets, V_f(S_i, S_j) = V_f(U[S_i], U[S_j]) (uniform
distribution over each class's samples, so the variance divides by N_c, not
N_c - 1). The reported value is Avg_{i != j}[V_f(S_i, S_j)] over all pairs of
distinct classes. Lower CDNV means stronger class separation.
"""

import os
from itertools import combinations

import torch
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
DATE_TAG = "261002"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_csv_path(slicing_mode, feature_mode):
    return f"{OUTPUT_DIR}/{DATE_TAG}_layer_wise_cdnv_{slicing_mode}_{feature_mode}.csv"


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
# 2. feature 변환 (layer_wise_class_separation_BCIC2A.py와 동일한 정의)
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
# 3. CDNV (Galanti et al., 2022)
# ==========================================
def cdnv(x_vec, y):
    """
    x_vec: [N, D'] feature 벡터
    y: [N] 클래스 레이블
    반환: Avg_{i != j} V_f(S_i, S_j)
    """
    means = {}
    variances = {}
    for c in torch.unique(y).tolist():
        x_c = x_vec[y == c]
        mu_c = x_c.mean(dim=0)
        # uniform distribution over S_c이므로 N_c로 나눔 (N_c - 1 아님)
        var_c = (x_c - mu_c).pow(2).sum(dim=1).double().mean().item()
        means[c] = mu_c.double()
        variances[c] = var_c

    classes = sorted(means.keys())
    if len(classes) < 2:
        return float("nan")

    # V_f는 대칭이므로 순서쌍(i != j) 평균 = 비순서쌍(i < j) 평균
    pair_values = []
    for i, j in combinations(classes, 2):
        mean_dist_sq = (means[i] - means[j]).pow(2).sum().item()
        pair_values.append((variances[i] + variances[j]) / (2.0 * mean_dist_sq))

    return sum(pair_values) / len(pair_values)


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
                        value = cdnv(x_vec, y)

                        results.append({
                            "layer": layer,
                            "split": split_name,
                            "method": method_name,
                            "slicing_mode": slicing_mode,
                            "feature_mode": feature_mode,
                            "cdnv": value,
                        })

                        print(
                            f"[Layer {layer}][{split_name}][{method_name}]"
                            f"[{slicing_mode}/{feature_mode}] CDNV={value:.4f}"
                        )

                del x, y
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    group_cols = ["layer", "method", "slicing_mode", "feature_mode"]
    metric_cols = ["cdnv"]
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
