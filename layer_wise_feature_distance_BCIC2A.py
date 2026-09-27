import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gc
import torch
import torch.nn.functional as F
import pandas as pd


# ==========================================
# 0. 설정값
# ==========================================
DATASET_PATH = "/workspace/EEGPT/dataset/BCIC-2A/layer_wise"  # 사용 중인 데이터셋 폴더명 입력
TRAIN_FILENAME = "train_features_labels_subjectids.pt"
VALID_FILENAME = "valid_features_labels_subjectids.pt"
TEST_FILENAME = "test_features_labels_subjectids.pt"

SUB_TAG = "SUB1_Model"

# split 이름 -> (경로상의 폴더명, 파일명)
SPLITS = {
    "train": ("train", TRAIN_FILENAME),
    "valid": ("valid", VALID_FILENAME),
    "test": ("only_test", TEST_FILENAME),
}

# 거리 비교의 기준(reference)이 되는 순수 frozen feature
REFERENCE_METHOD = "frozen"

# method 이름 -> 데이터 경로상의 {model} 폴더명
METHODS = {
    "frozen": "NOTRAIN",
    "linear_probe": "linear_probe",
    "finetune": "finetune",
    "LORA": "LORA",
    "VERA": "VERA",
    "dynamic_pearl": "dynamic_pearl",
}

LAYERS = range(1, 9)
EMBED_NUM = 4  # sliced feature에서 사용할, 뒤쪽 time token 개수

# 원본 feature shape [N, ?, ?, D]에서 time token이 위치한 축 번호.
# BCIC2A는 (N, channel, time, D) 순서라 time이 뒤에서 두 번째 축(-2).
# 데이터셋마다 channel/time 축 순서가 다를 수 있으므로(예: PhysioP300은
# (N, time, channel, D) 순서), 그 데이터셋에 맞게 이 값만 바꿔주면 됨.
TIME_AXIS = -2

SLICING_MODES = ["raw", "sliced"]     # raw: 원본 feature, sliced: x[..., -EMBED_NUM:, :]
FEATURE_MODES = ["flatten", "pooling"]  # flatten: 그대로 펼침, pooling: channel/time 축 평균

OUTPUT_DIR = DATASET_PATH
DATE_TAG = "260926"


def output_csv_path(slicing_mode, feature_mode):
    # 예: 260926_layer_wise_feature_distance_raw_flatten.csv
    return f"{OUTPUT_DIR}/{DATE_TAG}_layer_wise_feature_distance_{slicing_mode}_{feature_mode}.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 데이터 로드
# ==========================================
def load_raw_feature(dataset_path, split_dir, model_folder, layer_num, filename):
    """
    저장된 레이어별 feature(.pt)를 원본 shape 그대로 불러옵니다.
    반환 shape: [N, channel_token, time_token, embed_dim]
    """
    layer_dir = f"{dataset_path}/{split_dir}/{model_folder}_model/{SUB_TAG}/layer_{layer_num}"
    file_path = os.path.join(layer_dir, filename)

    if not os.path.exists(file_path):
        print(f"[경고] {file_path} 경로에 파일이 존재하지 않습니다. 스킵합니다.")
        return None

    data = torch.load(file_path)
    x = data["x"].float()
    return x


# ==========================================
# 2. feature 변환 (slicing / flatten / pooling)
# ==========================================
def take_last_n_along_axis(x, n, axis):
    # x의 axis 축에서 뒤쪽 n개만 잘라서 반환 (축 위치에 상관없이 동작)
    axis = axis % x.dim()
    index = torch.arange(x.shape[axis] - n, x.shape[axis], device=x.device)
    return x.index_select(axis, index)


def apply_slicing(x, embed_num, time_axis=TIME_AXIS):
    # time_axis 축에서 뒤쪽 embed_num개 time token만 사용
    return take_last_n_along_axis(x, embed_num, time_axis)


def match_time_length(r, o, time_axis=TIME_AXIS):
    """
    raw + flatten 조합에서만 사용.
    method마다 저장된 time token 수가 다를 수 있으므로(dynamic_pearl의 prompt 등),
    어느 쪽이 더 길든 상관없이 둘 다 time_axis 축 기준 뒤쪽 min(len_r, len_o)개로 잘라 길이를 맞춥니다.
    """
    axis = time_axis % r.dim()
    r_time_len = r.shape[axis]
    o_time_len = o.shape[axis]
    min_time_len = min(r_time_len, o_time_len)

    if r_time_len != min_time_len:
        r = take_last_n_along_axis(r, min_time_len, axis)
    if o_time_len != min_time_len:
        o = take_last_n_along_axis(o, min_time_len, axis)

    return r, o


def to_flatten(x):
    # x: [N, C, T, D] -> [N, C*T*D]
    return x.flatten(start_dim=1)


def to_pooling(x):
    # x: [N, ?, ?, D] -> [N, D] (channel/time 두 축(dim=1, dim=2)을 평균내어 축소.
    # 두 축 모두에 대해 평균 내므로 channel/time의 축 순서와 무관하게 동일하게 동작함)
    return x.mean(dim=(1, 2))


def build_feature_pair(ref_x, other_x, slicing_mode, feature_mode, embed_num):
    """
    ref_x, other_x: 원본 raw feature [N, C, T, D] (아직 아무 처리도 하지 않은 상태)
    slicing_mode: 'raw' 또는 'sliced'
    feature_mode: 'flatten' 또는 'pooling'
    반환: (ref_vec, other_vec) - 각각 [N, D'] 형태의 벡터
    """
    r, o = ref_x, other_x

    if slicing_mode == "sliced":
        # 모든 method에서 뒤쪽 embed_num개 time token만 사용하므로
        # method간 원본 time token 수 차이와 무관하게 길이가 embed_num으로 통일됨
        r = apply_slicing(r, embed_num)
        o = apply_slicing(o, embed_num)
    else:  # raw
        if feature_mode == "flatten":
            # flatten은 원소 단위 정렬이 필요하므로 두 feature의 time token 길이를 서로 맞춤
            r, o = match_time_length(r, o)
        # pooling은 time 축을 평균으로 없애버리므로 길이가 달라도 그대로 사용

    if feature_mode == "flatten":
        r_vec = to_flatten(r)
        o_vec = to_flatten(o)
    elif feature_mode == "pooling":
        r_vec = to_pooling(r)
        o_vec = to_pooling(o)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    return r_vec, o_vec


# ==========================================
# 3. cosine distance 계산
# ==========================================
def cosine_distance_same_index(a, b):
    """같은 index 샘플끼리의 cosine distance 평균. a, b: [N, D]"""
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim = (a * b).sum(dim=1)
    return (1.0 - sim).mean().item()


def cosine_distance_full_pairwise(a, b):
    """모든 (i, j) 조합에 대한 cosine distance 평균. a: [N, D], b: [M, D]"""
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim_matrix = a @ b.T
    return (1.0 - sim_matrix).mean().item()


def cosine_distance_mean_vector(a, b):
    """전체 샘플을 평균 낸 단일 벡터 간의 cosine distance. a, b: [N, D]"""
    a_mean = F.normalize(a.mean(dim=0, keepdim=True), dim=1)
    b_mean = F.normalize(b.mean(dim=0, keepdim=True), dim=1)
    sim = (a_mean * b_mean).sum(dim=1)
    return (1.0 - sim).item()


# ==========================================
# 4. 메인 실험 루프
# ==========================================
def main():
    results = []

    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")

        for split_name, (split_dir, filename) in SPLITS.items():
            ref_folder = METHODS[REFERENCE_METHOD]
            ref_x = load_raw_feature(DATASET_PATH, split_dir, ref_folder, layer, filename)
            if ref_x is None:
                continue
            ref_x = ref_x.to(DEVICE)

            for method_name, method_folder in METHODS.items():
                if method_name == REFERENCE_METHOD:
                    continue

                other_x = load_raw_feature(DATASET_PATH, split_dir, method_folder, layer, filename)
                if other_x is None:
                    continue

                if ref_x.shape[0] != other_x.shape[0]:
                    print(
                        f"[경고] Layer {layer} {split_name} {method_name}: "
                        f"sample 수가 reference({ref_x.shape[0]})와 다릅니다({other_x.shape[0]}). 스킵합니다."
                    )
                    continue

                other_x = other_x.to(DEVICE)

                for slicing_mode in SLICING_MODES:
                    for feature_mode in FEATURE_MODES:
                        r_vec, o_vec = build_feature_pair(
                            ref_x, other_x, slicing_mode, feature_mode, EMBED_NUM
                        )

                        same_idx_dist = cosine_distance_same_index(r_vec, o_vec)
                        full_pairwise_dist = cosine_distance_full_pairwise(r_vec, o_vec)
                        mean_vec_dist = cosine_distance_mean_vector(r_vec, o_vec)

                        results.append({
                            "layer": layer,
                            "split": split_name,
                            "method": method_name,
                            "slicing_mode": slicing_mode,
                            "feature_mode": feature_mode,
                            "same_index_pairwise_mean": same_idx_dist,
                            "full_pairwise_mean": full_pairwise_dist,
                            "mean_vector_distance": mean_vec_dist,
                        })

                        print(
                            f"[Layer {layer}][{split_name}][{method_name}]"
                            f"[{slicing_mode}/{feature_mode}] "
                            f"same_idx={same_idx_dist:.4f} "
                            f"full_pairwise={full_pairwise_dist:.4f} "
                            f"mean_vec={mean_vec_dist:.4f}"
                        )

                del other_x
                gc.collect()
                torch.cuda.empty_cache()

            del ref_x
            gc.collect()
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    # train/valid/test 평균값 계산 후 'average' split으로 추가
    group_cols = ["layer", "method", "slicing_mode", "feature_mode"]
    metric_cols = ["same_index_pairwise_mean", "full_pairwise_mean", "mean_vector_distance"]
    avg_df = df.groupby(group_cols)[metric_cols].mean().reset_index()
    avg_df["split"] = "average"

    final_df = pd.concat([df, avg_df], ignore_index=True)
    final_df = final_df[["layer", "split", "method", "slicing_mode", "feature_mode"] + metric_cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # slicing_mode/feature_mode 조합별로 별도의 CSV 파일에 저장
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
