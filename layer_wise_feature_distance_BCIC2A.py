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

SLICING_MODES = ["raw", "sliced"]     # raw: 원본 feature, sliced: x[..., -EMBED_NUM:, :]
FEATURE_MODES = ["flatten", "pooling"]  # flatten: 그대로 펼침, pooling: channel/time 축 평균

OUTPUT_CSV = f"{DATASET_PATH}/layer_wise_feature_distance.csv"

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
def apply_slicing(x, embed_num):
    # x: [N, C, T, D] -> [N, C, embed_num, D] (뒤쪽 embed_num개 time token만 사용)
    return x[..., -embed_num:, :]


def match_time_length(x, ref_time_len):
    """
    raw + flatten 조합에서만 사용.
    dynamic_pearl처럼 prompt로 인해 time token 수가 reference보다 많은 경우,
    뒤쪽 ref_time_len개만 남기고 앞쪽(prompt 관련 토큰)을 잘라 길이를 맞춥니다.
    """
    cur_time_len = x.shape[-2]
    if cur_time_len == ref_time_len:
        return x
    if cur_time_len < ref_time_len:
        raise ValueError(
            f"feature의 time token 수({cur_time_len})가 reference({ref_time_len})보다 적습니다."
        )
    return x[..., -ref_time_len:, :]


def to_flatten(x):
    # x: [N, C, T, D] -> [N, C*T*D]
    return x.flatten(start_dim=1)


def to_pooling(x):
    # x: [N, C, T, D] -> [N, D] (channel: dim=1, time: dim=2 에 대해 평균)
    return x.mean(dim=(1, 2))


def build_feature_pair(ref_x, other_x, slicing_mode, feature_mode, embed_num, ref_raw_time_len):
    """
    ref_x, other_x: 원본 raw feature [N, C, T, D] (아직 아무 처리도 하지 않은 상태)
    slicing_mode: 'raw' 또는 'sliced'
    feature_mode: 'flatten' 또는 'pooling'
    반환: (ref_vec, other_vec) - 각각 [N, D'] 형태의 벡터
    """
    r, o = ref_x, other_x

    if slicing_mode == "sliced":
        # 모든 method에서 뒤쪽 embed_num개 time token만 사용하므로
        # dynamic_pearl의 prompt 토큰 여부와 무관하게 길이가 embed_num으로 통일됨
        r = apply_slicing(r, embed_num)
        o = apply_slicing(o, embed_num)
    else:  # raw
        if feature_mode == "flatten":
            # flatten은 원소 단위 정렬이 필요하므로 time token 길이를 reference 기준으로 맞춤
            o = match_time_length(o, ref_raw_time_len)
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
            ref_raw_time_len = ref_x.shape[-2]
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
                            ref_x, other_x, slicing_mode, feature_mode, EMBED_NUM, ref_raw_time_len
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

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[완료] 결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
