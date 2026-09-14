"""
Resume-merge helper for feature_extraction_and_linear_probe_TUAB.py /
feature_extraction_and_linear_probe_TUEV.py.

Use this when a split's feature EXTRACTION already finished (all
batch_*.pt files exist under TEMP_ROOT/{split}/batches/) but the process
was killed or interrupted during the slow merge step (the old code that
re-read every batch file twice per layer).

It does NOT reload the model or redo the GPU forward pass. It only
re-iterates the dataset (cheap, CPU-only, no GPU) to recover the labels /
subject ids in the exact same order as the original run (the DataLoader
uses shuffle=False, so the order is deterministic), then merges the
already-extracted batch files with the new single-pass logic and writes
temp_dir/{split}/layer_{n}.pt.

After this finishes for a split, the corresponding main script
(feature_extraction_and_linear_probe_TUAB.py / _TUEV.py) will detect that
layer_*.pt already exist for that split and skip re-extracting it.

IMPORTANT:
  1. Kill the old running process first.
  2. Confirm TEMP_ROOT/{split}/batches/ still has all its batch_*.pt files
     (they are only deleted after every layer for that split has been
     successfully merged and saved).

Usage:
    python resume_merge.py TUAB train
    python resume_merge.py TUAB valid
    python resume_merge.py TUAB test
    python resume_merge.py TUEV train
"""

import argparse
import importlib
import gc
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def resume_merge(pipeline, split_name):
    assert split_name in ("train", "valid", "test")

    opts, _ = pipeline.get_args()
    train_dataset, test_dataset, val_dataset = pipeline.get_dataset(opts)
    dataset = {"train": train_dataset, "valid": val_dataset, "test": test_dataset}[split_name]

    split_temp_dir = os.path.join(pipeline.TEMP_ROOT, split_name)
    batch_temp_dir = os.path.join(split_temp_dir, "batches")

    if not os.path.isdir(batch_temp_dir):
        raise FileNotFoundError(
            f"{batch_temp_dir} 가 없습니다. '{split_name}' split은 아직 추출된 적이 없거나 "
            f"이미 정리(삭제)된 상태입니다. 이 스크립트가 아니라 원래 스크립트로 처음부터 추출하세요."
        )

    # 원래 추출과 동일한 batch_size / shuffle=False 를 써야 배치 순서가 일치한다.
    loader = DataLoader(dataset, batch_size=pipeline.EXTRACT_BATCH_SIZE, num_workers=0, shuffle=False)
    num_batches = len(loader)

    existing_batches = sorted(
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(batch_temp_dir)
        if f.startswith("batch_") and f.endswith(".pt")
    )
    if existing_batches != list(range(num_batches)):
        raise RuntimeError(
            f"[{split_name}] batches 폴더에 있는 배치 파일 개수/번호({len(existing_batches)}개)가 "
            f"기대값({num_batches}개, batch_0.pt ~ batch_{num_batches - 1}.pt)과 다릅니다. "
            f"feature 추출 자체가 끝나지 않았을 수 있으니, 이 경우엔 이 resume 스크립트를 쓰지 말고 "
            f"원래 스크립트로 이 split을 처음부터 다시 추출하세요."
        )

    print(f"[{split_name}] 배치 파일 {num_batches}개 확인 완료.")
    print(f"[{split_name}] label/subject_id만 재수집합니다 (GPU/모델 사용 안 함, 매우 빠름).")

    # feature는 이미 디스크에 있으므로 모델 forward 없이 label/subject_id만 다시 모은다.
    y_list = []
    s_list = []
    for _, labels, subject_ids in tqdm(loader, desc=f"[{split_name}] Re-collecting labels"):
        y_list.append(labels)
        if torch.is_tensor(subject_ids):
            s_list.extend(subject_ids.tolist())
        else:
            s_list.extend(subject_ids)

    entire_dataset_length = len(dataset)
    y_tensor = torch.cat(y_list, dim=0)
    del y_list
    gc.collect()

    # ---- 여기서부터는 수정된(chunk 단위 merge) 로직과 동일 ----
    # LAYER_MERGE_CHUNK_SIZE개 layer씩 묶어서 배치를 순회하며 채운다. chunk_size가
    # 클수록 배치 파일을 다시 읽는 횟수는 줄지만 그만큼 layer tensor를 동시에
    # 메모리에 들고 있어야 하므로, RAM이 부족하면 pipeline 모듈의
    # LAYER_MERGE_CHUNK_SIZE를 줄여야 한다. (구버전 pipeline 모듈과의 호환을 위해
    # 없으면 가장 안전한 1로 처리)
    chunk_size = getattr(pipeline, "LAYER_MERGE_CHUNK_SIZE", 1)
    target_layers = pipeline.TARGET_LAYERS
    layer_chunks = [target_layers[i:i + chunk_size] for i in range(0, len(target_layers), chunk_size)]

    layer_paths = {}
    for chunk in layer_chunks:
        print(f"[{split_name}] Pre-allocating tensors for layers {chunk}...")
        first_batch = torch.load(os.path.join(batch_temp_dir, "batch_0.pt"), map_location='cpu', weights_only=False)
        layer_tensors = {}
        for layer_num in chunk:
            layer_idx = layer_num - 1
            feat = first_batch[f'layer_{layer_idx}']
            layer_tensors[layer_num] = torch.empty((entire_dataset_length, *feat.shape[1:]), dtype=feat.dtype)
        del first_batch
        gc.collect()

        current_idx = 0
        for step in tqdm(range(num_batches), desc=f"[{split_name}] Merging layers {chunk}"):
            batch_data = torch.load(os.path.join(batch_temp_dir, f"batch_{step}.pt"), map_location='cpu', weights_only=False)
            rows = batch_data[f'layer_{chunk[0] - 1}'].shape[0]
            for layer_num in chunk:
                layer_idx = layer_num - 1
                layer_tensors[layer_num][current_idx: current_idx + rows] = batch_data[f'layer_{layer_idx}']
            current_idx += rows
            del batch_data

        gc.collect()

        for layer_num in chunk:
            layer_tensor = layer_tensors.pop(layer_num)
            print(f"[{split_name}] Final shape for Layer {layer_num} = {layer_tensor.shape}")

            save_dict = {"x": layer_tensor, "y": y_tensor, "s": s_list}
            layer_path = os.path.join(split_temp_dir, f"layer_{layer_num}.pt")
            torch.save(save_dict, layer_path)
            layer_paths[layer_num] = layer_path

            del layer_tensor, save_dict
            gc.collect()

    torch.cuda.empty_cache()

    # batch 파일은 여기서 지우지 않는다 (문제 생기면 재시도할 수 있도록 남겨둠).
    # 확인 후 필요하면 batch_temp_dir을 직접 삭제해도 된다.
    print(f"[{split_name}] merge 완료 -> {split_temp_dir}")
    for layer_num, path in layer_paths.items():
        print(f"  layer_{layer_num}: {path}")

    return layer_paths


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["TUAB", "TUEV"], help="어떤 통합 스크립트의 temp를 재개할지")
    parser.add_argument("split", choices=["train", "valid", "test"], help="재개할 split")
    args = parser.parse_args()

    module_name = f"feature_extraction_and_linear_probe_{args.dataset}"
    pipeline = importlib.import_module(module_name)

    # pipeline.get_args()가 내부적으로 argparse.parse_args()로 sys.argv를 그대로
    # 다시 파싱하므로, 우리가 여기서 받은 "TUAB"/"train" 같은 인자가 남아있으면
    # "unrecognized arguments" 에러가 난다. get_args() 호출 전에 비워준다.
    sys.argv = sys.argv[:1]

    resume_merge(pipeline, args.split)
