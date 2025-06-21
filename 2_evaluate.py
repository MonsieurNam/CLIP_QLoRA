# 2_evaluate.py
# Script hoàn chỉnh để đánh giá mô hình Text-Video Retrieval trên MSR-VTT,
# xử lý đúng logic một-nhiều và có cấu hình linh hoạt.

import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import PeftModel
import numpy as np
import argparse

# Import các class từ file utils.py
# Đảm bảo file utils.py nằm cùng thư mục hoặc trong python path
from utils import DualEncoderModel, MSRVTT_Dataset, CLIPProcessor

# ==============================================================================
# PHẦN 1: CẤU HÌNH ĐÁNH GIÁ
# ==============================================================================

def get_eval_config(env="4090"):
    """Hàm chứa tất cả các cấu hình cho việc đánh giá, tùy thuộc vào môi trường."""
    
    # --- Cấu hình chung ---
    base_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "projection_dim": 256,
        "num_frames": 16,
        "eval_batch_size": 64,
        "checkpoint_name": "adapter_final_converted", # hoặc 'checkpoint_step_X'
    }

    # --- Cấu hình riêng cho từng môi trường ---
    if env.lower() == "4090":
        print(f">> Đang tải cấu hình đánh giá cho môi trường RTX 4090...")
        env_config = {
            "project_root_path": "/root/CLIP_QLoRA", # THAY ĐỔI NẾU CẦN
        }
    elif env.lower() == "colab":
        print(f">> Đang tải cấu hình đánh giá cho môi trường Colab T4...")
        env_config = {
            "project_root_path": "/content/drive/MyDrive/Colab_Projects/InteractiveVideoRetrieval",
        }
    else:
        raise ValueError(f"Môi trường không xác định: {env}. Chỉ hỗ trợ '4090' hoặc 'colab'.")

    # Gộp cấu hình
    config = {**base_config, **env_config}

    # Tính toán các đường dẫn phụ thuộc
    config["data_path"] = os.path.join(config["project_root_path"], "data")
    config["preprocessed_data_path"] = os.path.join(config["data_path"], f"MSRVTT_Preprocessed_{config['num_frames']}frames")
    config["checkpoint_folder"] = "/root/CLIP_QLoRA/"
    config["adapter_path"] = os.path.join(config["checkpoint_folder"], config["checkpoint_name"])
    
    return config

# ==============================================================================
# PHẦN 2: HÀM ĐÁNH GIÁ CHI TIẾT (LOGIC MỚI)
# ==============================================================================

@torch.no_grad()
def evaluate_msrvtt(model, dataloader, device):
    """
    Hàm đánh giá chi tiết cho MSR-VTT retrieval, xử lý đúng logic một-nhiều.
    """
    model.eval()
    
    # --- 1. Tạo embedding cho tất cả các mẫu trong dataloader ---
    print("Bắt đầu tạo embedding cho tập dữ liệu test...")
    
    all_text_embeds = []
    video_id_to_embed = {}
    all_text_video_ids = []

    for batch in tqdm(dataloader, desc="Generating embeddings"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        video_ids_batch = batch["video_id"]
        
        all_text_video_ids.extend(video_ids_batch)
        
        video_embeds, text_embeds = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        all_text_embeds.append(text_embeds.cpu())

        for i, vid in enumerate(video_ids_batch):
            if vid not in video_id_to_embed:
                video_id_to_embed[vid] = video_embeds[i].cpu()
                
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    # --- 2. Chuẩn bị các tensor cuối cùng để tính toán ---
    unique_video_ids = sorted(video_id_to_embed.keys())
    all_video_embeds = torch.stack([video_id_to_embed[vid] for vid in unique_video_ids], dim=0)
    video_id_to_idx = {vid: i for i, vid in enumerate(unique_video_ids)}
    
    all_text_embeds = all_text_embeds.to(device)
    all_video_embeds = all_video_embeds.to(device)

    # --- 3. Tính toán ma trận tương đồng ---
    print("\nTính toán ma trận tương đồng...")
    sim_matrix = torch.matmul(all_text_embeds, all_video_embeds.T)
    
    # --- 4. Đánh giá Text-to-Video Retrieval ---
    print("Đang đánh giá Text-to-Video Retrieval...")
    ranks = []
    for i in range(sim_matrix.shape[0]):
        ground_truth_video_id = all_text_video_ids[i]
        ground_truth_video_idx = video_id_to_idx[ground_truth_video_id]
        
        sim_scores_for_text_i = sim_matrix[i]
        sorted_indices = torch.argsort(sim_scores_for_text_i, descending=True)
        
        rank = (sorted_indices == ground_truth_video_idx).nonzero(as_tuple=True)[0].item()
        ranks.append(rank + 1)
        
    ranks = torch.tensor(ranks, dtype=torch.float32)
    r1 = (ranks <= 1).float().mean().item() * 100
    r5 = (ranks <= 5).float().mean().item() * 100
    r10 = (ranks <= 10).float().mean().item() * 100
    mdr = torch.median(ranks).item()
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MdR": mdr}

# ==============================================================================
# PHẦN 3: HÀM CHÍNH ĐỂ CHẠY ĐÁNH GIÁ
# ==============================================================================
def main(args):
    config = get_eval_config(env=args.env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng device: {device}")
    print(f"Cấu hình đánh giá đang sử dụng: {json.dumps(config, indent=2)}")

    if not os.path.exists(config["adapter_path"]):
        print(f"LỖI: Không tìm thấy checkpoint LoRA tại {config['adapter_path']}")
        return

    # --- 1. Tải mô hình đã fine-tune ---
    print("Tải mô hình gốc...")
    base_model = DualEncoderModel(model_name=config["model_name"], projection_dim=config["projection_dim"])
    
    print(f"Tải LoRA adapter từ {config['adapter_path']}...")
    eval_model = PeftModel.from_pretrained(base_model, config["adapter_path"])
    
    print("Hợp nhất adapter vào mô hình gốc để suy luận nhanh hơn...")
    eval_model = eval_model.merge_and_unload()
    eval_model.to(device)

    # --- 2. Chuẩn bị dữ liệu test ---
    class MSRVTT_Eval_Dataset(MSRVTT_Dataset):
        def __getitem__(self, idx):
            data = super().__getitem__(idx)
            data['video_id'] = self.data[idx]['video_id']
            return data

    def custom_collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        video_ids = [item['video_id'] for item in batch]
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "video_id": video_ids
        }
            
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    test_json_path = os.path.join(config["preprocessed_data_path"], 'test_data.json')
    if not os.path.exists(test_json_path):
        print(f"LỖI: Không tìm thấy file test_data.json tại {test_json_path}")
        return
        
    test_dataset = MSRVTT_Eval_Dataset(
        json_file_path=test_json_path,
        preprocessed_data_folder=config["preprocessed_data_path"],
        processor=processor,
        num_frames=config["num_frames"]
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config["eval_batch_size"], 
        shuffle=False, 
        num_workers=2,
        collate_fn=custom_collate_fn
    )

    # --- 3. Chạy đánh giá ---
    results = evaluate_msrvtt(eval_model, test_dataloader, device)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG ---")
    print(json.dumps(results, indent=4))
    
    results_path = os.path.join(config["adapter_path"], f"results_{config['checkpoint_name']}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Đã lưu kết quả vào: {results_path}")

# ==============================================================================
# PHẦN 4: ĐIỂM BẮT ĐẦU SCRIPT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script cho Text-Video Retrieval.")
    parser.add_argument(
        "--env",
        type=str,
        default="4090",
        choices=["4090", "colab"],
        help="Môi trường để tải checkpoint tương ứng: '4090' hoặc 'colab'."
    )
    args = parser.parse_args()
    main(args)