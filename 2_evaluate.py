# evaluate.py
# Script đánh giá cuối cùng, đã được làm sạch và tối ưu hóa.
# - Loại bỏ custom_collate_fn không cần thiết.
# - Cải thiện logic tải mô hình và đánh giá.

import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import PeftModel
import argparse
from transformers import CLIPProcessor

# Import các class từ file utils.py
from utils import DualEncoderModel, MSRVTT_Dataset

# ==============================================================================
# PHẦN 1: CẤU HÌNH ĐÁNH GIÁ
# ==============================================================================
def get_eval_config(env="colab"):
    """Hàm chứa tất cả các cấu hình cho việc đánh giá."""
    
    base_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "projection_dim": 256,
        "num_frames": 16,
        "eval_batch_size": 128, # Tăng batch size khi suy luận để nhanh hơn
        "adapter_folder_name": "final_adapter", 
    }

    if env.lower() == "4090":
        print(f">> Đang tải cấu hình cho RTX 4090...")
        env_config = {"project_root_path": "/root/CLIP_QLoRA"}
    elif env.lower() == "colab":
        print(f">> Đang tải cấu hình cho Colab T4...")
        env_config = {"project_root_path": "/content/drive/MyDrive/CLIP_QLoRA_MSRVTT"}
    else:
        raise ValueError(f"Môi trường không xác định: {env}. Chỉ hỗ trợ '4090' hoặc 'colab'.")

    config = {**base_config, **env_config}
    config["output_dir"] = os.path.join(config["project_root_path"], f"output_{env}")
    config["adapter_path"] = os.path.join(config["output_dir"], config["adapter_folder_name"])
    config["preprocessed_data_path"] = os.path.join(config["project_root_path"], "data", f"MSRVTT_Preprocessed_{config['num_frames']}frames")
    
    return config

# ==============================================================================
# PHẦN 2: HÀM ĐÁNH GIÁ CHI TIẾT
# ==============================================================================
@torch.no_grad()
def evaluate_retrieval(model, dataloader, device):
    """Hàm đánh giá chi tiết cho MSR-VTT, đã được tối ưu hóa."""
    model.eval()
    
    all_text_embeds, video_id_to_embed, text_to_video_id_map = [], {}, []
    
    print("Bắt đầu tạo embedding cho tập test...")
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        # Chuyển dữ liệu sang GPU
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Sửa: pixel_values từ Dataloader mặc định đã có đúng shape
        # (batch_size, num_frames, C, H, W)
        video_embeds_batch, text_embeds_batch = model(**batch)
        
        all_text_embeds.append(text_embeds_batch.cpu())
        text_to_video_id_map.extend(batch["video_id"]) # video_id là một list of strings

        for i, vid in enumerate(batch["video_id"]):
            if vid not in video_id_to_embed:
                video_id_to_embed[vid] = video_embeds_batch[i].cpu()
                
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    unique_video_ids = sorted(video_id_to_embed.keys())
    all_video_embeds = torch.stack([video_id_to_embed[vid] for vid in unique_video_ids], dim=0)
    video_id_to_idx = {vid: i for i, vid in enumerate(unique_video_ids)}
    
    all_text_embeds = all_text_embeds.to(device)
    all_video_embeds = all_video_embeds.to(device)

    print("\nTính toán ma trận tương đồng Text-to-Video...")
    sim_matrix_t2v = torch.matmul(all_text_embeds, all_video_embeds.T)
    
    print("Đang đánh giá Text-to-Video Retrieval...")
    ground_truth_indices = torch.tensor([video_id_to_idx[vid] for vid in text_to_video_id_map], device=device)
    sorted_indices = torch.argsort(sim_matrix_t2v, descending=True, dim=1)
    ranks_tensor = (sorted_indices == ground_truth_indices.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    ranks = ranks_tensor.cpu()

    r1 = (ranks <= 1).float().mean().item() * 100
    r5 = (ranks <= 5).float().mean().item() * 100
    r10 = (ranks <= 10).float().mean().item() * 100
    mdr = torch.median(ranks.float()).item()
    
    return {"T2V_R@1": r1, "T2V_R@5": r5, "T2V_R@10": r10, "T2V_MdR": mdr}

# ==============================================================================
# PHẦN 3: HÀM CHÍNH
# ==============================================================================
def main(args):
    config = get_eval_config(env=args.env)
    if args.adapter:
        config["adapter_path"] = args.adapter
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng device: {device}")
    
    if not os.path.isdir(config["adapter_path"]):
        print(f"LỖI: Không tìm thấy thư mục adapter tại '{config['adapter_path']}'")
        return

    # --- 1. Tải mô hình đã fine-tune ---
    print("Tải mô hình gốc (FP32/BF16)...")
    base_model = DualEncoderModel(model_name=config["model_name"], projection_dim=config["projection_dim"])
    
    print(f"Tải LoRA adapter từ '{config['adapter_path']}'...")
    eval_model = PeftModel.from_pretrained(base_model, config["adapter_path"])
    
    print("Hợp nhất adapter vào mô hình gốc để suy luận nhanh nhất...")
    eval_model = eval_model.merge_and_unload()
    eval_model.to(device)

    # --- 2. Chuẩn bị dữ liệu test ---
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    test_json_path = os.path.join(config["preprocessed_data_path"], 'test_data.json')
    
    test_dataset = MSRVTT_Dataset(
        json_file_path=test_json_path,
        preprocessed_data_folder=config["preprocessed_data_path"],
        processor=processor,
        num_frames=config["num_frames"]
    )
    # **SỬA LỖI QUAN TRỌNG:** Loại bỏ collate_fn
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config["eval_batch_size"], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )

    # --- 3. Chạy đánh giá ---
    results = evaluate_retrieval(eval_model, test_dataloader, device)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG ---")
    print(json.dumps(results, indent=4))
    
    results_path = os.path.join(config["adapter_path"], "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Đã lưu kết quả vào: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script cho Text-Video Retrieval.")
    parser.add_argument("--env", type=str, default="colab", choices=["4090", "colab"], help="Môi trường để chạy.")
    parser.add_argument("--adapter", type=str, default=None, help="Đường dẫn trực tiếp đến thư mục adapter.")
    args = parser.parse_args()
    main(args)