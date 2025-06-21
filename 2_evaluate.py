# 2_evaluate.py

import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import PeftModel
import numpy as np

# Import các class từ file utils.py
from utils import DualEncoderModel, MSRVTT_Dataset, CLIPProcessor

# ==============================================================================
# PHẦN 1: CẤU HÌNH ĐÁNH GIÁ
# ==============================================================================

def get_eval_config():
    """Hàm chứa tất cả các cấu hình cho việc đánh giá."""
    config = {
        # --- Đường dẫn ---
        "project_root_path": "/content/drive/MyDrive/Intern_FPT/AI/DATASET/", # THAY ĐỔI ĐƯỜNG DẪN NÀY
        
        # --- Cấu hình Mô hình ---
        "model_name": "openai/clip-vit-base-patch32",
        "projection_dim": 256,
        
        # --- Cấu hình Dữ liệu ---
        "num_frames": 16,
        
        # --- Cấu hình Đánh giá ---
        "eval_batch_size": 64, # Có thể dùng batch size lớn hơn khi suy luận
        # Chọn checkpoint để đánh giá
        "checkpoint_name": "final_checkpoint", # hoặc "checkpoint_step_X"
    }
    
    # Tính toán các đường dẫn dựa trên thư mục gốc
    config["data_path"] = os.path.join(config["project_root_path"], "data")
    config["preprocessed_data_path"] = os.path.join(config["data_path"], f"MSRVTT_Preprocessed_{config['num_frames']}frames")
    # Lấy đường dẫn từ cấu hình huấn luyện
    config["checkpoint_folder"] = os.path.join(config["project_root_path"], f"checkpoints_{config['model_name'].split('/')[-1]}_qlora")
    config["adapter_path"] = os.path.join(config["checkpoint_folder"], config["checkpoint_name"])

    return config

# ==============================================================================
# PHẦN 2: HÀM ĐÁNH GIÁ CHI TIẾT
# ==============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Hàm đánh giá chi tiết cho MSR-VTT retrieval.
    """
    model.eval()
    
    all_video_ids = []
    all_video_embeds = []
    all_text_embeds = []
    
    # --- 1. Tạo embedding cho tất cả các mẫu trong dataloader ---
    print("Bắt đầu tạo embedding cho tập dữ liệu test...")
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        # Chuyển batch lên GPU
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Lấy video_ids từ batch (cần sửa lại class Dataset để trả về)
        all_video_ids.extend(batch["video_id"])
        
        # Tính embedding
        video_embeds, text_embeds = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        all_video_embeds.append(video_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
        
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # --- 2. Xử lý video embedding: một video chỉ có một embedding duy nhất ---
    # Nhóm các video embedding theo video_id và lấy trung bình
    video_embeds_dict = {}
    video_id_to_idx = {}
    unique_video_ids = sorted(list(set(all_video_ids)))
    
    temp_video_embeds = torch.cat(all_video_embeds, dim=0)

    for i, video_id in enumerate(all_video_ids):
        if video_id not in video_embeds_dict:
            video_embeds_dict[video_id] = []
        video_embeds_dict[video_id].append(temp_video_embeds[i])

    # Tính embedding trung bình cho mỗi video
    final_video_embeds = torch.stack(
        [torch.stack(video_embeds_dict[vid]).mean(dim=0) for vid in unique_video_ids]
    )
    
    for i, vid in enumerate(unique_video_ids):
        video_id_to_idx[vid] = i
        
    # --- 3. Tính toán ma trận tương đồng ---
    sim_matrix = torch.matmul(all_text_embeds, final_video_embeds.T)
    
    # --- 4. Đánh giá Text-to-Video Retrieval ---
    print("\nĐang đánh giá Text-to-Video Retrieval...")
    ranks = []
    for i in range(len(all_video_ids)):
        text_id = i
        ground_truth_video_id = all_video_ids[i]
        ground_truth_video_idx = video_id_to_idx[ground_truth_video_id]
        
        # Sắp xếp các video theo độ tương đồng với text i
        sorted_indices = torch.argsort(sim_matrix[text_id, :], descending=True)
        
        # Tìm rank của video đúng
        rank = (sorted_indices == ground_truth_video_idx).nonzero(as_tuple=True)[0].item()
        ranks.append(rank + 1)
        
    ranks = torch.tensor(ranks)
    r1 = (ranks <= 1).float().mean().item() * 100
    r5 = (ranks <= 5).float().mean().item() * 100
    r10 = (ranks <= 10).float().mean().item() * 100
    mdr = torch.median(ranks).item()
    
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MdR": mdr}
    

# ==============================================================================
# PHẦN 3: HÀM CHÍNH ĐỂ CHẠY ĐÁNH GIÁ
# ==============================================================================
def main():
    config = get_eval_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    # Cần sửa lại class Dataset để trả về cả video_id
    class MSRVTT_Eval_Dataset(MSRVTT_Dataset):
        def __getitem__(self, idx):
            # Lấy dữ liệu gốc từ class cha
            data = super().__getitem__(idx)
            # Thêm video_id vào
            data['video_id'] = self.data[idx]['video_id']
            return data
            
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    test_json_path = os.path.join(config["preprocessed_data_path"], 'test_data.json')
    if not os.path.exists(test_json_path):
        print(f"LỖI: Không tìm thấy file test_data.json tại {test_json_path}")
        print("Vui lòng chạy lại script 0_preprocess_data.py để tạo file test.")
        return
        
    test_dataset = MSRVTT_Eval_Dataset(
        json_file_path=test_json_path,
        preprocessed_data_folder=config["preprocessed_data_path"],
        processor=processor,
        num_frames=config["num_frames"]
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["eval_batch_size"], shuffle=False, num_workers=2
    )

    # --- 3. Chạy đánh giá ---
    results = evaluate(eval_model, test_dataloader, device)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG ---")
    print(json.dumps(results, indent=4))
    
    # Lưu kết quả
    results_path = os.path.join(config["checkpoint_folder"], f"results_{config['checkpoint_name']}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Đã lưu kết quả vào: {results_path}")

if __name__ == "__main__":
    main()