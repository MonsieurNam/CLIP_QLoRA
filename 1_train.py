# 1_train.py
# Script hoàn chỉnh để fine-tuning mô hình Dual-Encoder cho Text-Video Retrieval
# sử dụng QLoRA và Accelerate.

import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import glob
import time
import json

# Import các class từ file utils.py
# Đảm bảo file utils.py nằm cùng thư mục
from utils import *

# ==============================================================================
# PHẦN 1: CẤU HÌNH DỰ ÁN
# ==============================================================================

def get_config():
    """Hàm tập trung tất cả các cấu hình của dự án."""
    
    # --- Tự động phát hiện kiểu dữ liệu được hỗ trợ ---
    # bf16 cho Ampere (RTX 30xx/40xx, A100) trở lên, fp16 cho các GPU cũ hơn
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mixed_precision_mode = "bf16"
    else:
        mixed_precision_mode = "fp16"

    config = {
        # --- Đường dẫn ---
        "project_root_path": "/root/CLIP_QLoRA", # !!! THAY ĐỔI ĐƯỜNG DẪN NÀY cho phù hợp !!!
        
        # --- Cấu hình Mô hình ---
        "model_name": "openai/clip-vit-base-patch32",
        "projection_dim": 256,
        
        # --- Cấu hình Dữ liệu ---
        "num_frames": 16,
        
        # --- Cấu hình Huấn luyện ---
        "num_epochs": 5,
        "batch_size": 32,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "mixed_precision": mixed_precision_mode,
        
        # --- Cấu hình QLoRA ---
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        
        # --- Cấu hình Lưu trữ ---
        "checkpointing_steps": 1000, # Lưu sau mỗi 1000 bước tối ưu
    }
    
    # Tính toán các đường dẫn phụ thuộc
    config["data_path"] = os.path.join(config["project_root_path"], "content")
    config["preprocessed_data_path"] = os.path.join(config["data_path"], f"MSRVTT_Preprocessed_{config['num_frames']}frames")
    config["checkpoint_folder"] = os.path.join(config["project_root_path"], f"checkpoints_{config['model_name'].split('/')[-1]}_qlora")

    return config

# ==============================================================================
# PHẦN 2: HÀM HUẤN LUYỆN CHÍNH
# ==============================================================================

def main():
    config = get_config()
    os.makedirs(config["checkpoint_folder"], exist_ok=True)
    
    # --- 1. Khởi tạo Accelerator ---
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=config["checkpoint_folder"]
    )
    
    accelerator.print(f"--- Bắt đầu dự án Fine-tuning Video Retrieval ---")
    accelerator.print(f"Sử dụng device: {accelerator.device}")
    accelerator.print(f"Cấu hình đang sử dụng: {json.dumps(config, indent=2)}")

    # --- 2. Tải và Cấu hình Mô hình với QLoRA ---
    if accelerator.is_main_process:
        print("Tải mô hình và áp dụng QLoRA...")
        
    precision_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=precision_map[config["mixed_precision"]],
        bnb_4bit_use_double_quant=True,
    )
    
    # Tải mô hình gốc, PEFT sẽ xử lý việc lượng tử hóa sau
    base_model = DualEncoderModel(model_name=config["model_name"], projection_dim=config["projection_dim"])
    
    # Chuẩn bị mô hình cho k-bit training (quan trọng để ổn định)
    base_model = prepare_model_for_kbit_training(base_model)
    
    lora_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "text_projection", "video_projection"],
        lora_dropout=config["lora_dropout"], bias="none", task_type="FEATURE_EXTRACTION"
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    
    if accelerator.is_main_process:
        lora_model.print_trainable_parameters()

    # --- 3. Chuẩn bị Dữ liệu ---
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    train_dataset = MSRVTT_Dataset(
        json_file_path=os.path.join(config["preprocessed_data_path"], 'train_data.json'),
        preprocessed_data_folder=config["preprocessed_data_path"],
        processor=processor,
        num_frames=config["num_frames"]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=os.cpu_count() // 2
    )
    
    # --- 4. Chuẩn bị Optimizer, Scheduler, Loss ---
    optimizer = bnb.optim.PagedAdamW8bit(lora_model.parameters(), lr=config["learning_rate"])
    loss_fn = InfoNCELoss()
    
    num_update_steps_per_epoch = -(-len(train_dataloader) // config["gradient_accumulation_steps"]) # Làm tròn lên
    total_training_steps = config["num_epochs"] * num_update_steps_per_epoch
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=int(0.1 * total_training_steps), num_training_steps=total_training_steps
    )
    
    # --- 5. Chuẩn bị với Accelerator và Logic Tải Checkpoint ---
    lora_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_model, optimizer, train_dataloader, lr_scheduler
    )

    checkpoint_search_path = os.path.join(config["checkpoint_folder"], "checkpoint_step_*")
    checkpoints = sorted(glob.glob(checkpoint_search_path), key=lambda x: int(x.split("_")[-1]))
    
    start_epoch = 0
    completed_steps = 0
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        accelerator.print(f"Phát hiện checkpoint. Tải lại trạng thái từ: {latest_checkpoint}")
        accelerator.load_state(latest_checkpoint)
        completed_steps = int(latest_checkpoint.split("_")[-1])
        start_epoch = completed_steps // num_update_steps_per_epoch
    else:
        accelerator.print("Không tìm thấy checkpoint. Bắt đầu huấn luyện từ đầu.")

    # --- 6. Vòng lặp Huấn luyện ---
    accelerator.print("--- Bắt đầu quá trình fine-tuning ---")
    lora_model.train()

    progress_bar = tqdm(
        range(completed_steps, total_training_steps), 
        disable=not accelerator.is_main_process,
        desc="Tổng tiến trình"
    )

    for epoch in range(start_epoch, config["num_epochs"]):
        epoch_total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # Bỏ qua các bước đã hoàn thành ở epoch đầu tiên
            if checkpoints and epoch == start_epoch and step < (completed_steps % num_update_steps_per_epoch):
                continue
                
            with accelerator.accumulate(lora_model):
                video_embeds, text_embeds = lora_model(**batch)
                loss = loss_fn(video_embeds, text_embeds)
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Chỉ cập nhật loss và thanh tiến trình khi có một bước optimizer
            if accelerator.sync_gradients:
                progress_bar.update(1)
                avg_loss = accelerator.gather(loss.repeat(config["batch_size"])).mean()
                epoch_total_loss += avg_loss.item()
                progress_bar.set_postfix(epoch=epoch+1, loss=avg_loss.item())

                # Lưu checkpoint định kỳ
                if progress_bar.n % config["checkpointing_steps"] == 0:
                    if accelerator.is_main_process:
                        output_dir = os.path.join(config["checkpoint_folder"], f"checkpoint_step_{progress_bar.n}")
                        accelerator.save_state(output_dir)
                        accelerator.print(f"\nĐã lưu checkpoint tại bước {progress_bar.n}")

        # In loss trung bình cuối mỗi epoch
        if accelerator.is_main_process:
             epoch_loss = epoch_total_loss / num_update_steps_per_epoch
             accelerator.print(f"Epoch {epoch + 1} | Average Loss: {epoch_loss:.4f}")

    accelerator.print("--- HUẤN LUYỆN HOÀN TẤT! ---")
    if accelerator.is_main_process:
        final_checkpoint_dir = os.path.join(config["checkpoint_folder"], "final_checkpoint")
        accelerator.save_state(final_checkpoint_dir)
        accelerator.print(f"Đã lưu trạng thái huấn luyện cuối cùng tại: {final_checkpoint_dir}")

# ==============================================================================
# PHẦN 3: ĐIỂM BẮT ĐẦU SCRIPT
# ==============================================================================

if __name__ == "__main__":
    # notebook_launcher(main, num_processes=1) # Dùng dòng này để chạy trong notebook
    main() # Dùng dòng này để chạy bằng `accelerate launch`
    # accelerate launch 1_train.py