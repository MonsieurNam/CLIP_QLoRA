# 1_train.py

import os
import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_linear_schedule_with_warmup
from accelerate import Accelerator, notebook_launcher
from tqdm.auto import tqdm
import glob
import time
import json

# Import các class từ file utils.py
# Đảm bảo file utils.py nằm cùng thư mục hoặc trong python path
from utils import DualEncoderModel, MSRVTT_Dataset, InfoNCELoss, LoraConfig, get_peft_model, BitsAndBytesConfig

# ==============================================================================
# PHẦN 1: CẤU HÌNH DỰ ÁN
# ==============================================================================

def get_config():
    """Hàm chứa tất cả các cấu hình cho dự án."""
    config = {
        # --- Đường dẫn ---
        "project_root_path": "/root", # THAY ĐỔI ĐƯỜNG DẪN NÀY
        
        # --- Cấu hình Mô hình ---
        "model_name": "openai/clip-vit-base-patch32",
        "projection_dim": 256,
        
        # --- Cấu hình Dữ liệu ---
        "num_frames": 16,
        
        # --- Cấu hình Huấn luyện ---
        "num_epochs": 5,
        "batch_size": 32, # Bắt đầu với 32 cho RTX 4090, giảm nếu OOM
        "gradient_accumulation_steps": 2, # Effective batch size = 32 * 2 = 64
        "learning_rate": 5e-5,
        "mixed_precision": "bf16", # "bf16" cho RTX 4090/Ampere+, "fp16" cho các GPU khác
        
        # --- Cấu hình QLoRA ---
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        
        # --- Cấu hình Lưu trữ ---
        "checkpointing_steps": 5000, # Lưu sau mỗi 1000 bước tối ưu
    }
    
    # Tính toán các đường dẫn dựa trên thư mục gốc
    config["data_path"] = os.path.join(config["project_root_path"], "data")
    config["preprocessed_data_path"] = os.path.join(config["data_path"], f"MSRVTT_Preprocessed_{config['num_frames']}frames")
    config["checkpoint_folder"] = os.path.join(config["project_root_path"], f"checkpoints_{config['model_name'].split('/')[-1]}_qlora")

    return config

# ==============================================================================
# PHẦN 2: HÀM HUẤN LUYỆN CHÍNH
# ==============================================================================

def main():
    config = get_config()
    os.makedirs(config["checkpoint_folder"], exist_ok=True)
    
    # Khởi tạo Accelerator
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=config["checkpoint_folder"]
    )
    
    accelerator.print(f"--- Bắt đầu dự án Fine-tuning Video Retrieval ---")
    accelerator.print(f"Cấu hình đang sử dụng: {json.dumps(config, indent=2)}")

    # --- 1. Tải và Cấu hình Mô hình ---
    if accelerator.is_main_process:
        print("Tải mô hình gốc...")
    base_model = DualEncoderModel(model_name=config["model_name"], projection_dim=config["projection_dim"])
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, config["mixed_precision"]),
        bnb_4bit_use_double_quant=True,
    )
    lora_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "text_projection", "video_projection"],
        lora_dropout=config["lora_dropout"], bias="none", task_type="FEATURE_EXTRACTION"
    )

    # Dùng get_peft_model để bọc mô hình
    # Tải lại mô hình với config để PEFT xử lý đúng
    model_to_train = DualEncoderModel(model_name=config["model_name"], projection_dim=config["projection_dim"])
    lora_model = get_peft_model(model_to_train, lora_config, bnb_quantization_config=bnb_config)
    accelerator.print("\n--- Tổng quan tham số có thể huấn luyện ---")
    lora_model.print_trainable_parameters()

    # --- 2. Chuẩn bị Dữ liệu ---
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    train_dataset = MSRVTT_Dataset(
        json_file_path=os.path.join(config["preprocessed_data_path"], 'train_data.json'),
        preprocessed_data_folder=config["preprocessed_data_path"],
        processor=processor,
        num_frames=config["num_frames"]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    
    # --- 3. Chuẩn bị Optimizer, Scheduler, Loss ---
    optimizer = bnb.optim.PagedAdamW8bit(lora_model.parameters(), lr=config["learning_rate"])
    loss_fn = InfoNCELoss()
    
    num_update_steps_per_epoch = len(train_dataloader) // config["gradient_accumulation_steps"]
    total_training_steps = config["num_epochs"] * num_update_steps_per_epoch
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=int(0.1 * total_training_steps), num_training_steps=total_training_steps
    )
    
    # --- 4. Chuẩn bị với Accelerator ---
    lora_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_model, optimizer, train_dataloader, lr_scheduler
    )

    # --- 5. Logic Tải lại Checkpoint ---
    checkpoint_search_path = os.path.join(config["checkpoint_folder"], "checkpoint_step_*")
    checkpoints = sorted(glob.glob(checkpoint_search_path))
    
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

    for epoch in range(start_epoch, config["num_epochs"]):
        total_loss = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config['num_epochs']}",
            disable=not accelerator.is_main_process # Chỉ hiển thị thanh tiến trình ở tiến trình chính
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(lora_model):
                video_embeds, text_embeds = lora_model(**batch)
                loss = loss_fn(video_embeds, text_embeds)
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # Log loss
            avg_loss = accelerator.gather(loss.repeat(config["batch_size"])).mean()
            total_loss += avg_loss.item() / config["gradient_accumulation_steps"]
            
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=avg_loss.item())
                
            if (completed_steps + 1) % config["checkpointing_steps"] == 0:
                if accelerator.is_main_process:
                    output_dir = os.path.join(config["checkpoint_folder"], f"checkpoint_step_{completed_steps + 1}")
                    accelerator.save_state(output_dir)
                    accelerator.print(f"\nĐã lưu checkpoint tại bước {completed_steps + 1}")

            completed_steps += 1 if (step + 1) % config["gradient_accumulation_steps"] == 0 else 0

        # In loss trung bình cuối mỗi epoch
        if accelerator.is_main_process:
             epoch_loss = total_loss / (len(train_dataloader) / config["gradient_accumulation_steps"])
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
    main()
    # accelerate launch 1_train.py