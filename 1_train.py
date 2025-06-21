# train.py
# Script huấn luyện cuối cùng, đã được làm sạch và tối ưu hóa.

import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_linear_schedule_with_warmup, BitsAndBytesConfig
from accelerate import Accelerator, notebook_launcher
from tqdm.auto import tqdm
import glob
import json
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import các class đã được nâng cấp từ utils.py
from utils import DualEncoderModel, MSRVTT_Dataset, InfoNCELoss

# ==============================================================================
# PHẦN 1: CẤU HÌNH DỰ ÁN
# ==============================================================================
def get_config(env="4090"):
    """Hàm tập trung tất cả các cấu hình của dự án."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mixed_precision_mode = "bf16"
    else:
        mixed_precision_mode = "fp16"

    # Cấu hình chung
    base_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "projection_dim": 256,
        "num_frames": 16,
        "num_epochs": 5,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "checkpointing_steps": 500,
        "validation_steps": 500,
    }

    # Cấu hình riêng cho từng môi trường
    if env.lower() == "4090":
        print(f">> Đang tải cấu hình cho RTX 4090...")
        env_config = {
            "project_root_path": "/root/CLIP_QLoRA",
            "batch_size": 32,
            "gradient_accumulation_steps": 2, # Effective BS = 64
            "learning_rate": 2e-5,
            "mixed_precision": mixed_precision_mode,
            "lora_rank": 32,
        }
    elif env.lower() == "colab":
        print(f">> Đang tải cấu hình cho Colab T4...")
        env_config = {
            "project_root_path": "/content/drive/MyDrive/CLIP_QLoRA_MSRVTT",
            "batch_size": 8,
            "gradient_accumulation_steps": 8, # Effective BS = 64
            "learning_rate": 3e-5,
            "mixed_precision": "fp16",
            "lora_rank": 16,
        }
    else:
        raise ValueError(f"Môi trường không xác định: {env}. Chỉ hỗ trợ '4090' hoặc 'colab'.")

    config = {**base_config, **env_config}
    config["output_dir"] = os.path.join(config["project_root_path"], f"output_{env}")
    return config

# ==============================================================================
# PHẦN 2: HÀM ĐÁNH GIÁ
# ==============================================================================
@torch.no_grad()
def evaluate(model, dataloader, loss_fn, accelerator):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Đang đánh giá", disable=not accelerator.is_main_process, leave=False):
        video_embeds, text_embeds = model(**batch)
        loss = loss_fn(video_embeds, text_embeds)
        total_loss += accelerator.gather_for_metrics(loss).mean().item()
    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss

# ==============================================================================
# PHẦN 3: HÀM HUẤN LUYỆN CHÍNH
# ==============================================================================
def main(args):
    config = get_config(env=args.env)
    os.makedirs(config["output_dir"], exist_ok=True)
    
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=config["output_dir"]
    )
    
    accelerator.init_trackers("video_retrieval_training")
    accelerator.print(f"Cấu hình: {json.dumps(config, indent=2)}")

    # --- 1. Tải và Cấu hình Mô hình (THEO QUY TRÌNH CHUẨN MỰC MỚI) ---
    accelerator.print("Tải mô hình lượng tử hóa và chuẩn bị cho PEFT...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, config["mixed_precision"]),
        bnb_4bit_use_double_quant=True
    )
    
    # Bước 1: Tải mô hình đã được lượng tử hóa ngay từ đầu.
    # `utils.py` đã được thiết kế để nhận config này.
    base_model = DualEncoderModel(
        model_name=config["model_name"], 
        projection_dim=config["projection_dim"],
        quantization_config=bnb_config
    )
    
    # Bước 2: Chuẩn bị mô hình đã lượng tử hóa cho việc huấn luyện k-bit.
    # Hàm này sẽ xử lý các lớp LayerNorm và bật gradient checkpointing.
    base_model = prepare_model_for_kbit_training(base_model)
    
    # Bước 3: Định nghĩa LoraConfig.
    lora_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "mlp.fc1", "mlp.fc2",
            "video_projection", "text_projection",
            "temporal_encoder.layers.0.linear1", "temporal_encoder.layers.0.linear2",
            "temporal_encoder.layers.1.linear1", "temporal_encoder.layers.1.linear2",
        ],
        lora_dropout=config["lora_dropout"], bias="none", task_type="FEATURE_EXTRACTION"
    )
    
    # Bước 4: Áp dụng PEFT. Không cần truyền quantization_config ở đây nữa.
    lora_model = get_peft_model(base_model, lora_config)
    
    lora_model.print_trainable_parameters()
    # --- 2. Chuẩn bị Dữ liệu ---
    data_root = os.path.join(config["project_root_path"], "data", f"MSRVTT_Preprocessed_{config['num_frames']}frames")
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    
    train_dataset = MSRVTT_Dataset(
        json_file_path=os.path.join(data_root, 'train_data.json'),
        preprocessed_data_folder=data_root, processor=processor, num_frames=config["num_frames"]
    )
    val_dataset = MSRVTT_Dataset(
        json_file_path=os.path.join(data_root, 'val_data.json'),
        preprocessed_data_folder=data_root, processor=processor, num_frames=config["num_frames"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=2)
    
    # --- 3. Chuẩn bị Optimizer, Scheduler, Loss ---
    optimizer = bnb.optim.PagedAdamW8bit(lora_model.parameters(), lr=config["learning_rate"])
    loss_fn = InfoNCELoss()
    
    num_update_steps_per_epoch = -(-len(train_dataloader) // config["gradient_accumulation_steps"])
    total_training_steps = config["num_epochs"] * num_update_steps_per_epoch
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=int(0.1 * total_training_steps),
        num_training_steps=total_training_steps
    )
    
    # --- 4. Chuẩn bị với Accelerator và Tải Checkpoint ---
    lora_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        lora_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    resume_from_checkpoint = None
    if os.path.exists(config["output_dir"]):
        checkpoints = glob.glob(os.path.join(config["output_dir"], "checkpoint_step_*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            resume_from_checkpoint = latest_checkpoint
            accelerator.print(f"Phát hiện checkpoint. Sẽ tiếp tục từ: {resume_from_checkpoint}")

    completed_steps = 0
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)
        completed_steps = int(resume_from_checkpoint.split("_")[-1])

    # --- 5. Vòng lặp Huấn luyện ---
    progress_bar = tqdm(range(completed_steps, total_training_steps), disable=not accelerator.is_main_process)
    start_epoch = completed_steps // num_update_steps_per_epoch
    
    for epoch in range(start_epoch, config["num_epochs"]):
        lora_model.train()
        for step, batch in enumerate(train_dataloader):
            if resume_from_checkpoint and epoch == start_epoch and step < (completed_steps % num_update_steps_per_epoch):
                continue

            with accelerator.accumulate(lora_model):
                video_embeds, text_embeds = lora_model(**batch)
                loss = loss_fn(video_embeds, text_embeds)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                current_step = progress_bar.n
                
                avg_loss = accelerator.gather_for_metrics(loss).mean()
                accelerator.log({"train_loss": avg_loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=current_step)
                progress_bar.set_postfix(loss=avg_loss.item())

                if current_step > 0 and current_step % config["checkpointing_steps"] == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        adapter_dir = os.path.join(config["output_dir"], f"adapter_step_{current_step}")
                        state_dir = os.path.join(config["output_dir"], f"checkpoint_step_{current_step}")
                        
                        unwrapped_model = accelerator.unwrap_model(lora_model)
                        unwrapped_model.save_pretrained(adapter_dir)
                        accelerator.save_state(state_dir)
                        accelerator.print(f"\nĐã lưu adapter tại {adapter_dir}")
                
                if current_step > 0 and current_step % config["validation_steps"] == 0:
                    val_loss = evaluate(lora_model, val_dataloader, loss_fn, accelerator)
                    accelerator.log({"val_loss": val_loss}, step=current_step)
                    accelerator.print(f"\nBước {current_step} | Validation Loss: {val_loss:.4f}")

    # --- 6. Lưu Checkpoint Cuối cùng ---
    accelerator.print("--- HUẤN LUYỆN HOÀN TẤT! ---")
    if accelerator.is_main_process:
        final_adapter_dir = os.path.join(config["output_dir"], "final_adapter")
        unwrapped_model = accelerator.unwrap_model(lora_model)
        unwrapped_model.save_pretrained(final_adapter_dir)
        accelerator.print(f"Đã lưu adapter cuối cùng tại: {final_adapter_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script cho Text-Video Retrieval.")
    parser.add_argument(
        "--env", type=str, default="colab", choices=["4090", "colab"],
        help="Môi trường để chạy huấn luyện: '4090' hoặc 'colab'."
    )
    args = parser.parse_args()
    
    # Để chạy trong notebook
    # notebook_launcher(main, args=(args,), num_processes=1)
    
    # Để chạy từ terminal
    main(args)