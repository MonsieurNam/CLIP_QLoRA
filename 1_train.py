# train.py
# Script huấn luyện cuối cùng, đã được làm sạch, tối ưu hóa và tích hợp các bản sửa lỗi.

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
from peft import LoraConfig, get_peft_model

# Import các class đã được nâng cấp từ utils.py
# Đảm bảo bạn đã có file utils.py đã được cập nhật
from utils import DualEncoderModel, MSRVTT_Dataset, InfoNCELoss

# ==============================================================================
# PHẦN 1: CẤU HÌNH DỰ ÁN
# ==============================================================================
def get_config(env="4090"):
    """Hàm tập trung tất cả các cấu hình của dự án, tùy thuộc vào môi trường."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mixed_precision_mode = "bf16"
    else:
        mixed_precision_mode = "fp16"

    # Cấu hình chung
    base_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "projection_dim": 256,
        "num_frames": 16,
        "num_epochs": 3,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "checkpointing_steps": 1000,
        "validation_steps": 1000,
        "max_grad_norm": 1.0, # Thêm Gradient Clipping
    }

    # Cấu hình riêng cho từng môi trường
    if env.lower() == "4090":
        print(f">> Đang tải cấu hình cho RTX 4090...")
        env_config = {
            "project_root_path": "/root/CLIP_QLoRA",
            "batch_size": 16,
            "gradient_accumulation_steps": 4,  # Effective BS = 64
            "learning_rate": 2e-5,
            "new_module_lr": 1e-4,
            "lora_weight_decay": 0.01,
            "new_module_weight_decay": 0.05,
            "mixed_precision": mixed_precision_mode,
            "lora_rank": 32,
        }
    elif env.lower() == "colab":
        print(f">> Đang tải cấu hình cho Colab T4...")
        env_config = {
            "project_root_path": "/content/drive/MyDrive/CLIP_QLoRA_MSRVTT",
            "batch_size": 8,
            "gradient_accumulation_steps": 8,  # Effective BS = 64
            "learning_rate": 3e-5,
            "new_module_lr": 1.5e-4,
            "lora_weight_decay": 0.01,
            "new_module_weight_decay": 0.05,
            "mixed_precision": "fp16",
            "lora_rank": 16,
        }
    else:
        raise ValueError(f"Môi trường không xác định: {env}. Chỉ hỗ trợ '4090' hoặc 'colab'.")

    config = {**base_config, **env_config}
    config["output_dir"] = os.path.join(config["project_root_path"], f"output_{env}")
    return config

def validate_config(config):
    """Xác thực các tham số cấu hình."""
    print("Xác thực cấu hình...")
    required_keys = ['model_name', 'batch_size', 'learning_rate', 'output_dir', 'project_root_path']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Thiếu khóa cấu hình bắt buộc: {key}")
    print("Cấu hình hợp lệ.")
    return True

# ==============================================================================
# PHẦN 2: HÀM ĐÁNH GIÁ (CÓ METRICS)
# ==============================================================================
@torch.no_grad()
def calculate_retrieval_metrics(sim_matrix):
    """Tính toán các chỉ số retrieval từ ma trận tương đồng."""
    num_queries = sim_matrix.shape[0]
    ground_truth = torch.arange(num_queries, device=sim_matrix.device)
    
    sorted_indices = torch.argsort(sim_matrix, descending=True, dim=1)
    ranks_tensor = (sorted_indices == ground_truth.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    
    r1 = (ranks_tensor <= 1).float().mean().item() * 100
    r5 = (ranks_tensor <= 5).float().mean().item() * 100
    r10 = (ranks_tensor <= 10).float().mean().item() * 100
    return {"R@1": r1, "R@5": r5, "R@10": r10}

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, accelerator):
    model.eval()
    total_loss = 0.0
    all_video_embeds, all_text_embeds = [], []
    
    for batch in tqdm(dataloader, desc="Đang đánh giá", disable=not accelerator.is_main_process, leave=False):
        if batch is None: continue
        video_embeds, text_embeds = model(**batch)
        loss = loss_fn(video_embeds, text_embeds)
        
        all_video_embeds.append(accelerator.gather_for_metrics(video_embeds))
        all_text_embeds.append(accelerator.gather_for_metrics(text_embeds))
        total_loss += accelerator.gather_for_metrics(loss).mean().item()
    
    avg_loss = total_loss / len(dataloader)
    
    metrics = {}
    if accelerator.is_main_process and all_video_embeds and all_text_embeds:
        video_embeds_cat = torch.cat(all_video_embeds, dim=0)
        text_embeds_cat = torch.cat(all_text_embeds, dim=0)
        sim_matrix = torch.matmul(text_embeds_cat, video_embeds_cat.T) # Text-to-Video
        metrics = calculate_retrieval_metrics(sim_matrix)

    model.train()
    return avg_loss, metrics

# ==============================================================================
# PHẦN 3: HÀM HUẤN LUYỆN CHÍNH
# ==============================================================================
def main(args):
    config = get_config(env=args.env)
    validate_config(config)
    os.makedirs(config["output_dir"], exist_ok=True)
    
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=config["output_dir"]
    )
    accelerator.init_trackers("video_retrieval_training")
    accelerator.print(f"Cấu hình: {json.dumps(config, indent=2)}")

    # --- 1. Tải và Cấu hình Mô hình ---
    accelerator.print("Tải mô hình, lượng tử hóa backbone và áp dụng LoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, config["mixed_precision"]),
        bnb_4bit_use_double_quant=True
    )
    
    model = DualEncoderModel(
        model_name=config["model_name"], projection_dim=config["projection_dim"],
        quantization_config=bnb_config
    )
    
    lora_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2",
        ],
        lora_dropout=config["lora_dropout"], bias="none"
    )

    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)
    model.text_encoder = get_peft_model(model.text_encoder, lora_config)
    
    accelerator.print("\n--- Tham số LoRA Vision Encoder ---")
    model.vision_encoder.print_trainable_parameters()
    accelerator.print("\n--- Tham số LoRA Text Encoder ---")
    model.text_encoder.print_trainable_parameters()
    
    if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
        accelerator.print("Biên dịch mô hình với torch.compile để tăng tốc...")
        model = torch.compile(model, mode='reduce-overhead')

    # --- 2. Chuẩn bị Dữ liệu ---
    data_root = os.path.join(config["project_root_path"], "data", f"MSRVTT_Preprocessed_{config['num_frames']}frames")
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None
        keys = batch[0].keys()
        return {k: torch.stack([d[k] for d in batch]) for k in keys}
        
    num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 0)
    train_dataset = MSRVTT_Dataset(os.path.join(data_root, 'train_data.json'), data_root, processor, config["num_frames"])
    val_dataset = MSRVTT_Dataset(os.path.join(data_root, 'val_data.json'), data_root, processor, config["num_frames"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # --- 3. Chuẩn bị Optimizer, Scheduler, Loss ---
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    new_module_params = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
    
    optimizer = bnb.optim.PagedAdamW8bit([
        {'params': lora_params, 'lr': config["learning_rate"], 'weight_decay': config["lora_weight_decay"]},
        {'params': new_module_params, 'lr': config["new_module_lr"], 'weight_decay': config["new_module_weight_decay"]}
    ])
    loss_fn = InfoNCELoss()
    
    num_update_steps_per_epoch = -(-len(train_dataloader) // config["gradient_accumulation_steps"])
    total_training_steps = config["num_epochs"] * num_update_steps_per_epoch
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_training_steps), num_training_steps=total_training_steps)
    
    # --- 4. Chuẩn bị với Accelerator và Logic Tải Checkpoint ---
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
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
        model.train()
        for step, batch in enumerate(train_dataloader):
            if batch is None: continue
            if resume_from_checkpoint and epoch == start_epoch and step < (completed_steps % num_update_steps_per_epoch):
                continue

            with accelerator.accumulate(model):
                video_embeds, text_embeds = model(**batch)
                loss = loss_fn(video_embeds, text_embeds)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
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
                        state_dir = os.path.join(config["output_dir"], f"checkpoint_step_{current_step}")
                        accelerator.save_state(state_dir)
                        accelerator.print(f"\nĐã lưu trạng thái huấn luyện tại {state_dir}")
                
                if current_step > 0 and current_step % config["validation_steps"] == 0:
                    val_loss, val_metrics = evaluate(model, val_dataloader, loss_fn, accelerator)
                    log_data = {"val_loss": val_loss, **{f"val/{k}": v for k, v in val_metrics.items()}}
                    accelerator.log(log_data, step=current_step)
                    accelerator.print(f"\nBước {current_step} | Val Loss: {val_loss:.4f} | Val Metrics: {val_metrics}")

    # --- 6. Lưu Checkpoint Cuối cùng ---
    accelerator.print("--- HUẤN LUYỆN HOÀN TẤT! ---")
    if accelerator.is_main_process:
        final_dir = os.path.join(config["output_dir"], "final_model_full_state")
        accelerator.save_state(final_dir)
        accelerator.print(f"Đã lưu trạng thái huấn luyện cuối cùng tại: {final_dir}")
        # Lưu riêng adapter cuối cùng để dễ dàng đánh giá
        final_adapter_dir = os.path.join(config["output_dir"], "final_adapter")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(final_adapter_dir, "full_model_weights.pth")) # Ví dụ lưu toàn bộ state dict
        accelerator.print(f"Đã lưu adapter/mô hình cuối cùng tại: {final_adapter_dir}")

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