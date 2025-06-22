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

# Import upgraded classes from utils.py
from utils import DualEncoderModel, MSRVTT_Dataset, InfoNCELoss

# ==============================================================================
# PART 1: PROJECT CONFIGURATION
# ==============================================================================
def get_config(env="4090"):
    """Gather all project configurations."""
    # Determine mixed precision mode
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mixed_precision_mode = "bf16"
    else:
        mixed_precision_mode = "fp16"

    base_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "projection_dim": 256,
        "num_frames": 16,
        "num_epochs": 1,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "checkpointing_steps": 100,
        "validation_steps": 100,
    }

    if env.lower() == "4090":
        print(f">> Đang tải cấu hình cho RTX 4090...")
        env_config = {
            "project_root_path": "/root/CLIP_QLoRA",
            # Giảm batch_size nhẹ để tránh OOM
            "batch_size": 16,
            "gradient_accumulation_steps": 4,  # effective BS = 64
            "learning_rate": 2e-5,
            "new_module_lr": 1e-4,
            "mixed_precision": mixed_precision_mode,  # "bf16" nếu có, else "fp16"
            "lora_rank": 32,
        }
    elif env.lower() == "colab":
        print(f">> Đang tải cấu hình cho Colab T4...")
        env_config = {
            "project_root_path": "/content/drive/MyDrive/CLIP_QLoRA_MSRVTT",
            "batch_size": 8,
            "gradient_accumulation_steps": 8,  # effective BS = 64
            "learning_rate": 3e-5,
            "new_module_lr": 1.5e-4,
            "mixed_precision": "fp16",
            "lora_rank": 16,
        }
    else:
        raise ValueError(f"Môi trường không xác định: {env}. Chỉ hỗ trợ '4090' hoặc 'colab'.")

    config = {**base_config, **env_config}
    config["output_dir"] = os.path.join(config["project_root_path"], f"output_{env}")
    return config

# ==============================================================================
# PART 2: EVALUATION FUNCTION
# ==============================================================================
@torch.no_grad()
def evaluate(model, dataloader, loss_fn, accelerator):
    """Evaluate model on validation set, return average loss."""
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
# PART 3: MAIN TRAINING FUNCTION
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

    # --- 1. Load and prepare model + quantization + LoRA ---
    accelerator.print("Tải mô hình, lượng tử hóa backbone và áp dụng LoRA...")

    # Map precision string to torch.dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        # nếu cần: "fp32": torch.float32
    }
    mp = config["mixed_precision"]
    if mp not in dtype_map:
        raise ValueError(f"Unsupported mixed_precision '{mp}' for BitsAndBytesConfig")
    bnb_compute_dtype = dtype_map[mp]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    # Load base DualEncoderModel (handles quantization inside)
    model = DualEncoderModel(
        model_name=config["model_name"],
        projection_dim=config["projection_dim"],
        quantization_config=bnb_config
    )

    # Apply LoRA to vision_encoder and text_encoder separately
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=config["lora_dropout"],
        bias="none"
    )
    # Wrap encoders
    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)
    model.text_encoder   = get_peft_model(model.text_encoder, lora_config)

    accelerator.print("\n--- Tham số LoRA Vision Encoder ---")
    model.vision_encoder.print_trainable_parameters()
    accelerator.print("\n--- Tham số LoRA Text Encoder ---")
    model.text_encoder.print_trainable_parameters()

    # --- 2. Prepare optimizer with different LR groups ---
    # Collect LoRA params
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    # Collect other new-module params via helper in utils.py
    new_module_params = model.get_trainable_parameters()  # đảm bảo helper chỉ trả các params cần train ngoài LoRA
    optimizer = bnb.optim.PagedAdamW8bit(
        [
            {"params": lora_params, "lr": config["learning_rate"]},
            {"params": new_module_params, "lr": config["new_module_lr"]},
        ],
        # default lr (will be overridden by groups)
        lr=config["learning_rate"]
    )

    # --- 3. Prepare datasets and dataloaders ---
    data_root = os.path.join(config["project_root_path"], "data", f"MSRVTT_Preprocessed_{config['num_frames']}frames")
    # Debug check paths
    accelerator.print(f"DEBUG: data_root = {data_root}")
    if not os.path.isdir(data_root):
        raise RuntimeError(f"data_root không tồn tại: {data_root}. Kiểm tra bước preprocessing.")
    train_json = os.path.join(data_root, "train_data.json")
    val_json   = os.path.join(data_root, "val_data.json")
    accelerator.print(f"DEBUG: train_data.json exists? {os.path.exists(train_json)}; val_data.json exists? {os.path.exists(val_json)}")
    if not os.path.isfile(train_json) or not os.path.isfile(val_json):
        raise RuntimeError(f"train_data.json hoặc val_data.json không tồn tại. Kiểm tra preprocessing.")

    processor = CLIPProcessor.from_pretrained(config["model_name"])
    train_dataset = MSRVTT_Dataset(
        json_file_path=train_json,
        preprocessed_data_folder=data_root,
        processor=processor,
        num_frames=config["num_frames"]
    )
    val_dataset = MSRVTT_Dataset(
        json_file_path=val_json,
        preprocessed_data_folder=data_root,
        processor=processor,
        num_frames=config["num_frames"]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=2
    )

    # --- 4. Prepare loss, scheduler, total steps ---
    loss_fn = InfoNCELoss()
    num_update_steps_per_epoch = -(-len(train_dataloader) // config["gradient_accumulation_steps"])
    total_training_steps = config["num_epochs"] * num_update_steps_per_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_training_steps),
        num_training_steps=total_training_steps
    )

    # --- 5. Accelerator.prepare and resume checkpoint if any ---
    # Note: pass `model` (already wrapped LoRA) to prepare
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
        try:
            completed_steps = int(resume_from_checkpoint.split("_")[-1])
        except:
            completed_steps = 0

     # --- 5. Vòng lặp Huấn luyện ---
    progress_bar = tqdm(range(completed_steps, total_training_steps), disable=not accelerator.is_main_process)
    start_epoch = completed_steps // num_update_steps_per_epoch
    
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train() # Sửa: model là tên biến được trả về từ accelerator.prepare
        for step, batch in enumerate(train_dataloader):
            if resume_from_checkpoint and epoch == start_epoch and step < (completed_steps % num_update_steps_per_epoch):
                continue

            with accelerator.accumulate(model):
                video_embeds, text_embeds = model(**batch)
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

                # === SỬA LỖI LOGIC LƯU CHECKPOINT ===
                if current_step > 0 and current_step % config["checkpointing_steps"] == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint_step_{current_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        accelerator.print(f"\nĐang lưu checkpoint hoàn chỉnh tại: {checkpoint_dir}")
                        
                        # Lưu trạng thái của Accelerator (optimizer, scheduler, etc.)
                        accelerator.save_state(os.path.join(checkpoint_dir, "accelerator_state"))
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        # Lưu adapter của vision encoder
                        vision_adapter_path = os.path.join(checkpoint_dir, "vision_adapter")
                        unwrapped_model.vision_encoder.save_pretrained(vision_adapter_path)
                        
                        # Lưu adapter của text encoder
                        text_adapter_path = os.path.join(checkpoint_dir, "text_adapter")
                        unwrapped_model.text_encoder.save_pretrained(text_adapter_path)
                        
                        # Lưu state_dict của các module mới
                        new_modules_state = {
                            'temporal_encoder': unwrapped_model.temporal_encoder.state_dict(),
                            'video_projection': unwrapped_model.video_projection.state_dict(),
                            'text_projection': unwrapped_model.text_projection.state_dict(),
                            'video_cls_token': unwrapped_model.video_cls_token,
                        }
                        torch.save(new_modules_state, os.path.join(checkpoint_dir, "new_modules.pth"))
                
                if current_step > 0 and current_step % config["validation_steps"] == 0:
                    val_loss = evaluate(model, val_dataloader, loss_fn, accelerator)
                    accelerator.log({"val_loss": val_loss}, step=current_step)
                    accelerator.print(f"\nBước {current_step} | Validation Loss: {val_loss:.4f}")

    # --- 6. Lưu Checkpoint Cuối cùng ---
    accelerator.print("--- HUẤN LUYỆN HOÀN TẤT! ---")
    if accelerator.is_main_process:
        final_dir = os.path.join(config["output_dir"], "final_model")
        os.makedirs(final_dir, exist_ok=True)
        
        accelerator.print(f"Đang lưu mô hình cuối cùng tại: {final_dir}")
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Lưu các thành phần tương tự như khi checkpoint
        unwrapped_model.vision_encoder.save_pretrained(os.path.join(final_dir, "vision_adapter"))
        unwrapped_model.text_encoder.save_pretrained(os.path.join(final_dir, "text_adapter"))
        
        new_modules_state = {
            'temporal_encoder': unwrapped_model.temporal_encoder.state_dict(),
            'video_projection': unwrapped_model.video_projection.state_dict(),
            'text_projection': unwrapped_model.text_projection.state_dict(),
            'video_cls_token': unwrapped_model.video_cls_token,
        }
        torch.save(new_modules_state, os.path.join(final_dir, "new_modules.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script for Text-Video Retrieval.")
    parser.add_argument(
        "--env", type=str, default="colab", choices=["4090", "colab"],
        help="Environment for training: '4090' or 'colab'."
    )
    args = parser.parse_args()

    # To run in notebook:
    # notebook_launcher(main, args=(args,), num_processes=1)
    main(args)
