# 3_convert_checkpoint.py
# Script để chuyển đổi checkpoint từ định dạng của Accelerator
# sang định dạng chuẩn của PEFT mà không cần huấn luyện lại.

import os
import torch
from accelerate import Accelerator
from utils import *

def main():
    # --- 1. CẤU HÌNH ---
    # Các cấu hình này phải khớp với cấu hình bạn đã dùng để huấn luyện
    # để có thể tải lại mô hình một cách chính xác.
    
    project_root_path = "/root/CLIP_QLoRA" # THAY ĐỔI NẾU CẦN
    model_name = "openai/clip-vit-base-patch32"
    projection_dim = 256
    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.05
    
    # Đường dẫn đến checkpoint CŨ (được tạo bởi save_state)
    input_checkpoint_dir = os.path.join(project_root_path, "checkpoints_clip-vit-base-patch32_qlora/final_checkpoint")  # THAY ĐỔI NẾU CẦN

    # Đường dẫn đến thư mục MỚI để lưu adapter đúng chuẩn
    output_adapter_dir = os.path.join(project_root_path, "adapter_final_converted")
    
    print(f"Bắt đầu quá trình chuyển đổi checkpoint từ: {input_checkpoint_dir}")

    if not os.path.exists(input_checkpoint_dir):
        print(f"LỖI: Không tìm thấy thư mục checkpoint tại '{input_checkpoint_dir}'")
        return

    # --- 2. KHỞI TẠO LẠI MÔ HÌNH VÀ CÁC THÀNH PHẦN ---
    # Cần phải khởi tạo lại toàn bộ kiến trúc và các thành phần y hệt như lúc huấn luyện
    # để Accelerator có thể tải lại trạng thái một cách chính xác.

    # Khởi tạo mô hình nền
    base_model = DualEncoderModel(model_name=model_name, projection_dim=projection_dim)

    # Cấu hình LoRA
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "text_projection", "video_projection"],
        lora_dropout=lora_dropout, bias="none", task_type="FEATURE_EXTRACTION"
    )

    # Tạo mô hình PEFT (chưa có trọng số)
    lora_model = get_peft_model(base_model, lora_config)

    # Khởi tạo Accelerator (không cần các tham số phức tạp)
    accelerator = Accelerator()
    
    # Chuẩn bị mô hình với Accelerator
    # Bước này quan trọng để accelerator biết về mô hình
    lora_model = accelerator.prepare(lora_model)

    # --- 3. TẢI TRẠNG THÁI VÀ LƯU LẠI ADAPTER ---
    try:
        print("Đang tải toàn bộ trạng thái huấn luyện...")
        accelerator.load_state(input_checkpoint_dir)
        print("Tải trạng thái thành công!")

        # Lấy lại mô hình đã được nạp trọng số từ accelerator
        unwrapped_model = accelerator.unwrap_model(lora_model)

        print(f"Đang lưu adapter theo định dạng PEFT tại: {output_adapter_dir}")
        # Sử dụng .save_pretrained() để lưu đúng chuẩn
        unwrapped_model.save_pretrained(output_adapter_dir)
        
        print("\n--- CHUYỂN ĐỔI HOÀN TẤT! ---")
        print(f"Adapter đã sẵn sàng để sử dụng tại: {output_adapter_dir}")
        print("Bây giờ bạn có thể dùng đường dẫn này trong script đánh giá của mình.")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình chuyển đổi: {e}")


if __name__ == "__main__":
    main()