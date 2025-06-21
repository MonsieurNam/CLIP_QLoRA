# model.py
import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

# 3_convert_checkpoint.py
import os
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

class DualEncoderModel(nn.Module)
    def __init__(self, model_name, projection_dim=256, bnb_config=None)
        super().__init__()
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_name, quantization_config=bnb_config)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_name, quantization_config=bnb_config)
        
        vision_hidden_size = self.vision_encoder.config.projection_dim
        text_hidden_size = self.text_encoder.config.projection_dim

        self.video_projection = nn.Linear(vision_hidden_size, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden_size, projection_dim, bias=False)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, kwargs)
        video_embeds, text_embeds = None, None
        if pixel_values is not None
            B, T, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(-1, C, H, W)
            frame_embeds = self.vision_encoder(pixel_values=pixel_values).image_embeds
            video_embeds = frame_embeds.view(B, T, -1).mean(dim=1)
            video_embeds = self.video_projection(video_embeds)
            video_embeds = video_embeds  video_embeds.norm(dim=-1, keepdim=True)

        if input_ids is not None
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.text_embeds
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds  text_embeds.norm(dim=-1, keepdim=True)

        return video_embeds, text_embeds



def main()
    # === CONFIG ===
    project_root_path = rootCLIP_QLoRA
    model_name = openaiclip-vit-base-patch32
    projection_dim = 256
    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.05

    checkpoint_path = rootCLIP_QLoRAcontentdriveMyDriveIntern_FPTAIDATASETtraining_checkpointscheckpoint_step_26500
    output_path = os.path.join(project_root_path, adapter_26500_converted)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=nf4,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # === INIT MODEL ===
    base_model = DualEncoderModel(model_name=model_name, projection_dim=projection_dim, bnb_config=bnb_config)
    model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=none,
        task_type=FEATURE_EXTRACTION,
        target_modules=[q_proj, v_proj, k_proj, o_proj, text_projection, video_projection]
    )
    lora_model = get_peft_model(model, lora_config)

    # === ACCELERATOR ===
    accelerator = Accelerator()
    lora_model = accelerator.prepare(lora_model)

    try
        print(fLoading checkpoint from {checkpoint_path}...)
        accelerator.load_state(checkpoint_path)
        print(✓ Checkpoint loaded!)

        unwrapped = accelerator.unwrap_model(lora_model)
        unwrapped.save_pretrained(output_path, safe_serialization=True, save_adapter=True)
        print(f✓ Adapter saved to {output_path})
    except Exception as e
        print(f[ERROR] Lỗi khi chuyển checkpoint {e})

if __name__ == __main__
    main()
