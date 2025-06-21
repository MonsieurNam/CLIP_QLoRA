# utils.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import os
import json

# ==================================
# CLASS MÔ HÌNH
# ==================================
class DualEncoderModel(nn.Module):
    def __init__(self, model_name, projection_dim=256):
        super().__init__()
        # Tải mô hình gốc chưa lượng tử hóa
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(model_name)

        vision_hidden_size = self.vision_encoder.config.projection_dim
        text_hidden_size = self.text_encoder.config.projection_dim

        self.video_projection = nn.Linear(vision_hidden_size, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden_size, projection_dim, bias=False)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        video_embeds, text_embeds = None, None
        if pixel_values is not None:
            batch_size, num_frames, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(-1, C, H, W)
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            frame_embeds = vision_outputs.image_embeds
            video_embeds = frame_embeds.view(batch_size, num_frames, -1).mean(dim=1)
            video_embeds = self.video_projection(video_embeds)
            video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
        if input_ids is not None:
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.text_embeds
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return video_embeds, text_embeds

# ==================================
# CLASS DATASET
# ==================================
class MSRVTT_Dataset(Dataset):
    def __init__(self, json_file_path, preprocessed_data_folder, processor, num_frames=16):
        print(f"Tải dữ liệu từ: {json_file_path}")
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        self.preprocessed_data_folder = preprocessed_data_folder
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        frame_paths = [os.path.join(self.preprocessed_data_folder, p) for p in sample['frame_paths']]
        video_frames = [Image.open(p).convert("RGB") for p in frame_paths]
        
        pixel_values = self.processor(images=video_frames, return_tensors="pt").pixel_values
        
        caption = sample['caption']
        text_inputs = self.processor(
            text=caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0)
        }

# ==================================
# CLASS LOSS
# ==================================
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, video_embeds, text_embeds):
        logits = torch.matmul(video_embeds, text_embeds.T) / self.temperature
        labels = torch.arange(video_embeds.shape[0], device=video_embeds.device)
        loss_v2t = nn.functional.cross_entropy(logits, labels)
        loss_t2v = nn.functional.cross_entropy(logits.T, labels)
        return (loss_v2t + loss_t2v) / 2.0