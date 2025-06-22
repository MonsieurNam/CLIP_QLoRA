# utils.py
# Chứa các class và hàm dùng chung cho dự án Text-Video Retrieval.
# Phiên bản cuối cùng:
# - DualEncoderModel nhận quantization_config trực tiếp.
# - Thêm PositionalEncoding cho Temporal Transformer.
# - Cải thiện docstrings và tính rõ ràng.

import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from transformers import (
    CLIPVisionModel,
    CLIPTextModel,
    CLIPProcessor,
    BitsAndBytesConfig # Import để type hinting
)
from PIL import Image
import os
import json

# ==============================================================================
# LỚP POSITIONAL ENCODING
# ==============================================================================
class PositionalEncoding(nn.Module):
    """
    Thêm thông tin vị trí vào các embedding đầu vào.
    Đây là kỹ thuật chuẩn trong các mô hình Transformer gốc.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # Sửa: Đảm bảo pe có cùng device và dtype với model sau này
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x.size(1) là độ dài chuỗi (seq_len)
        # Chuyển pe sang device của x
        x = x + self.pe[:x.size(1)].transpose(0, 1).to(x.device, dtype=x.dtype)
        return self.dropout(x)

# ==============================================================================
# CLASS MÔ HÌNH DUAL-ENCODER (ĐÃ NÂNG CẤP)
# ==============================================================================

class DualEncoderModel(nn.Module):
    def __init__(self, model_name: str, projection_dim: int = 256, num_temporal_layers: int = 2, quantization_config: BitsAndBytesConfig = None):
        super().__init__()
        
        # --- PHẦN 1: Tải các backbone đã được lượng tử hóa (sẽ bị đóng băng) ---
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            model_name, quantization_config=quantization_config
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, quantization_config=quantization_config
        )

        # --- PHẦN 2: Định nghĩa các module MỚI và có thể HỌC (phải là float32) ---
        vision_hidden_size = self.vision_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # Các module này sẽ không bị lượng tử hóa và sẽ được optimizer cập nhật
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=vision_hidden_size, nhead=8, dim_feedforward=vision_hidden_size * 4,
                dropout=0.1, activation='gelu', batch_first=True
            ),
            num_layers=num_temporal_layers
        )
        self.video_pos_encoder = PositionalEncoding(d_model=vision_hidden_size)
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, vision_hidden_size))
        self.video_projection = nn.Linear(vision_hidden_size, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden_size, projection_dim, bias=False)

    def get_trainable_parameters(self):
        """Hàm trợ giúp để lấy các tham số của các module mới."""
        new_modules_params = [
            *self.temporal_encoder.parameters(),
            self.video_cls_token,
            *self.video_projection.parameters(),
            *self.text_projection.parameters()
        ]
        return new_modules_params

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        # ... forward pass ...
        # Đảm bảo các phép tính với module mới được thực hiện ở float32
        
        # Vision
        vision_outputs = self.vision_encoder(pixel_values=pixel_values.view(-1, *pixel_values.shape[2:]))
        frame_embeds = vision_outputs.last_hidden_state[:, 0, :]
        frame_embeds = frame_embeds.view(pixel_values.shape[0], pixel_values.shape[1], -1)
        
        # Chuyển sang float32 để xử lý với các module có thể học
        frame_embeds_f32 = frame_embeds.to(torch.float32)
        
        cls_tokens = self.video_cls_token.expand(frame_embeds.shape[0], -1, -1)
        temporal_input = torch.cat((cls_tokens, frame_embeds_f32), dim=1)
        temporal_input_with_pos = self.video_pos_encoder(temporal_input)
        temporal_output = self.temporal_encoder(temporal_input_with_pos)
        video_embeds = temporal_output[:, 0, :]
        video_embeds = self.video_projection(video_embeds)
        video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
        
        # Text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        
        # Chuyển sang float32
        text_embeds_f32 = text_embeds.to(torch.float32)

        text_embeds = self.text_projection(text_embeds_f32)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return video_embeds, text_embeds

# ==============================================================================
# CLASS DATASET
# ==============================================================================
class MSRVTT_Dataset(Dataset):
    def __init__(self, json_file_path: str, preprocessed_data_folder: str, processor: CLIPProcessor, num_frames: int = 16):
        print(f"Tải dữ liệu từ: {json_file_path}")
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File JSON không tồn tại: {json_file_path}")
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        self.preprocessed_data_folder = preprocessed_data_folder
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            frame_paths = [os.path.join(self.preprocessed_data_folder, p) for p in sample['frame_paths']]
            video_frames = [Image.open(p).convert("RGB") for p in frame_paths]
            
            # Processor sẽ xử lý việc resize, normalize
            pixel_values = self.processor(images=video_frames, return_tensors="pt").pixel_values
            
            caption = sample.get('caption', '') # Dùng .get để an toàn hơn
            text_inputs = self.processor(
                text=caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
            )
            
            return {
                "pixel_values": pixel_values,
                "input_ids": text_inputs.input_ids.squeeze(0),
                "attention_mask": text_inputs.attention_mask.squeeze(0),
                "video_id": sample.get('video_id', '') 
            }
        except Exception as e:
            # print(f"Lỗi khi tải mẫu {idx}, video_id: {sample.get('video_id', 'N/A')}. Lỗi: {e}. Bỏ qua mẫu này.")
            return self.__getitem__((idx + 1) % len(self)) # Trả về mẫu hợp lệ tiếp theo

# ==============================================================================
# CLASS LOSS FUNCTION
# ==============================================================================
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, video_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        # Đảm bảo kiểu dữ liệu phù hợp trước khi tính toán
        video_embeds_f32 = video_embeds.to(torch.float32)
        text_embeds_f32 = text_embeds.to(torch.float32)
        
        logits = torch.matmul(video_embeds_f32, text_embeds_f32.T) / self.temperature
        
        labels = torch.arange(video_embeds.shape[0], device=video_embeds.device)
        
        loss_v2t = nn.functional.cross_entropy(logits, labels)
        loss_t2v = nn.functional.cross_entropy(logits.T, labels)
        
        return (loss_v2t + loss_t2v) / 2.0