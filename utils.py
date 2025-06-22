import torch
import torch.nn as nn
import math
import os
import json
from PIL import Image
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor, BitsAndBytesConfig

# ==============================================================================
# LỚP POSITIONAL ENCODING (ĐÃ CẢI TIẾN)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cải tiến: Đảm bảo pe cùng device và dtype với input x.
        x = x + self.pe[:, :x.size(1), :].to(device=x.device, dtype=x.dtype)
        return self.dropout(x)

# ==============================================================================
# CLASS MÔ HÌNH DUAL-ENCODER (ĐÃ CẢI TIẾN)
# ==============================================================================
class DualEncoderModel(nn.Module):
    def __init__(self, model_name: str, projection_dim: int = 256, num_temporal_layers: int = 2, quantization_config: BitsAndBytesConfig = None, **kwargs):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_name, quantization_config=quantization_config)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, quantization_config=quantization_config)
        
        vision_hidden_size = self.vision_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # Các module mới được giữ ở độ chính xác cao
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=vision_hidden_size, nhead=8, dim_feedforward=vision_hidden_size * 4, dropout=0.1, activation='gelu', batch_first=True),
            num_layers=num_temporal_layers
        )
        self.video_pos_encoder = PositionalEncoding(d_model=vision_hidden_size)
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, vision_hidden_size))
        self.video_projection = nn.Linear(vision_hidden_size, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden_size, projection_dim, bias=False)

    def get_trainable_parameters(self):
        return list(self.temporal_encoder.parameters()) + \
               [self.video_cls_token] + \
               list(self.video_projection.parameters()) + \
               list(self.text_projection.parameters())

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        # Cải tiến: Xử lý dtype một cách nhất quán hơn
        target_dtype = next(self.temporal_encoder.parameters()).dtype

        # Vision
        if pixel_values is not None:
            batch_size, num_frames, C, H, W = pixel_values.shape
            vision_outputs = self.vision_encoder(pixel_values=pixel_values.view(-1, C, H, W))
            frame_embeds = vision_outputs.last_hidden_state[:, 0, :].view(batch_size, num_frames, -1)
            
            # Chuyển sang dtype của các lớp có thể học
            frame_embeds = frame_embeds.to(target_dtype)
            
            cls_tokens = self.video_cls_token.expand(batch_size, -1, -1)
            temporal_input = torch.cat((cls_tokens, frame_embeds), dim=1)
            temporal_input_with_pos = self.video_pos_encoder(temporal_input)
            video_embeds = self.temporal_encoder(temporal_input_with_pos)[:, 0, :]
            video_embeds = self.video_projection(video_embeds)
            video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
        else:
            video_embeds = None

        # Text
        if input_ids is not None:
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state[:, 0, :]
            text_embeds = self.text_projection(text_embeds.to(target_dtype))
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        else:
            text_embeds = None
            
        return video_embeds, text_embeds

# ==============================================================================
# CLASS DATASET (ĐÃ CẢI TIẾN)
# ==============================================================================
class MSRVTT_Dataset(Dataset):
    def __init__(self, json_file_path: str, preprocessed_data_folder: str, processor: CLIPProcessor, num_frames: int = 16):
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        self.preprocessed_data_folder = preprocessed_data_folder
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Cải tiến: Xử lý lỗi mạnh mẽ hơn
        max_retries = 5
        for i in range(max_retries):
            current_idx = (idx + i) % len(self)
            sample = self.data[current_idx]
            try:
                frame_paths = [os.path.join(self.preprocessed_data_folder, p) for p in sample['frame_paths']]
                video_frames = [Image.open(p).convert("RGB") for p in frame_paths]
                pixel_values = self.processor(images=video_frames, return_tensors="pt").pixel_values
                caption = sample.get('caption', '')
                text_inputs = self.processor(
                    text=caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
                )
                return {
                    "pixel_values": pixel_values,
                    "input_ids": text_inputs.input_ids.squeeze(0),
                    "attention_mask": text_inputs.attention_mask.squeeze(0),
                }
            except Exception as e:
                if i == max_retries - 1:
                    print(f"Lỗi nghiêm trọng: Không thể tải mẫu nào sau {max_retries} lần thử, bắt đầu từ index {idx}. Lỗi cuối cùng: {e}")
                    return None # Trả về None để collate_fn có thể lọc ra
        return None

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