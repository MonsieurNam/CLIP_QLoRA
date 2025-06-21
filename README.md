# CLIP-QLoRA for Text-Video Retrieval (MSR-VTT)

## 📁 Project Directory Structure

```
/YourProjectRoot/
├── data/
│   ├── MSRVTT_Preprocessed_16frames/
│   │   ├── video0/
│   │   │   ├── frame_0000.jpg
│   │   │   └── ...
│   │   ├── ...
│   │   ├── train_data.json
│   │   ├── val_data.json
│   │   └── test_data.json
│   └── msrvtt_data/
│       ├── train-val-video/
│       └── train_val_videodatainfo.json
│
├── checkpoints/
│   ├── retrieval_adapter_epoch_1/
│   └── ...
│
├── 0_preprocess_data.py   # Script để tiền xử lý dữ liệu
├── 1_train.py             # Script để huấn luyện mô hình
├── 2_evaluate.py          # Script để đánh giá mô hình
└── utils.py               # Chứa các hàm và class dùng chung
```
---

## 🚀 Environment Setup

### ✅ Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### ✅ Step 2a: Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

---

### ✅ Step 2b: Install `bitsandbytes` (precompiled)

```bash
pip install bitsandbytes==0.43.1
```

---

### ✅ Step 2c: Install Remaining Dependencies

```bash
pip install transformers==4.41.2 peft==0.11.1 accelerate==0.31.0 datasets==2.20.0 decord tensorboard safetensors
```

---

## 🏁 Training

Launch training with:

```bash
accelerate launch 1_train.py
```

## Cấu hình tăng tốc huấn luyện với `accelerate`
Trước khi chạy huấn luyện, bạn cần cấu hình `accelerate` như sau:
```
accelerate config
```
Ví dụ lựa chọn:
```
In which compute environment are you running?
This machine                                                                                                   
Which type of machine are you using?                                                                           
No distributed training                                                                                        
Do you want to run your training on CPU only? [yes/NO]: no                                                                                                    
Do you wish to optimize your script with torch dynamo?[yes/NO]: no                                              
Do you want to use DeepSpeed? [yes/NO]: no                                                                     
What GPU(s) (by id) should be used? [all]: enter               
Would you like to enable numa efficiency? [yes/NO]: yes         
Do you wish to use FP16 or BF16 (mixed precision)?
bf16
```

## Chạy huấn luyện với `tmux` để không bị mất kết nối
```
# Bước 1: Khởi tạo session tmux mới tên là train_clip
tmux new -s train_clip

# Bước 2: Kích hoạt môi trường ảo (nếu chưa)
source venv/bin/activate

# Bước 3: Chạy script huấn luyện
accelerate launch 1_train.py

# Bước 4: Để thoát khỏi tmux mà không dừng chương trình:
Ctrl + B, rồi nhấn D (detach)

# Bước 5: Để quay lại tmux session:
tmux attach -t train_clip

# Bước 6: Để liệt kê các tmux session đang chạy:
tmux ls

# Bước 7: Để xoá session khi xong việc:
tmux kill-session -t train_clip
```
