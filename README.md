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

## ⚙️ `accelerate config` Recommendation

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> No distributed training

Do you want to run your training on CPU only?
> NO

Do you wish to optimize your script with torch dynamo?
> NO

Do you want to use DeepSpeed?
> NO

What GPU(s) (by id) should be used for training on this machine?
> (Just press Enter for 'all')

Would you like to enable numa efficiency?
> YES

Do you wish to use FP16 or BF16 (mixed precision)?
> bf16
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