## Project Directory Structure

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

```
accelerate config

In which compute environment are you running?: This machine (nhấn Enter)
Which type of machine are you using?: No distributed training (chọn 0)
Do you want to run your training on CPU?: no
Do you want to use DeepSpeed?: no
Do you want to use FullyShardedDataParallel?: no
Do you want to use Megatron-LM?: no
How many GPUs should be used for distributed training?: 1
Do you wish to use bfloat16?: yes (Đây là câu hỏi quan trọng nhất cho RTX 4090)
```
## cài môi trường ảo
```
python3 -m venv venv
source venv/bin/activate
```
## Bước a: Cài đặt PyTorch và các thư viện liên quan
```
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```
## Bước b: Cài đặt bitsandbytes từ nguồn đã biên dịch sẵn 
```
pip install bitsandbytes==0.43.1
```
## Bước c: Cài đặt các thư viện còn lại, bao gồm cả safetensors
```
pip install transformers==4.41.2 peft==0.11.1 accelerate==0.31.0 datasets==2.20.0 decord tensorboard safetensors
```
