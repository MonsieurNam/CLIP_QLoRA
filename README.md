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

```
accelerate config
```

In which compute environment are you running?: This machine (nhấn Enter)
Which type of machine are you using?: No distributed training (chọn 0)
Do you want to run your training on CPU?: no
Do you want to use DeepSpeed?: no
Do you want to use FullyShardedDataParallel?: no
Do you want to use Megatron-LM?: no
How many GPUs should be used for distributed training?: 1
Do you wish to use bfloat16?: yes (Đây là câu hỏi quan trọng nhất cho RTX 4090)
