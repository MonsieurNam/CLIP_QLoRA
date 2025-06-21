/YourProjectRoot/
|
├── data/
│   ├── MSRVTT_Preprocessed_16frames/
│   │   ├── video0/
│   │   │   ├── frame_0000.jpg
│   │   │   └── ...
│   │   ├── ...
│   │   ├── train_data.json
│   │   ├── val_data.json
│   │   └── test_data.json
│   └── msrvtt_data/  (Thư mục chứa video gốc và file json gốc)
│       ├── train-val-video/
│       └── train_val_videodatainfo.json
|
├── checkpoints/
│   ├── retrieval_adapter_epoch_1/
│   └── ...
|
├── 0_preprocess_data.py   # Script để tiền xử lý dữ liệu
├── 1_train.py             # Script để huấn luyện mô hình
├── 2_evaluate.py          # Script để đánh giá mô hình
└── utils.py               # Script chứa các class và hàm dùng chung
