# 0_preprocess_data.py

import os
import json
import torch
import decord
from PIL import Image
from tqdm import tqdm

# Cấu hình để decord trả về torch tensor, nhưng chúng ta sẽ chuyển sang numpy để lưu ảnh
decord.bridge.set_bridge('torch')

# ==============================================================================
# PHẦN 1: CẤU HÌNH CÁC ĐƯỜNG DẪN VÀ THAM SỐ
# ==============================================================================

# !!! THAY ĐỔI ĐƯỜNG DẪN NÀY cho phù hợp với máy của bạn !!!
# Đây là thư mục gốc chứa toàn bộ dự án
PROJECT_ROOT_PATH = "/content/drive/MyDrive/Intern_FPT/AI/DATASET/"

# Thư mục chứa dữ liệu gốc của MSR-VTT
MSRVTT_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "MSRVTT")
MSRVTT_VIDEO_FOLDER = os.path.join(MSRVTT_DATA_PATH, "train-val-video")
MSRVTT_TEST_VIDEO_FOLDER = os.path.join(MSRVTT_DATA_PATH, "test-video") # Thư mục video test
MSRVTT_JSON_PATH = os.path.join(MSRVTT_DATA_PATH, "train_val_videodatainfo.json")
MSRVTT_TEST_JSON_PATH = os.path.join(MSRVTT_DATA_PATH, "test_videodatainfo.json") # File mô tả tập test

# Thư mục để lưu dữ liệu đã được tiền xử lý
PREPROCESSED_DATA_FOLDER = os.path.join(PROJECT_ROOT_PATH, "MSRVTT_Preprocessed_16frames")

# Tham số tiền xử lý
NUM_FRAMES = 16  # Số khung hình cần trích xuất cho mỗi video

# ==============================================================================
# PHẦN 2: HÀM TIỀN XỬ LÝ
# ==============================================================================

def process_split(videos_metadata, video_captions, video_folder, split_name):
    """
    Hàm chung để xử lý một tập dữ liệu (train, val, hoặc test).
    
    Args:
        videos_metadata (list): Danh sách các đối tượng video từ file JSON.
        video_captions (dict): Từ điển map video_id với danh sách các caption.
        video_folder (str): Đường dẫn đến thư mục chứa các file video .mp4.
        split_name (str): Tên của tập dữ liệu ('train', 'validate', 'test').

    Returns:
        list: Danh sách các mẫu đã được xử lý.
    """
    processed_samples = []
    
    # Lọc ra các video thuộc split hiện tại
    split_videos = [v for v in videos_metadata if v.get('split') == split_name]
    if not split_videos and split_name == 'test': # Xử lý riêng cho tập test nếu không có trường 'split'
        split_videos = videos_metadata
        
    print(f"\nBắt đầu xử lý {len(split_videos)} video cho tập '{split_name}'...")

    for video_meta in tqdm(split_videos, desc=f"Đang xử lý tập {split_name}"):
        video_id = video_meta['video_id']
        video_path = os.path.join(video_folder, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            # print(f"Cảnh báo: Bỏ qua video không tồn tại {video_path}")
            continue

        try:
            # 1. Đọc video và lấy mẫu khung hình
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                # print(f"Cảnh báo: Video rỗng {video_id}")
                continue
            
            indices = torch.linspace(0, total_frames - 1, NUM_FRAMES, dtype=torch.long)
            frames = vr.get_batch(indices)

            # 2. Lưu các khung hình đã trích xuất
            video_frame_folder = os.path.join(PREPROCESSED_DATA_FOLDER, video_id)
            os.makedirs(video_frame_folder, exist_ok=True)
            
            frame_paths = []
            for i, frame_tensor in enumerate(frames):
                frame_image = Image.fromarray(frame_tensor.numpy())
                # Lưu đường dẫn tương đối để dễ di chuyển
                relative_path = os.path.join(video_id, f"frame_{i:04d}.jpg")
                full_path = os.path.join(PREPROCESSED_DATA_FOLDER, relative_path)
                frame_image.save(full_path)
                frame_paths.append(relative_path)

            # 3. Kết hợp khung hình với chú thích
            if video_id in video_captions:
                for caption in video_captions[video_id]:
                    sample = {
                        "video_id": video_id,
                        "frame_paths": frame_paths,
                        "caption": caption
                    }
                    processed_samples.append(sample)
            else:
                 # Đối với tập test, có thể không có caption trong file train/val
                 # Ta vẫn tạo một mẫu không có caption để có thể tính embedding video
                 sample = {
                        "video_id": video_id,
                        "frame_paths": frame_paths,
                        "caption": "" # Caption rỗng
                    }
                 processed_samples.append(sample)


        except Exception as e:
            print(f"Lỗi khi xử lý video {video_id}: {e}")
            
    return processed_samples

def main():
    """Hàm chính để chạy toàn bộ quá trình tiền xử lý."""
    os.makedirs(PREPROCESSED_DATA_FOLDER, exist_ok=True)
    
    # --- Xử lý tập Train và Validation ---
    print("--- Bắt đầu xử lý dữ liệu Train/Validation ---")
    if not os.path.exists(MSRVTT_JSON_PATH):
        print(f"Lỗi: Không tìm thấy file {MSRVTT_JSON_PATH}")
        return
        
    with open(MSRVTT_JSON_PATH, 'r') as f:
        train_val_info = json.load(f)

    sentences_list = train_val_info['sentences']
    videos_list_train_val = train_val_info['videos']
    
    video_to_captions = {}
    for sentence in sentences_list:
        video_id = sentence['video_id']
        caption = sentence['caption']
        if video_id not in video_to_captions:
            video_to_captions[video_id] = []
        video_to_captions[video_id].append(caption)

    # Xử lý tập train
    train_samples = process_split(videos_list_train_val, video_to_captions, MSRVTT_VIDEO_FOLDER, 'train')
    train_file_path = os.path.join(PREPROCESSED_DATA_FOLDER, 'train_data.json')
    with open(train_file_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    print(f"\nĐã lưu {len(train_samples)} mẫu huấn luyện vào {train_file_path}")

    # Xử lý tập validation
    val_samples = process_split(videos_list_train_val, video_to_captions, MSRVTT_VIDEO_FOLDER, 'validate')
    val_file_path = os.path.join(PREPROCESSED_DATA_FOLDER, 'val_data.json')
    with open(val_file_path, 'w') as f:
        json.dump(val_samples, f, indent=2)
    print(f"\nĐã lưu {len(val_samples)} mẫu kiểm định vào {val_file_path}")

    # --- Xử lý tập Test ---
    print("\n--- Bắt đầu xử lý dữ liệu Test ---")
    if not os.path.exists(MSRVTT_TEST_JSON_PATH):
        print(f"Cảnh báo: Không tìm thấy file {MSRVTT_TEST_JSON_PATH}. Sẽ chỉ xử lý video.")
        # Nếu không có file json, ta tự tạo metadata từ danh sách file
        video_files_test = os.listdir(MSRVTT_TEST_VIDEO_FOLDER)
        videos_list_test = [{"video_id": f.split('.')[0], "split": "test"} for f in video_files_test if f.endswith('.mp4')]
        test_captions = {} # Không có caption
    else:
        with open(MSRVTT_TEST_JSON_PATH, 'r') as f:
            test_info = json.load(f)
        videos_list_test = test_info['videos']
        test_sentences_list = test_info.get('sentences', [])
        test_captions = {}
        for sentence in test_sentences_list:
            video_id = sentence['video_id']
            caption = sentence['caption']
            if video_id not in test_captions:
                test_captions[video_id] = []
            test_captions[video_id].append(caption)

    # Xử lý tập test
    test_samples = process_split(videos_list_test, test_captions, MSRVTT_TEST_VIDEO_FOLDER, 'test')
    test_file_path = os.path.join(PREPROCESSED_DATA_FOLDER, 'test_data.json')
    with open(test_file_path, 'w') as f:
        json.dump(test_samples, f, indent=2)
    print(f"\nĐã lưu {len(test_samples)} mẫu kiểm tra vào {test_file_path}")
    
    print("\n\n>>> TIỀN XỬ LÝ TOÀN BỘ DỮ LIỆU HOÀN TẤT! <<<")


# ==============================================================================
# PHẦN 3: CHẠY SCRIPT
# ==============================================================================

if __name__ == "__main__":
    main()