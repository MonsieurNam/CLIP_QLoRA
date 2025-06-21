import os
import json
import torch
import decord
from PIL import Image
from tqdm import tqdm

# Configure decord to return torch tensors
decord.bridge.set_bridge('torch')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Change these paths as needed
PROJECT_ROOT_PATH = "root/CLIP_QLoRA"

# Paths for MSR-VTT dataset
MSRVTT_DATA_PATH = PROJECT_ROOT_PATH
MSRVTT_VIDEO_FOLDER = os.path.join(MSRVTT_DATA_PATH, "train-val-video")
MSRVTT_TEST_VIDEO_FOLDER = "/root/CLIP_QLoRA/TestVideo"
MSRVTT_JSON_PATH = os.path.join(MSRVTT_DATA_PATH, "train_val_videodatainfo.json")
MSRVTT_TEST_JSON_PATH = os.path.join(MSRVTT_DATA_PATH, "test_videodatainfo.json")

# Folder to save preprocessed frames and JSON
PREPROCESSED_DATA_FOLDER = os.path.join(PROJECT_ROOT_PATH, "MSRVTT_Preprocessed_16frames")

# Number of frames to sample per video
NUM_FRAMES = 16

# =============================================================================
# UTILITY FUNCTION: process one split (train/validate/test)
# =============================================================================
def process_split(videos_metadata, video_captions, video_folder, split_name):
    """
    Process a dataset split: sample frames, save images, and pair with captions.
    videos_metadata: list of dicts, each contains 'video_id' and possibly 'split'
    video_captions: dict mapping video_id to list of captions (or empty)
    video_folder: path to folder containing video_id.mp4 files
    split_name: 'train', 'validate', or 'test'
    Returns: list of samples (dicts with video_id, frame_paths, caption)
    """
    processed_samples = []
    
    # Filter by split; for test, if metadata has no 'split', treat all as test
    if split_name != 'test':
        split_videos = [v for v in videos_metadata if v.get('split') == split_name]
    else:
        # If metadata items have 'split' field, filter; otherwise use all
        has_split_field = any('split' in v for v in videos_metadata)
        if has_split_field:
            split_videos = [v for v in videos_metadata if v.get('split') == 'test']
        else:
            split_videos = videos_metadata[:]
    print(f"\nBắt đầu xử lý {len(split_videos)} video cho tập '{split_name}'...")
    
    # If video folder does not exist, skip processing
    if not os.path.isdir(video_folder):
        print(f"[WARN] Folder video cho tập '{split_name}' không tồn tại: {video_folder}. Bỏ qua split này.")
        return processed_samples

    for video_meta in tqdm(split_videos, desc=f"Đang xử lý tập {split_name}"):
        video_id = video_meta.get('video_id')
        if not video_id:
            continue
        video_path = os.path.join(video_folder, f"{video_id}.mp4")

        if not os.path.isfile(video_path):
            print(f"  [WARN] Bỏ qua: video không tồn tại: {video_path}")
            continue

        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                print(f"  [WARN] Video rỗng (0 frames): {video_id}")
                continue
            
            # Sample indices evenly
            indices = torch.linspace(0, total_frames - 1, NUM_FRAMES, dtype=torch.long)
            frames = vr.get_batch(indices)

            # Prepare output folder for frames
            video_frame_folder = os.path.join(PREPROCESSED_DATA_FOLDER, video_id)
            os.makedirs(video_frame_folder, exist_ok=True)
            
            frame_paths = []
            for i, frame_tensor in enumerate(frames):
                # Convert to PIL Image
                frame_image = Image.fromarray(frame_tensor.numpy())
                relative_path = os.path.join(video_id, f"frame_{i:04d}.jpg")
                full_path = os.path.join(PREPROCESSED_DATA_FOLDER, relative_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                frame_image.save(full_path)
                frame_paths.append(relative_path)

            # Pair with captions (or empty)
            captions = video_captions.get(video_id) if video_captions else None
            if captions:
                for caption in captions:
                    sample = {
                        "video_id": video_id,
                        "frame_paths": frame_paths,
                        "caption": caption
                    }
                    processed_samples.append(sample)
            else:
                sample = {
                    "video_id": video_id,
                    "frame_paths": frame_paths,
                    "caption": ""
                }
                processed_samples.append(sample)

        except Exception as e:
            print(f"  [ERROR] Lỗi khi xử lý video {video_id}: {e}")

    return processed_samples

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    # Ensure output folder exists
    try:
        os.makedirs(PREPROCESSED_DATA_FOLDER, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Không thể tạo thư mục PREPROCESSED_DATA_FOLDER: {PREPROCESSED_DATA_FOLDER}, lỗi: {e}")
        return

    # --- Process Train/Validation ---
    print("--- Bắt đầu xử lý dữ liệu Train/Validation ---")
    train_val_samples = []
    sentences_list = []
    videos_list_train_val = []

    if os.path.isfile(MSRVTT_JSON_PATH):
        try:
            with open(MSRVTT_JSON_PATH, 'r') as f:
                train_val_info = json.load(f)
            sentences_list = train_val_info.get('sentences', [])
            videos_list_train_val = train_val_info.get('videos', [])
            if not videos_list_train_val:
                print(f"[WARN] JSON train/val có, nhưng key 'videos' rỗng hoặc không tồn tại.")
        except Exception as e:
            print(f"[ERROR] Không thể đọc JSON train/val tại {MSRVTT_JSON_PATH}: {e}")
    else:
        print(f"[WARN] File JSON train/val không tồn tại: {MSRVTT_JSON_PATH}. Bỏ qua train/val nếu không có dữ liệu metadata.")

    # Build video_to_captions dict if possible
    video_to_captions = {}
    for sentence in sentences_list:
        vid = sentence.get('video_id')
        cap = sentence.get('caption', "")
        if vid:
            video_to_captions.setdefault(vid, []).append(cap)

    # Process train split
    if videos_list_train_val:
        train_samples = process_split(videos_list_train_val, video_to_captions, MSRVTT_VIDEO_FOLDER, 'train')
        train_file_path = os.path.join(PREPROCESSED_DATA_FOLDER, 'train_data.json')
        try:
            with open(train_file_path, 'w') as f:
                json.dump(train_samples, f, indent=2)
            print(f"Đã lưu {len(train_samples)} mẫu huấn luyện vào {train_file_path}")
        except Exception as e:
            print(f"[ERROR] Không thể lưu train_data.json: {e}")
    else:
        print("[INFO] Không có metadata train để xử lý.")

    # Process validation split
    if videos_list_train_val:
        val_samples = process_split(videos_list_train_val, video_to_captions, MSRVTT_VIDEO_FOLDER, 'validate')
        val_file_path = os.path.join(PREPROCESSED_DATA_FOLDER, 'val_data.json')
        try:
            with open(val_file_path, 'w') as f:
                json.dump(val_samples, f, indent=2)
            print(f"Đã lưu {len(val_samples)} mẫu kiểm định vào {val_file_path}")
        except Exception as e:
            print(f"[ERROR] Không thể lưu val_data.json: {e}")
    else:
        print("[INFO] Không có metadata validation để xử lý.")

    # --- Process Test ---
    print("\n--- Bắt đầu xử lý dữ liệu Test ---")
    videos_list_test = []
    test_captions = {}

    if os.path.isfile(MSRVTT_TEST_JSON_PATH):
        try:
            with open(MSRVTT_TEST_JSON_PATH, 'r') as f:
                test_info = json.load(f)
            videos_list_test = test_info.get('videos', [])
            test_sentences_list = test_info.get('sentences', [])
            for sentence in test_sentences_list:
                vid = sentence.get('video_id')
                cap = sentence.get('caption', "")
                if vid:
                    test_captions.setdefault(vid, []).append(cap)
            if not videos_list_test:
                print(f"[WARN] JSON test có, nhưng key 'videos' rỗng hoặc không tồn tại.")
        except Exception as e:
            print(f"[ERROR] Không thể đọc JSON test tại {MSRVTT_TEST_JSON_PATH}: {e}")
    else:
        # If no JSON test, list mp4 in test folder
        if os.path.isdir(MSRVTT_TEST_VIDEO_FOLDER):
            try:
                files = os.listdir(MSRVTT_TEST_VIDEO_FOLDER)
                videos_list_test = [{"video_id": f.split('.')[0], "split": "test"} 
                                    for f in files if f.lower().endswith('.mp4')]
                if not videos_list_test:
                    print(f"[WARN] Thư mục test video tồn tại nhưng không có file .mp4: {MSRVTT_TEST_VIDEO_FOLDER}")
            except Exception as e:
                print(f"[ERROR] Không thể liệt kê test video folder: {e}")
        else:
            print(f"[WARN] Folder test video không tồn tại: {MSRVTT_TEST_VIDEO_FOLDER}. Bỏ qua test.")

    # Process test split if any videos found
    if videos_list_test:
        test_samples = process_split(videos_list_test, test_captions, MSRVTT_TEST_VIDEO_FOLDER, 'test')
        test_file_path = os.path.join(PREPROCESSED_DATA_FOLDER, 'test_data.json')
        try:
            with open(test_file_path, 'w') as f:
                json.dump(test_samples, f, indent=2)
            print(f"Đã lưu {len(test_samples)} mẫu kiểm tra vào {test_file_path}")
        except Exception as e:
            print(f"[ERROR] Không thể lưu test_data.json: {e}")
    else:
        print("[INFO] Không có videos test để xử lý.")

    print("\n>>> TIỀN XỬ LÝ TOÀN BỘ DỮ LIỆU HOÀN TẤT! <<<")

if __name__ == "__main__":
    main()
