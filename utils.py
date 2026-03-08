import os
from pathlib import Path
import json
import cv2

def make_train_video_path(input_file="/workspace/dataset/sim_dataset/videos", output_file="/workspace/dataset"):
    root = Path(input_file)
    video_paths = sorted(root.rglob("*.mp4"))
    save_path = Path(output_file) / "train_video_path.txt"

    with open(save_path, "w") as f:
        for p in video_paths:
            f.write(str(p) + "\n")

def make_test_video_path(input_file="/workspace/dataset/videos", output_file="/workspace/dataset"):
    root = Path(input_file)
    video_paths = sorted(root.rglob("*.mp4"))
    save_path = Path(output_file) / "test_video_path.txt"

    with open(save_path, "w") as f:
        for p in video_paths:
            f.write(str(p) + "\n")

def read_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 5.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return fps, width, height, frame_count


def get_sampled_frame_count(src_fps, total_source_frames, target_fps=5.0):
    if src_fps <= 0:
        src_fps = target_fps

    step = max(int(round(src_fps / target_fps)), 1)

    if total_source_frames <= 0:
        return 0

    return (total_source_frames - 1) // step + 1


def get_video_maximum_metadata(path_file, target_fps=5.0):
    with open(path_file, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    result = {
        "path_file": path_file,
        "target_fps": target_fps,
        "num_videos": len(video_paths),
        "max_width": 0,
        "max_height": 0,
        "max_source_frames": 0,
        "max_sampled_frames": 0,
        "max_duration_seconds": 0.0,
        "max_width_video": "",
        "max_height_video": "",
        "max_source_frames_video": "",
        "max_sampled_frames_video": "",
        "max_duration_video": "",
        "recommended_limit_mm_per_prompt": {
            "image": 0,
            "video": {
                "count": 1,
                "num_frames": 0,
                "width": 0,
                "height": 0
            }
        }
    }

    for video_path in video_paths:
        fps, width, height, frame_count = read_video_metadata(video_path)
        sampled_frames = get_sampled_frame_count(fps, frame_count, target_fps=target_fps)
        duration = frame_count / fps if fps > 0 else 0.0

        if width > result["max_width"]:
            result["max_width"] = width
            result["max_width_video"] = video_path

        if height > result["max_height"]:
            result["max_height"] = height
            result["max_height_video"] = video_path

        if frame_count > result["max_source_frames"]:
            result["max_source_frames"] = frame_count
            result["max_source_frames_video"] = video_path

        if sampled_frames > result["max_sampled_frames"]:
            result["max_sampled_frames"] = sampled_frames
            result["max_sampled_frames_video"] = video_path

        if duration > result["max_duration_seconds"]:
            result["max_duration_seconds"] = round(duration, 4)
            result["max_duration_video"] = video_path

    result["recommended_limit_mm_per_prompt"]["video"]["num_frames"] = result["max_sampled_frames"]
    result["recommended_limit_mm_per_prompt"]["video"]["width"] = result["max_width"]
    result["recommended_limit_mm_per_prompt"]["video"]["height"] = result["max_height"]

    return result


if __name__ == "__main__":
    os.makedirs("/workspace/dataset", exist_ok=True)

    make_train_video_path()
    make_test_video_path()

    # train_path_file = "/workspace/dataset/train_video_path.txt"
    test_path_file = "/workspace/dataset/test_video_path.txt"

    # train_meta = get_video_maximum_metadata(train_path_file, target_fps=5.0)
    test_meta = get_video_maximum_metadata(test_path_file, target_fps=5.0)

    save_path = "/workspace/dataset/video_maximum_metadata.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                # "train": train_meta,
                "test": test_meta
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(json.dumps({"test": test_meta}, ensure_ascii=False, indent=2))