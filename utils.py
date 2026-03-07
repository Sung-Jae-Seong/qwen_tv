import os
from pathlib import Path

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

if __name__ == "__main__":
    make_train_video_path()
    make_test_video_path()