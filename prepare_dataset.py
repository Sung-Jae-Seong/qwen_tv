"""
Prepare SFT dataset for Qwen3-VL fine-tuning on CCTV accident detection.

This script reads labels.csv and video_annotations JSON files to create
JSONL format dataset for supervised fine-tuning.
"""

import os
import json
import gzip
import argparse
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm





def load_annotation(annotation_path: str, base_path: str = ".") -> Dict[str, Any]:
    """Load annotation JSON file (handles both .json and .json.gz)."""
    full_path = os.path.join(base_path, annotation_path)
    
    # Handle .json.gz files - they are stored as directories with .json inside
    if full_path.endswith('.gz'):
        # Replace .json.gz with .json to get directory path
        json_dir = full_path.replace('.json.gz', '.json')
        
        # List files in directory and find the .json file
        if os.path.isdir(json_dir):
            for f in os.listdir(json_dir):
                if f.endswith('.json'):
                    full_path = os.path.join(json_dir, f)
                    break
        else:
            # Try alternative: look for the directory as-is
            alt_dir = full_path.replace('.gz', '')
            if os.path.isdir(alt_dir):
                for f in os.listdir(alt_dir):
                    if f.endswith('.json'):
                        full_path = os.path.join(alt_dir, f)
                        break
    
    try:
        with open(full_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Return empty annotation if file not found
        return {"base": [], "collision": []}


def generate_why_description(
    annotation: Dict[str, Any],
    collision_bbox: List[List[int]],
    vehicle_ids: List[int],
    accident_type: str
) -> str:
    """Generate explanation for why this is the detected collision."""
    
    if not annotation or "collision" not in annotation or not annotation["collision"]:
        return f"Detected {accident_type} collision based on visual analysis of the video."
    
    collision_info = annotation["collision"][0]  # Use first collision info
    iteration = collision_info.get("iteration", "unknown")
    ids = collision_info.get("ids", [])
    
    # Find corresponding base information at collision frame
    base_info = {}
    if "base" in annotation:
        for frame_data in annotation["base"]:
            if frame_data.get("iteration") == iteration:
                base_info = frame_data
                break
    
    # Build description
    why_parts = []
    
    if ids:
        id_str = ", ".join(str(id) for id in ids)
        why_parts.append(f"Vehicles (ID {id_str}) collided")
    else:
        why_parts.append("Collision detected")
    
    why_parts.append(f"at frame {iteration}")
    
    # Add bbox information
    if collision_bbox and len(collision_bbox) == 2:
        [[x1, y1], [x2, y2]] = collision_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        why_parts.append(f"in the frame region around ({center_x:.0f}, {center_y:.0f})")
    
    why_parts.append(f"Type: {accident_type}")
    
    return ". ".join(why_parts) + "."


def create_sft_sample(
    row: pd.Series,
    base_path: str,
    dataset_split: str = "train"
) -> Dict[str, Any]:
    """Create a single SFT sample for one accident video."""
    
    # Instruction prompt (from inference.py)
    instruction = (
        "return the time when the collision occurs, "
        "and return the collision bounding box with left-top and right-bottom coordinates. "
        "And return the collision type. The collision type includes: head-on, rear-end, sideswipe, single, "
        "and t-bone collisions. "
        "Head-on is defined as a collision where the front ends of two vehicles hit each other. "
        "Rear-end is defined as a collision where the front end of one vehicle hits the rear end of another vehicle. "
        "Sideswipe is defined as a slight collision where the sides of two vehicles hit each other. "
        "Single is defined as an accident that involves only one vehicle, such as a vehicle hitting a stationary object "
        "or a vehicle losing control and crashing without colliding with another vehicle. "
        "T-bone is defined as a collision where the front end of one vehicle hits the side of another vehicle, forming a 'T' shape."
    )
    
    return_format = (
        "please return the result in JSON format only, not markdown.\n"
        "here is the JSON format:\n"
        "{\n"
        '    "time": exact time in seconds, the temporal location of the video where collision occured,\n'
        '    "coordinate": left-top and right-bottom, the position of bounding box on the video frame that contains the collision,\n'
        '    "type": choose and return one of the following [head-on, rear-end, sideswipe, single, t-bone],\n'
        '    "why": explain why did you return that time, coordinate and type.\n'
        "}\n"
        "---\n"
        "example:\n"
        "{\n"
        '    "time": seconds,\n'
        '    "coordinate": [\n'
        "        [x1, y1],\n"
        "        [x2, y2]\n"
        "    ],\n"
        '    "type": "one of the following [head-on, rear-end, sideswipe, single, t-bone]",\n'
        '    "why": "..."\n'
        "}"
    )
    
    prompt = instruction + "\n" + return_format
    
    # Load annotation to get collision details for GT answer
    annotation = load_annotation(row['annotations_path'], base_path)
    
    # Create GT answer based on labels.csv and annotation
    accident_time_val = row['accident_time']
    
    # Coordinates in GT: multiply by 1000 to match inference.py format
    x1 = int(row['x1'] * 1000)
    y1 = int(row['y1'] * 1000)
    x2 = int(row['x2'] * 1000)
    y2 = int(row['y2'] * 1000)
    
    # Get collision bbox from annotation for why description
    collision_bbox = None
    vehicle_ids = []
    if annotation and "collision" in annotation and annotation["collision"]:
        collision_bbox = annotation["collision"][0].get("collision_bbox")
        vehicle_ids = annotation["collision"][0].get("ids", [])
    
    # Generate why description
    why = generate_why_description(annotation, collision_bbox, vehicle_ids, row['type'])
    
    gt_answer = {
        "time": accident_time_val,
        "coordinate": [[x1, y1], [x2, y2]],
        "type": row['type'],
        "why": why
    }
    
    # Construct video path
    video_path = os.path.join(base_path, row['rgb_path'])
    
    # Create SFT message pair
    sample = {
        "video": video_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
            {
                "role": "assistant",
                "content": json.dumps(gt_answer, ensure_ascii=False)
            }
        ]
    }
    
    return sample


def prepare_dataset(
    labels_csv_path: str,
    output_dir: str,
    base_path: str = ".",
    train_ratio: float = 0.9,
    max_samples: int = None,
    verbose: bool = True
) -> None:
    """
    Prepare SFT dataset from labels.csv and annotations.
    
    Args:
        labels_csv_path: Path to labels.csv
        output_dir: Output directory for JSONL files
        base_path: Base path for video and annotation files
        train_ratio: Ratio of training to total samples (rest is validation)
        max_samples: Maximum number of samples to process (for testing)
        verbose: Print progress information
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read labels
    df = pd.read_csv(labels_csv_path)
    
    if max_samples:
        df = df.head(max_samples)
    
    if verbose:
        print(f"Total samples in labels.csv: {len(df)}")
    
    # Split into train/val
    n_train = int(len(df) * train_ratio)
    train_df = df[:n_train]
    val_df = df[n_train:]
    
    if verbose:
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Prepare train dataset
    train_samples = []
    if verbose:
        print("\nPreparing training samples...")
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), disable=not verbose):
        try:
            sample = create_sft_sample(row, base_path, "train")
            train_samples.append(sample)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to process row {idx}: {e}")
            continue
    
    # Prepare val dataset
    val_samples = []
    if verbose:
        print("\nPreparing validation samples...")
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), disable=not verbose):
        try:
            sample = create_sft_sample(row, base_path, "val")
            val_samples.append(sample)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to process row {idx}: {e}")
            continue
    
    # Save to JSONL files
    train_output = os.path.join(output_dir, "train_sft.jsonl")
    val_output = os.path.join(output_dir, "val_sft.jsonl")
    
    if verbose:
        print(f"\nSaving training samples to {train_output}...")
    
    with open(train_output, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    if verbose:
        print(f"Saving validation samples to {val_output}...")
    
    with open(val_output, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    if verbose:
        print(f"\nDataset preparation complete!")
        print(f"Train: {len(train_samples)} samples -> {train_output}")
        print(f"Val: {len(val_samples)} samples -> {val_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT dataset for Qwen3-VL fine-tuning"
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="dataset/sim_dataset/labels.csv",
        help="Path to labels.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/sim_dataset",
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=".",
        help="Base path for video and annotation files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of training to total samples"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        labels_csv_path=args.labels_csv,
        output_dir=args.output_dir,
        base_path=args.base_path,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
        verbose=True
    )


if __name__ == "__main__":
    main()
