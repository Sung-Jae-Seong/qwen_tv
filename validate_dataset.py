#!/usr/bin/env python3
"""
Simple validation script to check if the dataset preparation works correctly.
This script doesn't require torch/transformers to be installed.
"""

import json
import os
from pathlib import Path


def validate_jsonl_file(jsonl_path: str, max_samples: int = 5):
    """Validate JSONL file structure."""
    print(f"Validating {jsonl_path}...")
    
    if not os.path.exists(jsonl_path):
        print(f"  ERROR: File not found: {jsonl_path}")
        return False
    
    valid_count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
                
                # Check structure
                required_keys = {'video', 'messages'}
                if not required_keys.issubset(sample.keys()):
                    print(f"  ERROR: Sample {idx} missing required keys: {required_keys}")
                    return False
                
                # Check messages structure
                messages = sample['messages']
                if not isinstance(messages, list) or len(messages) != 2:
                    print(f"  ERROR: Sample {idx} has invalid messages structure")
                    return False
                
                # Check user message
                user_msg = messages[0]
                if user_msg.get('role') != 'user':
                    print(f"  ERROR: Sample {idx} first message is not from user")
                    return False
                
                # Check assistant message
                asst_msg = messages[1]
                if asst_msg.get('role') != 'assistant':
                    print(f"  ERROR: Sample {idx} second message is not from assistant")
                    return False
                
                # Check assistant response is valid JSON
                asst_content = asst_msg.get('content', '')
                try:
                    gt_json = json.loads(asst_content)
                    # Check GT JSON structure
                    required_gt_keys = {'time', 'coordinate', 'type', 'why'}
                    if not required_gt_keys.issubset(gt_json.keys()):
                        print(f"  ERROR: Sample {idx} GT JSON missing required keys: {required_gt_keys}")
                        return False
                except json.JSONDecodeError:
                    print(f"  ERROR: Sample {idx} assistant content is not valid JSON")
                    return False
                
                valid_count += 1
                
            except json.JSONDecodeError as e:
                print(f"  ERROR: Sample {idx} is not valid JSON: {e}")
                return False
    
    print(f"  ✓ Validated {valid_count} samples successfully")
    
    # Count total lines
    with open(jsonl_path, 'r') as f:
        total_lines = sum(1 for _ in f if _.strip())
    
    print(f"  ✓ Total samples: {total_lines}")
    return True


def check_video_paths(jsonl_path: str, base_path: str = "."):
    """Check if video paths in JSONL actually exist."""
    print(f"\nChecking video paths in {jsonl_path}...")
    
    missing_count = 0
    found_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            
            sample = json.loads(line)
            video_path = sample.get('video', '')
            full_path = os.path.join(base_path, video_path)
            
            if os.path.exists(full_path):
                found_count += 1
            else:
                missing_count += 1
                if missing_count <= 3:  # Show first 3 missing
                    print(f"  WARNING: Video not found: {full_path}")
    
    print(f"  ✓ Found: {found_count}, Missing: {missing_count}")
    return missing_count == 0


def main():
    dataset_dir = "dataset/sim_dataset"
    train_file = os.path.join(dataset_dir, "train_sft.jsonl")
    val_file = os.path.join(dataset_dir, "val_sft.jsonl")
    
    print("=" * 60)
    print("Dataset Validation Report")
    print("=" * 60)
    
    success = True
    
    if os.path.exists(train_file):
        if not validate_jsonl_file(train_file):
            success = False
        if not check_video_paths(train_file, "."):
            print("  Note: Some video paths may not exist (expected for test mode)")
    else:
        print(f"ERROR: Train file not found: {train_file}")
        success = False
    
    if os.path.exists(val_file):
        print()
        if not validate_jsonl_file(val_file):
            success = False
    else:
        print(f"ERROR: Val file not found: {val_file}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All validations passed!")
    else:
        print("✗ Some validations failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
