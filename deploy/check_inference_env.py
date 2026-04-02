#!/usr/bin/env python3

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def import_and_report(module_name: str, attr_names: list[str] | None = None) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        if attr_names:
            for attr_name in attr_names:
                getattr(module, attr_name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print("== Inference Environment Check ==")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"cwd: {os.getcwd()}")
    print(f"bundle_root: {ROOT_DIR}")

    required = [
        ("pandas", None),
        ("torch", None),
        ("transformers", ["AutoProcessor", "AutoModelForImageTextToText"]),
        ("qwen_vl_utils", ["process_vision_info"]),
        ("inference_qwen35_9b_base_video_classification", None),
        ("inference_qwen35_9b_base_video_classification_two_stage", None),
    ]
    optional = [
        ("av", None),
        ("decord", None),
    ]

    failed = []

    print("\n[required modules]")
    for module_name, attrs in required:
        ok, detail = import_and_report(module_name, attrs)
        status = "OK" if ok else "FAIL"
        print(f"{status:>4}  {module_name}: {detail}")
        if not ok:
            failed.append(module_name)

    print("\n[optional video backends]")
    optional_success = 0
    for module_name, attrs in optional:
        ok, detail = import_and_report(module_name, attrs)
        status = "OK" if ok else "WARN"
        print(f"{status:>4}  {module_name}: {detail}")
        if ok:
            optional_success += 1

    if optional_success == 0:
        failed.append("av/decord")

    try:
        import torch

        print("\n[cuda]")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"visible_device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            for index in range(torch.cuda.device_count()):
                print(f"cuda_device_{index}: {torch.cuda.get_device_name(index)}")
    except Exception as exc:
        print("\n[cuda]")
        print(f"cuda_check_failed: {type(exc).__name__}: {exc}")
        failed.append("torch.cuda")

    if failed:
        print("\nEnvironment check failed.")
        print("Missing or broken components:", ", ".join(failed))
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
