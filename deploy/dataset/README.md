# Dataset Notes

This deploy bundle expects the default input list at:

```text
dataset/test_video_path.txt
```

Recommended layout:

```text
dataset/
├── test_video_path.txt
└── videos/
    ├── sample1.mp4
    └── sample2.mp4
```

`test_video_path.txt` can contain:

- absolute paths, or
- relative paths like `videos/sample1.mp4`

If you use a base submission CSV instead of a path list, you can place it anywhere and pass it with `--base_submission_csv`.
