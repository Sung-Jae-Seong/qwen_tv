import os
import ast
import json
import time
from pathlib import Path

import cv2
from PIL import Image
from dotenv import load_dotenv

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from huggingface_hub import login
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


RESULT_DIR = Path("result")
PARSED_JSON_DIR = RESULT_DIR / "parsed_json"
TEST_PATH_FILE = Path("dataset/test_video_path.txt")
DEFAULT_TARGET_FPS = 5.0
DEFAULT_MAX_VIDEOS = 100


def default_result():
    return {
        "time": 0.0,
        "coordinate": [[0, 0], [0, 0]],
        "type": "head-on",
        "reasoning": "invalid response format",
        "collision_frame": 0,
        "result_frame": 0,
    }


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_in_json(llm_response, video_path):
    text = llm_response.strip()
    parsed = None

    try:
        parsed = json.loads(text)
    except Exception:
        try:
            repaired = text
            if not repaired.endswith("}"):
                if not repaired.endswith('"'):
                    repaired += '"'
                repaired += "}"
            parsed = ast.literal_eval(repaired)
        except Exception:
            parsed = default_result()

    if not isinstance(parsed, dict):
        parsed = default_result()

    video_name = Path(video_path).stem
    save_path = PARSED_JSON_DIR / f"{video_name}.json"

    try:
        save_json(parsed, save_path)
    except Exception as e:
        print(f"failed to save parsed json for {video_path}: {e}", flush=True)

    return parsed


def load_video_frames_and_metadata(video_path, target_fps=DEFAULT_TARGET_FPS):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = target_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(int(src_fps / target_fps), 1)

    frames = []
    sampled_indices = []
    idx = 0

    while True:
        grabbed = cap.grab()
        if not grabbed:
            break

        if idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            sampled_indices.append(idx)

        idx += 1

    cap.release()

    metadata = {
        "src_fps": float(src_fps),
        "width": width,
        "height": height,
        "total_source_frames": total_source_frames,
        "sampled_indices": sampled_indices,
        "target_fps": float(target_fps),
    }
    return frames, metadata


def build_prompt(fps, width, height):
    instruction = """
This video is a CCTV-view traffic accident video.
The accident region is a really local area in the whole video so you have to analyze the corners and edges of the video carefully.
Follow the instructions below to analyze the traffic accident video and extract the accident frame (time), accident region, and accident type.

1. Analysis
You should watch the video end to end.
Analyze this video from behinning to end frame by frame and gather information about the traffic accident.
Focus mainly on the road and vehicle movements. Since the video may include low resolution, occlusion, low-light conditions, and similar challenges, analyze it carefully step by step.
Collision includes both collisions between different vehicles and collisions where a single vehicle hits a stationary object.
Tracking the movement of vehicles helps to find the collision moment.

2. Reasoning
Return briefly why did you decide like that.
It might be hard to detect the collision, so you might think there is no collision in the video but there must be a accident(collision) in the video.

3. Temporal Prediction
The video is represented as indexed frames in chronological order.
Find the indexed frame where physical contact between vehicles begins or single and the result of the accident after collisions.
Return that two frame index (collision_frame and result_frame).
Then return the corresponding time for that indexed collision_frame.

4. Spatial Prediction
Also return one collision bounding box on that indexed collision_frame using left-top and right-bottom coordinates.
The bbox area include one or two vehicles with collisions directly occurring
The bounding box should enclose the collision region or the involved vehicles at the first contact moment.
If one of the collided vehicles is occluded by other structures, predict the region to include the occluded vehicle as well.
The bounding box must contain at least one vehicle.

5. Type Prediction
Then return the collision type. The collision type includes: head-on, rear-end, sideswipe, single, and t-bone collisions.
Head-on is defined as a collision where the front ends of two vehicles hit each other.
Rear-end is defined as a collision where the front end of one vehicle hits the rear end of another vehicle.
Sideswipe is defined as a slight collision where the sides of two vehicles hit each other.
Single is defined as an accident that involves only one vehicle, such as a vehicle hitting a stationary object or a vehicle losing control and crashing without colliding with another vehicle.
T-bone is defined as a collision where the front end of one vehicle hits the side of another vehicle, forming a 'T' shape.
""".strip()

    return_format = """
Please return the result in JSON format only, not markdown.
Here is the JSON format:
{
    "reasoning": "explain the situation of the video after accident occurs and why did you decide like that",
    "collision_frame": exact indexed frame where the collision occurs,
    "result_frame": exact indexed frame after the collision occurs,
    "time": exact time corresponding to that indexed frame,
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "choose one from [head-on, rear-end, sideswipe, single, t-bone]"
}

Example:
{
    "reasoning": "After carefully analyzing the video frame by frame, I observed that at frame 150, the front end of a red car made contact with the rear end of a blue car, which indicates a rear-end collision. The bounding box coordinates [100, 200], [300, 400] enclose the area where the two vehicles are in contact. The collision type is classified as rear-end because the red car hit the back of the blue car. After the collision, at frame 180, both vehicles came to a stop, which confirms that the accident occurred.",
    "collision_frame": 150,
    "result_frame": 180,
    "time": "5.00",
    "coordinate": [
        [100, 200],
        [300, 400]
    ],
    "type": "rear-end"
}
""".strip()

    video_info = (
        f"The original video has a frame rate of {fps:.2f} frames per second. "
        f"Each original frame corresponds to {1000 / fps:.2f} milliseconds. "
        f"The video resolution is {width}x{height}. "
        f"Use this information to calculate the exact time and predict accurate bounding box coordinates."
    )

    return "\n\n".join([instruction, video_info, return_format])


def compute_duration(metadata, total_num_sampled_frames):
    src_fps = metadata["src_fps"]
    total_source_frames = metadata["total_source_frames"]
    target_fps = metadata["target_fps"]

    if src_fps > 0 and total_source_frames > 0:
        return total_source_frames / src_fps
    if target_fps > 0:
        return total_num_sampled_frames / target_fps
    return 0.0


def build_messages(prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_video_paths(path_file, max_videos):
    with open(path_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()][:max_videos]


class VideoInferenceVLM:
    def video_inference(self, video_path, max_new_tokens=128):
        raise NotImplementedError("Subclasses should implement this method.")


class Qwen3VLInference(VideoInferenceVLM):
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", target_fps=DEFAULT_TARGET_FPS):
        self.model_id = model_id
        self.target_fps = target_fps
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=2,
            dtype="float16",
            max_model_len=16384,
            max_num_seqs=1,
            gpu_memory_utilization=0.92,
            mm_encoder_tp_mode="data",
            limit_mm_per_prompt={
                "image": 0,
                "video": {
                    "count": 1,
                    "num_frames": 537,
                    "width": 3840,
                    "height": 2160
                }
            },
            
            mm_processor_cache_gb=0,
            enforce_eager=False,
        )

    def video_inference(self, video_path, max_new_tokens=128):
        video_frames, metadata = load_video_frames_and_metadata(
            video_path=video_path,
            target_fps=self.target_fps,
        )

        total_num_sampled_frames = len(video_frames)
        duration = compute_duration(metadata, total_num_sampled_frames)
        prompt = build_prompt(
            fps=metadata["src_fps"],
            width=metadata["width"],
            height=metadata["height"],
        )
        messages = build_messages(prompt)

        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        outputs = self.llm.generate(
            {
                "prompt": text_prompt,
                "multi_modal_data": {
                    "video": (
                        video_frames,
                        {
                            "fps": metadata["target_fps"],
                            "total_num_frames": metadata["total_source_frames"]
                            if metadata["total_source_frames"] > 0
                            else total_num_sampled_frames,
                            "duration": duration,
                            "video_backend": "opencv",
                            "frames_indices": metadata["sampled_indices"]
                            if metadata["sampled_indices"]
                            else list(range(total_num_sampled_frames)),
                            "do_sample_frames": False,
                        },
                    ),
                },
            },
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        text = outputs[0].outputs[0].text
        return text, metadata


def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    ensure_dir(RESULT_DIR)
    ensure_dir(PARSED_JSON_DIR)

    video_paths = load_video_paths(TEST_PATH_FILE, DEFAULT_MAX_VIDEOS)
    inference = Qwen3VLInference()

    total_start_time = time.time()
    all_results = []

    for i, video_path in enumerate(video_paths, start=1):
        start_time = time.time()

        try:
            raw_text, metadata = inference.video_inference(
                video_path=video_path,
                max_new_tokens=512,
            )
            parsed = parse_in_json(raw_text, video_path)
        except Exception as e:
            parsed = default_result()
            parsed["reasoning"] = f"inference failed: {str(e)}"

        elapsed = time.time() - start_time

        parsed["video_path"] = video_path
        parsed["inference_seconds"] = round(elapsed, 2)
        all_results.append(parsed)

        print(f"{i}/{len(video_paths)} done - {video_path} - {elapsed:.2f} sec", flush=True)

    total_elapsed_seconds = time.time() - total_start_time
    save_json(all_results, RESULT_DIR / "all_results.json")

    print(f"Total elapsed time: {total_elapsed_seconds:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()