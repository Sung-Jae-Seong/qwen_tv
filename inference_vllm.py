import os
import re
import ast
import json
import time
import cv2
from PIL import Image
from dotenv import load_dotenv

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from huggingface_hub import login
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def parse_in_json(llm_response, video_path):
    text = llm_response.strip()

    try:
        temp = json.loads(text)
    except Exception:
        try:
            repaired = text
            if not repaired.endswith("}"):
                if not repaired.endswith('"'):
                    repaired += '"'
                repaired += "}"
            temp = ast.literal_eval(repaired)
        except Exception:
            temp = {
                "time": 0.0,
                "coordinate": [[0, 0], [0, 0]],
                "type": "head-on",
                "reasoning": "invalid response format",
                "collision_frame": 0,
                "result_frame": 0
            }

    if not isinstance(temp, dict):
        temp = {
            "time": 0.0,
            "coordinate": [[0, 0], [0, 0]],
            "type": "head-on",
            "reasoning": "invalid response format",
            "collision_frame": 0,
            "result_frame": 0
        }

    save_dir = os.path.join("result", "parsed_json")
    os.makedirs(save_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_dir, f"{video_name}.json")

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(temp, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"failed to save parsed json for {video_path}: {e}", flush=True)

    return temp


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


def load_video_frames(video_path, target_fps=5.0):
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = target_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(int(round(src_fps / target_fps)), 1)

    frames = []
    sampled_indices = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            sampled_indices.append(idx)
        idx += 1

    cap.release()

    metadata = {
        "src_fps": src_fps,
        "width": width,
        "height": height,
        "total_source_frames": total_source_frames,
        "sampled_indices": sampled_indices,
        "target_fps": target_fps
    }
    return frames, metadata


class VideoInferenceVLM:
    def video_inference(self, video_path, prompt, max_new_tokens=128):
        raise NotImplementedError("Subclasses should implement this method.")


class Qwen3VLInference(VideoInferenceVLM):
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", max_encoder_cache_size=12288):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_encoder_cache_size = max_encoder_cache_size
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=2,
            dtype="float16",
            max_model_len=16384,
            max_num_seqs=1,
            gpu_memory_utilization=0.92,
            mm_encoder_tp_mode="data",
            limit_mm_per_prompt={"image": 0, "video": 1},
            mm_processor_cache_gb=0,
            enforce_eager=False,
        )

    def _build_llm_input(self, video_frames, metadata, prompt):
        total_num_sampled_frames = len(video_frames)

        if metadata["src_fps"] > 0 and metadata["total_source_frames"] > 0:
            duration = metadata["total_source_frames"] / metadata["src_fps"]
        else:
            duration = total_num_sampled_frames / metadata["target_fps"] if metadata["target_fps"] > 0 else 0.0

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": text_prompt,
            "multi_modal_data": {
                "video": (
                    video_frames,
                    {
                        "fps": metadata["target_fps"],
                        "total_num_frames": metadata["total_source_frames"] if metadata["total_source_frames"] > 0 else total_num_sampled_frames,
                        "duration": duration,
                        "video_backend": "opencv",
                        "frames_indices": metadata["sampled_indices"] if metadata["sampled_indices"] else list(range(total_num_sampled_frames)),
                        "do_sample_frames": False,
                    },
                ),
            },
        }

    def _trim_video_to_frame_count(self, video_frames, metadata, keep_count):
        keep_count = max(1, min(keep_count, len(video_frames)))
        trimmed_frames = video_frames[:keep_count]
        trimmed_metadata = dict(metadata)
        if metadata.get("sampled_indices"):
            trimmed_metadata["sampled_indices"] = metadata["sampled_indices"][:keep_count]
        return trimmed_frames, trimmed_metadata

    def video_inference(self, video_path, prompt, max_new_tokens=128):
        video_frames, metadata = load_video_frames(video_path, target_fps=5.0)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        while True:
            try:
                llm_input = self._build_llm_input(video_frames, metadata, prompt)
                outputs = self.llm.generate(
                    llm_input,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                return [outputs[0].outputs[0].text]

            except ValueError as e:
                error_text = str(e)
                if "exceeds the pre-allocated encoder cache size" not in error_text:
                    raise

                match = re.search(r"video item with length (\d+), which exceeds .* size (\d+)", error_text)
                if not match:
                    raise

                current_length = int(match.group(1))
                cache_limit = int(match.group(2))
                current_frame_count = len(video_frames)

                if current_frame_count <= 1:
                    raise

                keep_count = int(current_frame_count * cache_limit / current_length) - 1
                keep_count = max(1, min(keep_count, current_frame_count - 1))

                print(
                    f"[trim] {os.path.basename(video_path)}: cache overflow "
                    f"({current_length} > {cache_limit}), trimming frames "
                    f"{current_frame_count} -> {keep_count}",
                    flush=True,
                )

                video_frames, metadata = self._trim_video_to_frame_count(
                    video_frames,
                    metadata,
                    keep_count,
                )


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
"""

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
"""

    video_info = (
        f"The original video has a frame rate of {fps:.2f} frames per second. "
        f"Each original frame corresponds to {1000 / fps:.2f} milliseconds. "
        f"The video resolution is {width}x{height}. "
        f"Use this information to calculate the exact time and predict accurate bounding box coordinates."
    )

    return instruction + "\n" + video_info + "\n" + return_format


def main():
    test_path_file = "dataset/test_video_path.txt"
    max_videos = 100

    with open(test_path_file, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    video_paths = video_paths[60:max_videos]

    os.makedirs("result", exist_ok=True)
    inference = Qwen3VLInference()

    total_start_time = time.time()
    all_results = []

    for i, video_path in enumerate(video_paths):
        fps, width, height, _ = read_video_metadata(video_path)
        prompt = build_prompt(fps, width, height)

        start_time = time.time()
        output = inference.video_inference(
            video_path=video_path,
            prompt=prompt,
            max_new_tokens=512
        )
        elapsed = time.time() - start_time

        parsed = parse_in_json(output[0], video_path)
        parsed["video_path"] = video_path
        parsed["inference_seconds"] = round(elapsed, 2)
        all_results.append(parsed)

        print(f"{i + 1}/{len(video_paths)} done - {video_path} - {elapsed:.2f} sec", flush=True)

    total_elapsed_seconds = time.time() - total_start_time

    save_path = os.path.join("result", "all_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Total elapsed time: {total_elapsed_seconds:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()