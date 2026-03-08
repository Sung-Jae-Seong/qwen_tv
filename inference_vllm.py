import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import re
import ast
import json
import time
import cv2
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import resources.prompt as prompt_module

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

def default_result():
    return {
        "time": 0.0,
        "coordinate": [[0, 0], [0, 0]],
        "type": "head-on",
        "reasoning": "invalid response format",
        "collision_frame": 0,
        "result_frame": 0,
    }

def parse_in_json(llm_response, video_path):
    text = llm_response.strip()
    if not text.endswith("}"):
        if not text.endswith('"'):
            text += '"'
        text += "}"
    try:
        temp = ast.literal_eval(text)
    except Exception:
        temp = default_result()
    if not isinstance(temp, dict):
        temp = default_result()

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

    step = max(int(src_fps / target_fps), 1)

    frames = []
    sampled_indices = []
    idx = 0

    while True:
        if not cap.grab():
            break

        if idx % step != 0:
            idx += 1
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

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


    def _handling_cache_overflow(self, error, video_path, video_frames, metadata):
        error_text = str(error)

        if "exceeds the pre-allocated encoder cache size" not in error_text:
            raise error

        match = re.search(
            r"video item with length (\d+), which exceeds .* size (\d+)",
            error_text,
        )
        if not match:
            raise error

        current_length = int(match.group(1))
        cache_limit = int(match.group(2))
        current_frame_count = len(video_frames)

        if current_frame_count <= 1:
            raise error

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

        return video_frames, metadata


    def video_inference(
        self,
        video_path,
        prompt,
        max_new_tokens=128,
        video_frames=None,
        metadata=None,
    ):
        if video_frames is None or metadata is None:
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
                video_frames, metadata = self._handling_cache_overflow(
                    e,
                    video_path,
                    video_frames,
                    metadata,
                )


def build_prompt(fps, width, height):
    instruction = prompt_module.instruction
    return_format = prompt_module.return_format

    video_info = (
        f"The original video has a frame rate of {fps:.2f} frames per second. "
        f"Each original frame corresponds to {1000 / fps:.2f} milliseconds. "
        f"The video resolution is {width}x{height}. "
        f"Use this information to calculate the exact time and predict accurate bounding box coordinates."
    )

    return instruction + "\n" + video_info + "\n" + return_format


def main():
    total_start_time = time.time()
    test_path_file = "dataset/test_video_path.txt"
    max_videos = 10

    with open(test_path_file, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    video_paths = video_paths[:max_videos]

    os.makedirs("result", exist_ok=True)
    inference = Qwen3VLInference()

    all_results = []

    for i, video_path in enumerate(video_paths):
        video_frames, metadata = load_video_frames(video_path, target_fps=5.0)
        prompt = build_prompt(metadata["src_fps"], metadata["width"], metadata["height"])

        start_time = time.time()
        output = inference.video_inference(
            video_path=video_path,
            prompt=prompt,
            max_new_tokens=512,
            video_frames=video_frames,
            metadata=metadata,
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
