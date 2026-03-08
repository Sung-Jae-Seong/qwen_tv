import os
import ast
import json
import time
import multiprocessing as mp
from datetime import datetime
import cv2
import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


def parse_in_json(llm_response, video_path):
    try:
        temp = ast.literal_eval(llm_response)
    except Exception:
        temp = {
            "time": 0.0,
            "coordinate": [[0, 0], [0, 0]],
            "type": "head-on",
            "why": "invalid response format"
        }

    if not isinstance(temp, dict):
        temp = {
            "time": 0.0,
            "coordinate": [[0, 0], [0, 0]],
            "type": "head-on",
            "why": "invalid response format"
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


class VideoInferenceVLM:
    def video_inference(self, video_path, prompt, max_new_tokens=128):
        raise NotImplementedError("Subclasses should implement this method.")


class Qwen3VLInference(VideoInferenceVLM):
    def __init__(self, model_id="Qwen/Qwen3-VL-8b-Instruct", device="cuda:0"):
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map={"": device}
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def video_inference(self, video_path, prompt, max_new_tokens=128):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 5,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text


def worker_process(gpu_id, start_idx, end_idx, video_paths, prompt, output_queue):
    try:
        torch.cuda.set_device(gpu_id)
        inference = Qwen3VLInference(device=f"cuda:{gpu_id}")
        results = []

        loop_start_time = time.time()

        for i in range(start_idx, end_idx):
            video_path = video_paths[i]
            cap = cv2.VideoCapture(video_path)
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            fps_info = f"The video has a frame rate of {fps} frames per second. It means that each indexed frame corresponds to {1000/fps:.2f} milliseconds. " \
                f"Use this information to calculate the exact time for the first collision frame based on its indexed frame number. "
            current_prompt = prompt + "\n" + fps_info
            output = inference.video_inference(
                video_path,
                current_prompt,
                max_new_tokens=256
            )
            output_json = parse_in_json(output[0], video_path)
            output_json["video_path"] = video_path
            results.append((i, output_json))
            print(f"gpu {gpu_id} - {i + 1}/{len(video_paths)} is done.", flush=True)


        elapsed_seconds = time.time() - loop_start_time

        output_queue.put({
            "gpu_id": gpu_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "elapsed_seconds": elapsed_seconds,
            "results": results,
            "error": None
        })
    except Exception as e:
        output_queue.put({
            "gpu_id": gpu_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "elapsed_seconds": 0.0,
            "results": [],
            "error": repr(e)
        })


def main():
    mp.set_start_method("spawn", force=True)

    instruction = """
This video is a CCTV-view traffic accident video.
The accident region is a really local area in the whole video so you have to analyze the corners and edges of the video carefully.
Follow the instructions below to analyze the traffic accident video and extract the accident frame (time), accident region, and accident type.

1. Analysis
You should watch the video end to end.
Analyze this video from behinning to end **frame by frame** and gather information about the traffic accident.
Focus mainly on the road and **vehicle movements**. Since the video may include low resolution, occlusion, low-light conditions, and similar challenges, analyze it carefully step by step.
Collision includes both collisions between different vehicles and collisions where a single vehicle hits a stationary object.
**Tracking the movement of vehicles** helps to find the collision moment.

2. Reasoning
Return briefly why did you decide like that.
It might be hard to detect the collision, so you might think there is no collision in the video but there must be a accident(collision) in the video.

3. Temporal Prediction
The video is represented as indexed frames in chronological order.
**Find the indexed frame** where physical contact between vehicles begins or single and the result of the accident after collisions.
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
please return the result in JSON format only, not markdown.
here is the JSON format:
{
    "reasoning ": "explain the situation of the video after accident occurs and why did you decide like that",
    "collision_frame": exact indexed frame where the collision occurs,
    "result_frame" : exact indexed frame after the collision occurs,
    "time": exact time corresponding to that indexed frame,
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "choose one from [head-on, rear-end, sideswipe, single, t-bone]",
}
---
example:
{
    "reasoning": "After carefully analyzing the video frame by frame, I observed that at frame 150, the front end of a red car made contact with the rear end of a blue car, which indicates a rear-end collision. The bounding box coordinates [100, 200], [300, 400] enclose the area where the two vehicles are in contact. The collision type is classified as rear-end because the red car hit the back of the blue car. After the collision, at frame 180, both vehicles came to a stop, which confirms that the accident occurred.",
    # do not answer like "There is no collision in the video." or "I observed that there is no visible collision"
    "collision_frame": int,
    "result_frame": int,
    "time": "second.millisecond", # without minutes like 00.00
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "choose one from [head-on, rear-end, sideswipe, single, t-bone]",
}
"""

    test_path_file = "dataset/test_video_path.txt"
    prompt = instruction + "\n" + return_format

    with open(test_path_file, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    total_videos = len(video_paths[:100])
    mid = total_videos // 2

    output_queue = mp.Queue()

    process_0 = mp.Process(
        target=worker_process,
        args=(0, 0, mid, video_paths, prompt, output_queue)
    )
    process_1 = mp.Process(
        target=worker_process,
        args=(1, mid, total_videos, video_paths, prompt, output_queue)
    )

    total_start_time = time.time()

    process_0.start()
    process_1.start()

    worker_outputs = [output_queue.get(), output_queue.get()]

    process_0.join()
    process_1.join()

    total_end_time = time.time()
    total_elapsed_seconds = total_end_time - total_start_time

    print(f"Total elapsed time: {total_elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
